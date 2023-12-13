import re
import os
import pandas as pd
import numpy as np
import time
from functools import reduce
from owslib.util import ServiceException
from owslib.wms import WebMapService
from datetime import datetime, timedelta
import pytz

import asyncio
import aiohttp

wms_url = 'https://geo.weather.gc.ca/geomet/?SERVICE=WMS&REQUEST=GetFeatureInfo'
wms = WebMapService(wms_url , version='1.3.0', timeout=300)
common_var_names = ['TT', 'HR', 'PR', 'N4'] # These are the common variable names between the different forecast models
forecast_variables = ["AIRTEMP", "HR", "RAIN","GLOBALRAD"] # This is the desired variable names for the final dataframe

# Async version of the request function
async def async_request(session:aiohttp.ClientSession,layer: str, timestep: datetime, coor: list) -> float:
    # Create an aiohttp client session for making HTTP requests
    # Iterate through each timestep
    print(timestep, layer)
    # Get the request URL using the OWSLib library
    try :
        url = wms.getfeatureinfo(layers=[layer],
                                 srs='EPSG:4326',
                                 bbox=tuple(coor),
                                 size=(100, 100),
                                 format='image/jpeg',
                                 query_layers=[layer],
                                 info_format='text/plain',
                                 xy=(1, 1),
                                 feature_count=1,
                                 time=str(timestep.isoformat()) + 'Z'
                                 ).geturl()
        print(url)
        # Make an asynchronous GET request to the URL
        async with session.get(url) as resp:
            # Check if the response status is OK (200)
            if resp.status == 200:
                # Read the text content of the response
                text = await resp.text()
                # Extract the value using a regex pattern
                value_str = re.search(r'value_0\s+\d*.*\d+', text)
                if value_str:
                    return float(re.sub('value_0 = \'', '', value_str.group()).strip('[""]'))
            else:
                # If the response status is not OK, print an error message
                print(f'Request could not be made for some reason at time = {timestep} and layer = {layer}')
                return float('nan')

    except ServiceException:
        # Handle any ServiceException errors
        print(f'Request could not be made for some reason at time = {timestep} and layer = {layer}')
        return float('nan')

# Extraction of temporal information from metadata (this is exactly how ECCC does it in their tutorial)
def time_parameters(layer : str):
    start_time, end_time, interval = (wms[layer]
                                      .dimensions['time']['values'][0]
                                      .split('/')
                                      )
    iso_format = '%Y-%m-%dT%H:%M:%SZ'
    start_time = datetime.strptime(start_time, iso_format)
    end_time = datetime.strptime(end_time, iso_format)
    interval = int(re.sub(r'\D', '', interval))
    return start_time, end_time, interval

def setup_time(layer : str):
    '''
    This is what's proposed by ECCC in their tutorial
    :param layer:
    :return:
    '''
    # Setup time
    au_tz = pytz.timezone('America/Montreal')
    # get numerical time zone based on location (in this example, the local time zone would be UTC-05:00):
    start_time, end_time, interval = time_parameters(layer)

    # Calculation of date and time for available predictions
    # (the time variable represents time at UTC±00:00)
    time_utc = [start_time]
    while time_utc[-1] < end_time:
        time_utc.append(time_utc[-1] + timedelta(hours=interval))

    # Convert time to local time zone
    time_local = [t.replace(tzinfo=pytz.utc).astimezone(au_tz).replace(tzinfo=None) for t in time_utc]

    return time_local,time_utc

# Async version of the run_* functions
async def async_run_model(session,model_name:str, varlist:list, coor:list, nb_timestep:int) -> pd.DataFrame:
    print(f'Getting {model_name}')
    time_local, time_utc = setup_time(varlist[0])
    results = []
    for layer in varlist:
        layer_results = await asyncio.gather(
            *[async_request(session, layer, timestep, coor) for timestep in time_utc[:nb_timestep]]
        )
        results.append(layer_results)
    model_df = pd.DataFrame(results, index=varlist).transpose()
    model_df['Date'] = time_local[:nb_timestep]
    model_df[f'{model_name}_PR'] = model_df[f'{model_name}_PR'].diff()
    model_df[f'{model_name}_N4'] = (model_df[f'{model_name}_N4'].diff() / 3600).clip(lower=0) # remove possible negative values
    return model_df

# Modification of the process_request function to use asyncio
async def async_process_request(station_info: pd.DataFrame, nb_timestep=24) -> dict:
    coor = [station_info['Lon'], station_info['Lat2'], station_info['Lon2'], station_info['Lat']]

    # Start all three model data fetches in parallel
    async with aiohttp.ClientSession() as session:
        RDPS_df = await async_run_model(session, 'RDPS', ['RDPS.ETA_TT', 'RDPS.ETA_HR', 'RDPS.ETA_PR', 'RDPS.ETA_N4'], coor, nb_timestep)
        GDPS_df = await async_run_model(session, 'GDPS', ['GDPS.ETA_TT', 'GDPS.ETA_HR', 'GDPS.ETA_PR', 'GDPS.ETA_N4'], coor, nb_timestep)
        HRDPS_df = await async_run_model(session, 'HRDPS', ['HRDPS.CONTINENTAL_TT', 'HRDPS.CONTINENTAL_HR', 'HRDPS.CONTINENTAL_PR', 'HRDPS.CONTINENTAL_N4'], coor, nb_timestep)

    return {'RDPS': RDPS_df, 'GDPS': GDPS_df, 'HRDPS': HRDPS_df}
    # return {'RDPS':RDPS_df}

def fill_missing_hours(df : pd.DataFrame, date_col : str) -> pd.DataFrame:
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    # Create a DatetimeIndex for every hour between min_date and max_date
    full_index = pd.date_range(min_date, max_date, freq='H')

    # Set the date column as the index of the dataframe
    df = (df.set_index(date_col)
          .reindex(full_index) # Reindex the dataframe using the full index, which fills in missing hours with NaNs
          .reset_index()  # Reset the index so the date column is a regular column again
          .rename(columns={'index':date_col})
          )
    return df

def concatenate_forecasts(forecast_dict : dict) -> pd.DataFrame:
    '''
    Combine forecast data into a single uniform time series
    :return: A single dataframe with all forecast data combined into a single time series
    '''

    # merge all three dataframes. Also check for any missing hours between min and max dates and if so, fill with nans
    df_merged = (reduce(lambda left, right: pd.merge(left, right, on=['Date'], how='outer'), [forecast_dict['HRDPS'],forecast_dict['RDPS'],forecast_dict['GDPS']])
             .pipe(fill_missing_hours,'Date'))

    # Currently df_merged is a single dataframe, but it has different columns for every forecast.
    # Create a list where each index is a specific variable (like PR) but it has all the different dataframes together
    groups = [df_merged.columns[df_merged.columns.str[-2:].isin([suffix])] for suffix in common_var_names]

    # Add new columns to df_merged, columns that are called from the list Forecast_Variables_List
    for i in range(len(common_var_names)):
        df_merged[forecast_variables[i]] = reduce(lambda left, right: left.combine_first(right),
                                                       [df_merged[gr] for gr in groups[i]])

    return df_merged[['Date'] + forecast_variables]

# Modification to the main execution to use asyncio
async def main():
    # Example usage with a single station info row
    # Read station information and process each station
    Path_To_Script = r"C:\Users\sebastien.durocher\PycharmProjects\GetWeatherData\source\Forecasts"
    InFile = os.path.join(Path_To_Script, 'VStations_test.dat')
    Stations_info = pd.read_csv(InFile, skiprows=2)
    Stations_info['Lon2'] = Stations_info['Lon'] + 0.1
    Stations_info['Lat2'] = Stations_info['Lat'] - 0.1
    single_station_info = Stations_info.iloc[0]

    forecasts = await async_process_request(single_station_info)
    forecast_dataframe = concatenate_forecasts(forecasts)
    forecast_dataframe = forecast_dataframe.interpolate()
    print(forecast_dataframe)

# Run the main function using asyncio
start_time = time.time()
asyncio.run(main())
elapsed_time = time.time() - start_time
print(f"Script execution time: {elapsed_time} seconds")