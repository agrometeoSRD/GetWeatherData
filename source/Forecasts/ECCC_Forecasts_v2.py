#!/usr/bin/env
"""
Creation date: 2023-03-29
Creator : sebastien durocher
Python version : 3.10

Description:
- Use the MSC API (oswlib) to get data from the geomet server.
- Input requires a spatial coordinate (From a station file) and output will save forecast for this coordinate within a csv file.

Updates:

Notes:
    - Unlike all past iterations of getting forecast, now the forecast script doesn't transform the data into RIMpro format
    - This only works for eastern time zones
"""
#TODO : Create automatic file to get acces to station data. Add the necessary errors to make sure the data is there
#TODO : Test fonctions for anomalous data
#TODO : Test forecast with multiple stations
#TODO : Create a seperate script that would accept the forecast data and convert it into RIMpro
#TODO : Apply asyncio to the script
#TODO : Create some kind of secondary dictionnary that associates each variable with its corresponding forecast variable, that way the user can easily specify the variables that he wants and then the code will get the corresponding layers
#TODO : Find some way to incorporate a percentage progress bar

# imports
import os
import re
import warnings
from datetime import datetime, timedelta
from dateutil import tz

import numpy as np
import pandas as pd
from owslib.util import ServiceException
from owslib.wms import WebMapService

warnings.filterwarnings("ignore")

# new imports
import aiohttp
import asyncio
import pytz


#%% Testing out chatgpt stuff
# Function to request weather data for a given layer, time, and coordinates
wms_url = 'https://geo.weather.gc.ca/geomet/?SERVICE=WMS&REQUEST=GetFeatureInfo'
wms = WebMapService(wms_url , version='1.3.0', timeout=300)
common_var_names = ['TT', 'HR', 'PR', 'N4'] # These are the common variable names between the different forecast models
forecast_variables = ["AIRTEMP", "HR", "RAIN","GLOBALRAD"] # This is the desired variable names for the final dataframe

def request(layer: str, times: list, coor: list) -> list:

    pixel_values = []
    for timestep in times:
        try:
            response = wms.getfeatureinfo(
                layers=[layer],
                srs='EPSG:4326',
                bbox=tuple(coor),
                size=(100, 100),
                format='image/jpeg',
                query_layers=[layer],
                info_format='text/plain',
                xy=(1, 1),
                feature_count=1,
                time=timestep.isoformat() + 'Z'
            )

            text = response.read().decode('utf-8')
            value_str = re.search(r'value_0\s+\d*.*\d+', text)
            if value_str:
                pixel_values.append(float(re.sub('value_0 = \'', '', value_str.group()).strip('[""]')))
            else:
                pixel_values.append(float('nan'))
        except ServiceException:
            pixel_values.append(float('nan'))

    return pixel_values

# Extraction of temporal information from metadata
def time_parameters(layer):
    start_time, end_time, interval = (wms[layer]
                                      .dimensions['time']['values'][0]
                                      .split('/')
                                      )
    iso_format = '%Y-%m-%dT%H:%M:%SZ'
    start_time = datetime.strptime(start_time, iso_format)
    end_time = datetime.strptime(end_time, iso_format)
    interval = int(re.sub(r'\D', '', interval))
    return start_time, end_time, interval

def setup_time(layer):
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

# Function to get GDPS data
def run_GDPS(coor: list, nb_timestep=None):
    print('Getting GDPS')
    GDPS_varlist = ['GDPS.ETA_TT', 'GDPS.ETA_HR', 'GDPS.ETA_PR', 'GDPS.ETA_N4']
    time_local, time_utc = setup_time(GDPS_varlist[0])

    pixel_value_dict_GDPS = {layer: request(layer, time_utc[:nb_timestep], coor) for layer in GDPS_varlist}
    GDPS_df = pd.DataFrame.from_dict(pixel_value_dict_GDPS, orient='index').transpose()
    GDPS_df['Date'] = time_local[:nb_timestep]
    GDPS_df['GDPS.ETA_PR'] = GDPS_df['GDPS.ETA_PR'].diff()
    GDPS_df['GDPS.ETA_N4'] = (GDPS_df['GDPS.ETA_N4'].diff() / 3600).clip(lower=0)

    return GDPS_df

# Function to get HRDPS data
def run_HRDPS(coor: list, nb_timestep=None):
    print('Getting HRDPS')
    HRDPS_varlist = ['HRDPS.CONTINENTAL_TT', 'HRDPS.CONTINENTAL_HR', 'HRDPS.CONTINENTAL_PR', 'HRDPS.CONTINENTAL_N4']
    time_local, time_utc = setup_time(HRDPS_varlist[0])

    pixel_value_dict_HRDPS = {layer: request(layer, time_utc[:nb_timestep], coor) for layer in HRDPS_varlist}
    HRDPS_df = pd.DataFrame.from_dict(pixel_value_dict_HRDPS, orient='index').transpose()
    HRDPS_df['Date'] = time_local[:nb_timestep]
    HRDPS_df['HRDPS.CONTINENTAL_PR'] = HRDPS_df['HRDPS.CONTINENTAL_PR'].diff()
    HRDPS_df['HRDPS.CONTINENTAL_N4'] = (HRDPS_df['HRDPS.CONTINENTAL_N4'].diff() / 3600).clip(lower=0)

    return HRDPS_df

def run_RDPS(coor: list, nb_timestep=None):
    print('Getting RDPS')
    RDPS_varlist = ['RDPS.ETA_TT', 'RDPS.ETA_HR', 'RDPS.ETA_PR','RDPS.ETA_N4']
    time_local, time_utc = setup_time(RDPS_varlist[0])

    pixel_value_dict_rdps = {layer: request(layer, time_utc[:nb_timestep], coor) for layer in RDPS_varlist}
    RDPS_df = pd.DataFrame.from_dict(pixel_value_dict_rdps, orient='index').transpose()
    RDPS_df['Date'] = time_local[:nb_timestep]
    RDPS_df['RDPS.ETA_PR'] = RDPS_df['RDPS.ETA_PR'].diff()
    RDPS_df['RDPS.ETA_N4'] = (RDPS_df['RDPS.ETA_N4'].diff() / 3600).clip(lower=0) # remove possible negative values
    return RDPS_df

# Main processing function to get RDPS, GDPS, and HRDPS data for each station
def process_request(station_info: pd.DataFrame, nb_timestep=24) -> dict:
    '''
    process_request is being applied on a pandas dataframe. Therefore the output will be a pd.series where each row is the forecast dictionnary

    :param station_info:
    :param nb_timestep:
    :return:
    '''
    coor = [station_info['Lon'], station_info['Lat2'], station_info['Lon2'], station_info['Lat']]

    RDPS_df = run_RDPS(coor, nb_timestep) # Function run_RDPS should be previously defined
    GDPS_df = run_GDPS(coor, nb_timestep)
    HRDPS_df = run_HRDPS(coor, nb_timestep)

    return {'RDPS': RDPS_df, 'GDPS': GDPS_df, 'HRDPS': HRDPS_df}

from functools import reduce
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

def process_forecast():
    pass
    # Linear interpolation of missing values


def load_past_forecast(past_path: str, filename : str) -> pd.DataFrame:
    '''
    Only accepts csv files for now
    #TODO : Add support for other file types

    :param path_input:
    :param filename:
    :return:
    '''
    try:
        df = pd.read_csv(f"{past_path}\\{filename}.csv", sep=None,skiprows=1)
        if len(df.columns) > 1:
            print("Detected separator: comma")
        else:
            df = pd.read_csv(f"{past_path}\\{filename}.csv",sep=';',skiprows=1)
            print("Detected separator: semicolon")
    except FileNotFoundError:
        print('No file found, returning empty dataframe with columns DATE and TIME')
        df = pd.DataFrame(columns=['DATE', 'TIME'])  # return empty dataframe
    return df

def combine_past_and_current_forecast(past_df : pd.DataFrame, current_df : pd.DataFrame) -> pd.DataFrame:
    pass

# Save forecast within a csv file.
def save_forecast(forecast_df:pd.DataFrame, save_path : str,filename : str):
    print('Saving forecast to : ')
    forecast_df.to_csv(f"{save_path}\\{filename}.csv", index=False, sep=';')

# Read station information and process each station
Path_To_Script = r"C:\Scripts\PycharmProjects\GetWeatherData\source\Forecasts"
InFile = os.path.join(Path_To_Script, 'VStations_test.dat')
Stations_info = pd.read_csv(InFile, skiprows=2)
Stations_info['Lon2'] = Stations_info['Lon'] + 0.1
Stations_info['Lat2'] = Stations_info['Lat'] - 0.1

results = Stations_info.apply(lambda row: process_request(row, nb_timestep=10), axis=1)
# Only works with one station value
for forecast in results:
    forecast_dataframe = concatenate_forecasts(forecast)
    # process_forecast
    # save_forecast()
    # TODO : WHATS NEXT : INTERPOLATE MISING HOURS
# 'results' is a Series of dictionaries containing RDPS, GDPS, and HRDPS data for each station