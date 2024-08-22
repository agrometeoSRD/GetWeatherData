# %%
# !/usr/bin/env
"""
File: noaa_forecast.py
Author: Sebastien Durocher
Python: 3.10
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/agrometeoSRD

Status : functional, but not tested since directory changes

Description:
    This script retrieves weather forecast data from the MSC API (Meteorological Service of Canada) using the OWSLib library to interact with the GeoMet server.
    The script is designed to extract meteorological variables such as temperature, humidity, precipitation, and solar radiation for specific spatial coordinates.
    The output is saved as a CSV file, containing the forecast data for a given location over a specified time period.

Features:
    - Extracts and processes data from the HRDPS (short term), RDPS (medium term) and GDPS (long term) models.
    - Supports cumulative precipitation and solar radiation calculations by converting their cumulative values into hourly values.
    - Provides a CLI interface for easy execution with customizable input files containing station information.
    - Automatically handles Eastern Time Zone conversions for local time predictions.
    - The script can be extended or modified to fit different time zones or other data processing needs.

Notes:
    - The script assumes the default time zone as Eastern Time (America/Montreal) and may require adjustments for other regions.
    - Precipitation is extracted in kg/m² and converted directly to mm assuming a 1:1 ratio with water density.
    - Solar radiation is provided in J/m² and is converted to W/m² for hourly values.
    - For detailed information about the variable descriptions, refer to the GeoMet GetCapabilities documentation:
      https://geo.weather.gc.ca/geomet?lang=en&service=WMS&version=1.3.0&request=GetCapabilities

Usage:
    - The script can be executed from the command line with a station file containing spatial coordinates:
      Example: `python -m source.forecasts.ec_forecast --dat-file THEPATH/test/vs_stations_test.dat`
    - The output CSV file will contain columns such as DATE, AIRTEMP, HR, RAIN, GLOBALRAD, with hourly forecast data.
    - Ensure that the input files are correctly formatted and that the script is run from the root directory of the project.

TODO:
    - Implement automated data retrieval for station coordinates and add necessary error handling.
    - Develop functions for identifying and processing anomalous data.
    - Integrate a progress bar to monitor the script's execution.

Special dependencies:
    - Required packages include OWSLib for web map services, pandas for data handling, and pytz for timezone management.
    - The script has been especially tested with Python 3.10 and may require adjustments for compatibility with other versions.

Inputs:
    - Spatial coordinates for the location of interest (provided through a station file).
    - Temporal parameters, including start and end dates for the forecast data.
    - Model-specific variables for data extraction (e.g., TT, HR, PR, N4 for temperature, humidity, precipitation, and radiation).

Outputs:
    - The forecast data is saved in a CSV file with a specific format, allowing for easy integration into further analyses or tools.

Example CLI Usage:
    - The script can be executed using a command line interface, with station files passed as arguments.
    (again make sure to run the script from the root directory of the project)
      `python -m src.getweatherdata.forecasts.ec_forecast --dat-file path_to_file.dat`

Created: 2023-03-29
"""


# imports
from utils.utils import load_config
import sys
import os
import re
import warnings
import argparse
from datetime import datetime, timedelta
from functools import reduce
import time
import pytz
import numpy as np
import pandas as pd
import logging
from owslib.util import ServiceException
from owslib.wms import WebMapService

warnings.filterwarnings("ignore")

# constants
# Request information for weather data for a given layer, time, and coordinates
wms_url = 'https://geo.weather.gc.ca/geomet/?SERVICE=WMS&REQUEST=GetFeatureInfo'
wms = WebMapService(wms_url, version='1.3.0', timeout=300)
common_var_names = ['TT', 'HR', 'PR', 'N4']  # These are the common variable names between the different forecast models

# Functions
def request(layer: str, times: list, coor: list) -> list:
    pixel_values = []
    for timestep in times:
        try:
            response = wms.getfeatureinfo(layers=[layer], srs='EPSG:4326', bbox=tuple(coor), size=(100, 100),
                format='image/jpeg', query_layers=[layer], info_format='text/plain', xy=(1, 1), feature_count=1,
                time=timestep.isoformat() + 'Z')

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
    start_time, end_time, interval = (wms[layer].dimensions['time']['values'][0].split('/'))
    iso_format = '%Y-%m-%dT%H:%M:%SZ'
    start_time = datetime.strptime(start_time, iso_format)
    end_time = datetime.strptime(end_time, iso_format)
    interval = int(re.sub(r'\D', '', interval))
    return start_time, end_time, interval


def setup_time(layer:str):
    '''
    This is what's proposed by ECCC in their tutorial to handle time
    Essentially we're creating an initial state from a single layer to get the initial time of the forecast
    :param layer: a random layer name. Could have been any layer
    :return: time_local, time_utc (both are lists that contain all of the forecast times)
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

    return time_local, time_utc


# Function to get GDPS data
def run_gdps(coor: list, nb_timestep: dict, date_col: str) -> pd.DataFrame:
    print('Getting GDPS')
    GDPS_varlist = ['GDPS.ETA_TT', 'GDPS.ETA_HR', 'GDPS.ETA_PR', 'GDPS.ETA_N4']
    time_local, time_utc = setup_time(GDPS_varlist[0])
    # time slicing happens a bit different in gdps because dt is by 3 hours and not 1 hour
    start_idx = [idx for idx, dt in enumerate(time_utc) if dt == (time_utc[0] + timedelta(hours=nb_timestep['RDPS']))][
        0]
    pixel_value_dict_GDPS = {layer: request(layer, time_utc[start_idx:], coor) for layer in GDPS_varlist}
    gdps_df = pd.DataFrame.from_dict(pixel_value_dict_GDPS, orient='index').transpose()
    gdps_df[date_col] = time_local[start_idx:]
    gdps_df['GDPS.ETA_PR'] = gdps_df['GDPS.ETA_PR'].diff().clip(lower=0)
    gdps_df['GDPS.ETA_N4'] = (gdps_df['GDPS.ETA_N4'] / (3 * 3600)).diff().clip(lower=0)

    return gdps_df


# Function to get HRDPS data
def run_hrdps(coor: list, nb_timestep: dict, date_col: str) -> pd.DataFrame:
    print('Getting HRDPS')
    HRDPS_varlist = ['HRDPS.CONTINENTAL_TT', 'HRDPS.CONTINENTAL_HR', 'HRDPS.CONTINENTAL_PR', 'HRDPS.CONTINENTAL_N4']
    time_local, time_utc = setup_time(HRDPS_varlist[0])

    pixel_value_dict_HRDPS = {layer: request(layer, time_utc[:nb_timestep['HRDPS']], coor) for layer in HRDPS_varlist}
    hrdps_df = pd.DataFrame.from_dict(pixel_value_dict_HRDPS, orient='index').transpose()
    hrdps_df[date_col] = time_local[:nb_timestep['HRDPS']]
    hrdps_df['HRDPS.CONTINENTAL_PR'] = hrdps_df['HRDPS.CONTINENTAL_PR'].diff().clip(lower=0)
    hrdps_df['HRDPS.CONTINENTAL_N4'] = (hrdps_df['HRDPS.CONTINENTAL_N4'] / 3600).diff().clip(lower=0)

    return hrdps_df


def run_rdps(coor: list, nb_timestep: dict, date_col: str) -> pd.DataFrame:
    print('Getting RDPS')
    RDPS_varlist = ['RDPS.ETA_TT', 'RDPS.ETA_HR', 'RDPS.ETA_PR', 'RDPS.ETA_N4']
    time_local, time_utc = setup_time(RDPS_varlist[0])

    pixel_value_dict_rdps = {layer: request(layer, time_utc[nb_timestep['HRDPS']:nb_timestep['RDPS']], coor) for layer
                             in RDPS_varlist}
    rdps_df = pd.DataFrame.from_dict(pixel_value_dict_rdps, orient='index').transpose()
    rdps_df[date_col] = time_local[nb_timestep['HRDPS']:nb_timestep['RDPS']]
    rdps_df['RDPS.ETA_PR'] = rdps_df['RDPS.ETA_PR'].diff().clip(lower=0)
    rdps_df['RDPS.ETA_N4'] = (rdps_df['RDPS.ETA_N4'] / 3600).diff().clip(lower=0)  # remove possible negative values
    return rdps_df


# Main processing function to get RDPS, GDPS, and HRDPS data for each station
def process_request(station_coor: pd.DataFrame, date_col: str) -> dict:
    """
    process_request is being applied on a pandas dataframe. Therefore, the output will be a pd.series where each row is the forecast dictionnary

    Added "intelligent" timestep pacing for each forecast.
    Meaning RDPS will only count times after HRDPS and GDPS will only count times after RDPS. HRDPS starts counting from the beggining.
    To change the number of timesteps for each forecast, change the timesteps_dict variable within the function

    :param station_coor: specific row of the station coordinates dataframe
    :param date_col: name of the date column
    :return: dictionnary that contains the forecast of each model (each key is the model, each value is the dataframe)
    """
    print(f'Acquiring forecast for station : {station_coor["Name"]}')
    coor = [station_coor['Lon'], station_coor['Lat2'], station_coor['Lon2'], station_coor['Lat']]
    timesteps_dict = {'HRDPS': 48, 'RDPS': 84, 'GDPS': 120}  # default values should be : 48, 84 and 120

    hrdps_df = run_hrdps(coor, timesteps_dict, date_col)
    rdps_df = run_rdps(coor, timesteps_dict, date_col)
    gdps_df = run_gdps(coor, timesteps_dict, date_col)

    return {'RDPS': rdps_df, 'GDPS': gdps_df, 'HRDPS': hrdps_df}

def concatenate_forecasts(forecast_dict: dict, date_col: str) -> pd.DataFrame:
    """
    Combine forecast data into a single uniform time series
    :return: A single dataframe with all forecast data combined into a single time series
    """

    # merge all three dataframes. Also check for any missing hours between min and max dates and if so, fill with nans
    df_merged = (reduce(lambda left, right: pd.merge(left, right, on=['Date'], how='outer'),
                        [forecast_dict['HRDPS'], forecast_dict['RDPS'], forecast_dict['GDPS']]))

    # Currently df_merged is a single dataframe, but it has different columns for every forecast.
    # Create a list where each index is a specific variable (like PR) but it has all the different dataframes together
    groups = [df_merged.columns[df_merged.columns.str[-2:].isin([suffix])] for suffix in common_var_names]

    # Add new columns to df_merged, columns that are called from the list Forecast_Variables_List
    for i in range(len(common_var_names)):
        df_merged[forecast_variables[i]] = reduce(lambda left, right: left.combine_first(right),
                                                  [df_merged[gr] for gr in groups[i]])

    return df_merged[[date_col] + forecast_variables]


def fill_missing_hours(df:pd.DataFrame, date_col:str)->pd.DataFrame:
    """
    Fill missing hours in the dataframe with NaNs in order to have a continous time from start to finish of the dataframe

    :param df: dataframe that contains the missing hours
    :param date_col: name of the date column (kinda silly at this point that we're carrying it everywhere)
    :return:
    """
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    # Create a DatetimeIndex for every hour between min_date and max_date
    full_index = pd.date_range(min_date, max_date, freq='H')

    # Set the date column as the index of the dataframe
    df = (df.set_index(date_col).reindex(
        full_index)  # Reindex the dataframe using the full index, which fills in missing hours with NaNs
          .reset_index()  # Reset the index so the date column is a regular column again
          .rename(columns={'index': date_col}))

    # interpolate missing hours (backfill)
    df = df.interpolate()  # fill missing hours
    return df


def load_forecast(past_path: str, filename: str, date_col: str) -> pd.DataFrame:
    """
    Only accepts csv files for now.
    csv should have an expected one row of header that will be skipped
    #TODO : Add support for other file types
    #TODO : Add condition if we skip a an uncessary header (like if no header, or if two headers)
    """

    # check if file exists. If it doesn't return as empty dataframe
    filename = f"{past_path}\\{filename}.csv"
    if not os.path.isfile(filename):
        print('No past forecast file found. Saved forecast will only be for current forecast')
        return pd.DataFrame(columns=[date_col] + forecast_variables)
    else:
        df = pd.read_csv(filename, sep=None, engine='python', parse_dates=[date_col],
                         dtype={variables['temp_col']: 'float64', variables['hr_col']: 'float64',
                             variables['rain_col']: 'float64', variables['rad_col']: 'float64'})
    return df


def combine_past_and_current_forecast(past_df: pd.DataFrame, current_df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    # The function should work under the following conditions:
    # 1) If past_df is empty, return current_df
    # 2) if current_df is empty, return past_df
    # 3) if both dataframes are empty, return an empty dataframe
    # 4) if past_df exists, but the dataframe does not match current_df, return current_df
    # 5) if past_df exists, but the time forecast has nothing in common with current_df, return current_df
    # 6) if past_df exists and has time forecast that overlap with current_df,
    # then combine the two dataframes but make sure current_df has priority

    # Condition 1 : If past_df is empty, return current_df
    if past_df.empty:
        return current_df

    # Condition 2 : If current_df is empty, return past_df
    if current_df.empty:
        return past_df

    # Condition 3 : If both dataframes are empty, return an empty dataframe
    if past_df.empty and current_df.empty:
        return pd.DataFrame(columns=[date_col] + forecast_variables)

    # Condition 4 : Ensure both dataframes have the same columns
    if set(past_df.columns) != set(current_df.columns):
        print('Columns are not the same, returning current forecast as the only appropriate forecast')
        return current_df

    # todo : missing answer for condition 5

    # Condition 6
    # Sort dataframes by date in chronological order
    past_df = past_df.sort_values(by=date_col, ascending=True)
    current_df = current_df.sort_values(by=date_col, ascending=True)

    # initial forecast hour of current_df has nan for rain (because of .diff() done previously)
    # will try to fill it with the last forecast hour of past_df
    first_date = current_df[date_col].iloc[0]
    past_matches = past_df[past_df[date_col] == first_date]
    # Check if there's a matching date in past_df and the RAIN value is not NaN
    if not past_matches.empty and not pd.isna(past_matches[variables['rain_col']].iloc[0]):
        past_rain = past_matches[variables['rain_col']].iloc[0]
    else:
        past_rain = 0  # Default to 0 if no matching date or value is NaN
    # update current_df with the corresponding past_rain value
    current_df.loc[current_df[date_col] == first_date, variables['rain_col']] = past_rain

    # slice past_df so to only have times that after first hour of current_df
    past_df = past_df[past_df[date_col] > first_date]

    combined_df = (current_df.combine_first(past_df)  # Combine dataframes, prioritizing current forecast
                   .sort_values(by=date_col)  # Sort by date in ascending order
                   .drop_duplicates(subset=date_col, keep='first')  # Remove duplicate rows
                   .reset_index(drop=True)  # Reset the index
                   )

    return combined_df


def sanity_check(df: pd.DataFrame):
    # check to make sure the dataframe has the right columns

    # check to make sure the values are within the right range

    pass


# Save forecast within a csv file.
def save_forecast(forecast_df: pd.DataFrame, save_path: str, filename: str):
    out = f"{save_path}\\{filename}.csv"
    print(f'Saving forecast to : {out}')
    forecast_df.to_csv(out, index=False, sep=';', na_rep=np.nan)


# %% Read station information and process each station
def parse_args():
    parser = argparse.ArgumentParser(description='Process weather forecast data.')
    parser.add_argument('--dat-file', default=r"C:\Users\sebastien.durocher\PycharmProjects\GetWeatherData\test\VStations.dat", help='Path to the .dat file with station information')
    return parser.parse_args()

def main(config,dat_file):
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    path_to_save = config['Paths']["SavedEcForecastsPath"]
    date_col = config['General']['DateColumn']

    # Load station info
    try:
        station_coordinates = pd.read_csv(dat_file, skiprows=2)
    except Exception as e:
        logger.error(f"Error reading file {dat_file}: {e}")
        sys.exit(1)

    # Additional processing
    station_coordinates['Lon2'] = station_coordinates['Lon'] + 0.1
    station_coordinates['Lat2'] = station_coordinates['Lat'] - 0.1

    # Process each station
    for i, row in station_coordinates.iterrows():
        # Log progress
        logger.info(f"Processing station {row['ID']}")
        try:
            forecast = process_request(row, date_col)
            forecast_dataframe = concatenate_forecasts(forecast, date_col)
            past_forecast = load_forecast(path_to_save, f'{row["ID"]}{config["Filename_extension"]["EcForecast"]}', date_col)
            final_forecast = combine_past_and_current_forecast(past_forecast, forecast_dataframe, date_col)
            final_forecast = fill_missing_hours(final_forecast, date_col)
            final_forecast[forecast_variables] = final_forecast[forecast_variables].round(3)
            save_forecast(final_forecast, path_to_save, f'{row["ID"]}{config["Filename_extension"]["EcForecast"]}')
        except Exception as e:
            logger.error(f"Error processing station {row['ID']}: {e}")

    logger.info(f"Script completed in {time.time() - start_time} seconds")

if __name__ == "__main__":
    args = parse_args()  # Parse command-line arguments
    config = load_config('ec_config.json')
    variables = config['General']
    forecast_variables = [variables['temp_col'], variables['hr_col'], variables['rain_col'], variables['rad_col']]
    main(config, args.dat_file)  # Pass the dat_file path to main
