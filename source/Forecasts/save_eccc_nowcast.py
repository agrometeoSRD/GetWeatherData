"""
File: save_eccc_nowcast.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description:
- Create a virtual station by saving HRDPS "nowcast (0-6 hours)" into a csv file for a specific coordinate
- If the file already exists, it just adds on top of it
- Searches for the most recent HRPDS forecast from saved_forecasts.csv
- Adds this most recent forecast to the corresponding forecast in folder historical_forecast

Created: 2024-02-20
"""

# Import statements
import sys
import os
import configparser
from utils import load_eccc_forecast_config_file
import logging
import time

import numpy as np
import pandas as pd

# Constants

# Functions
def load_forecast(past_path: str, filename: str, date_col:str) -> pd.DataFrame:
    """
    Only accepts csv files for now.
    csv should have an expected one row of header that will be skipped
    #TODO : Add support for other file types
    #TODO : Add condition if we skip a an uncessary header (like if no header, or if two headers)

    :param path_input:
    :param filename:
    :return:
    """

    # check if file exists. If it doesn't return as empty dataframe
    filename = f"{past_path}\\{filename}.csv"
    if not os.path.isfile(filename):
        print('No file found, returning empty dataframe with columns DATE and TIME')
        return pd.DataFrame(columns=[date_col] + forecast_variables)
    else:
        df = pd.read_csv(filename, sep=None,dtype={date_col:'datetime64[ns]'})
        if len(df.columns) > 1:
            print("Detected separator: comma")
        else:
            df = pd.read_csv(f"{past_path}\\{filename}.csv", sep=';',dtype={date_col:'datetime64[ns]'})
            print("Detected separator: semicolon")
    return df

def load_most_recent_forecast(path: str, filename: str, date_col:str) -> pd.DataFrame:
    most_recent_forecast = load_forecast(path, filename, date_col)
    return most_recent_forecast

def load_historical_forecast(path : str,filename : str,date_col:str) -> pd.DataFrame:
    historical_forecast = load_forecast(path,filename,date_col)
    return historical_forecast

def combine_past_and_current_forecast(past_df: pd.DataFrame, current_df: pd.DataFrame,date_col:str) -> pd.DataFrame:
    # The function should work under the following conditions:
    # 1) If past_df is empty, return current_df
    # 2) if current_df is empty, return past_df
    # 3) if both dataframes are empty, return an empty dataframe
    # 4) if past_df exists, but the dataframe does not match current_df, return current_df
    # 5) if past_df exists, but the time forecast has nothing in common with current_df, return current_df (2024-02-20 : use function that fills missing hours with nan)
    # 6) if past_df exists and has time forecast that overlap with current_df, then combine the two dataframes but make sure current_df has priority

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

    # Condition 6
    # Sort dataframes by date in chronological order
    past_df = past_df.sort_values(by=date_col, ascending=True)
    current_df = current_df.sort_values(by=date_col, ascending=True)
    # cut current_df so that only get 0 to 6 hours
    time_init = current_df.loc[date_col,0]
    time_final = time_init+pd.Timedelta(hours=6)
    current_df = current_df[(current_df[date_col] >= time_init) & (current_df[date_col] <= time_final)]

    combined_df = (current_df.combine_first(past_df) # Combine dataframes, prioritizing current forecast
                   .sort_values(by=date_col)   # Sort by date in ascending order
                   .drop_duplicates(subset=date_col, keep='first')  # Remove duplicate rows
                   )

    return combined_df

def fill_missing_hours(df, date_col):
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    # Create a DatetimeIndex for every hour between min_date and max_date
    full_index = pd.date_range(min_date, max_date, freq='H')

    # Set the date column as the index of the dataframe
    df = (df.set_index(date_col)
          .reindex(full_index)  # Reindex the dataframe using the full index, which fills in missing hours with NaNs
          .reset_index()  # Reset the index so the date column is a regular column again
          .rename(columns={'index': date_col})
          )

    # interpolate missing hours (backfill)
    df = df.interpolate()  # fill missing hours
    return df

def save_forecast(forecast_df: pd.DataFrame, save_path: str, filename: str):
    out = f"{save_path}\\{filename}.csv"
    print(f'Saving forecast to : {out}')
    forecast_df.to_csv(out, index=False, sep=';',na_rep=np.nan)

def main(config):
    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load configuration
    config = configparser.ConfigParser()
    config.read(config)

    path_to_script = config.get('Paths', 'ScriptPath')
    path_to_current = config.get('Paths','SavedForecastsPath')
    path_to_historical = config.get('Paths', 'SavedHistoricalForecastsPath')
    date_col = config.get('General', 'DateColumn')

    # Load station info
    InFile = os.path.join(path_to_script, 'VStations_p1_test.dat')
    try:
        Stations_info = pd.read_csv(InFile, skiprows=2)
    except Exception as e:
        logger.error(f"Error reading file {InFile}: {e}")
        sys.exit(1)

    # Process each station
    for i, row in Stations_info.iterrows():
        # Log progress
        logger.info(f"Processing station {row['ID']}")
        try:
            current_forecast = load_most_recent_forecast(path_to_current,f'{row["ID"]}_saved_forecast',date_col)
            historical_forecast = load_historical_forecast(path_to_historical,f'{row["ID"]}_historical',date_col)
            combined_forecast = combine_past_and_current_forecast(historical_forecast,current_forecast,date_col)
            combined_forecast = fill_missing_hours(combined_forecast,date_col)
            save_forecast(combined_forecast,path_to_historical,f'{row["ID"]}_historical')
        except Exception as e:
            logger.error(f"Error processing station {row['ID']}: {e}")


# Main execution --------------------------------------

if __name__ == "__main__":
    config = load_eccc_forecast_config_file()
    variables = config['General']
    forecast_variables = [config['temp_col'], config['hr_col'], config['rain_col'], config['rad_col']]
    main(config)
