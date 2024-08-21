"""
File: save_ec_nowcast.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description:
- Create a virtual station by saving HRDPS "nowcast (0-6 hours)" into a csv file for a specific coordinate
- If the file already exists, it just adds on top of it
- Searches for the most recent HRPDS forecast from saved_forecasts.csv
- Adds this most recent forecast to the corresponding forecast in folder vs_forecast

Created: 2024-02-20
"""

# Import statements
import os
import logging
import time
import numpy as np
import pandas as pd
import glob

from utils.utils import load_config

# Constants

# Functions
def generate_dataframe(time_start,timesteps=24):
    # time_start = '2024-02-20 00:00:00'
    # Create date range
    date_range = pd.date_range(start=time_start, periods=timesteps, freq='h')

    # Create DataFrame
    df = pd.DataFrame({
        'Date': date_range,
        'AIRTEMP [C]': np.random.uniform(-20, 20, timesteps),
        'HR [percent]': np.random.uniform(0, 100, timesteps),
        'RAIN [mm]': np.random.uniform(0, 1, timesteps),
        'GLOBALRAD [Wm2]': np.random.uniform(0, 500, timesteps)
    })
    return df

def load_forecast(path: str, filename: str, date_col:str) -> pd.DataFrame:
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
    filename = f"{path}\\{filename}.csv"
    if not os.path.isfile(filename):
        print('No saved forecast found. Creating new virtual station with only current 0-6 hours forecast')
        return pd.DataFrame(columns=[date_col] + forecast_variables)
    else:
        df = pd.read_csv(filename,sep=None,engine='python',parse_dates=[date_col],dtype={
                        variables['temp_col']: 'float64',
                        variables['hr_col']: 'float64',
                        variables['rain_col']: 'float64',
                        variables['rad_col']: 'float64'
                    })
    return df

def load_most_recent_forecast(path: str, filename: str, date_col:str) -> pd.DataFrame:
    most_recent_forecast = load_forecast(path, filename, date_col)
    # cut current_df so that only get 0 to 6 hours
    time_init = most_recent_forecast.loc[0,date_col]
    time_final = pd.Timestamp(pd.Timestamp.now().replace(minute=0,second=0,microsecond=0))
    most_recent_forecast = most_recent_forecast[(most_recent_forecast[date_col] >= time_init) & (most_recent_forecast[date_col] <= time_final)]
    return most_recent_forecast

def load_vs_forecast(path : str,filename : str,date_col:str) -> pd.DataFrame:
    vs_forecast = load_forecast(path,filename,date_col)
    return vs_forecast

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

    combined_df = pd.concat([current_df, past_df]).sort_values(by=date_col).drop_duplicates(subset=date_col, keep='first')

    return combined_df

def fill_missing_hours(df, date_col):
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    # Create a DatetimeIndex for every hour between min_date and max_date
    full_index = pd.date_range(min_date, max_date, freq='h')

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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    path_to_vs = config['Paths']["SavedEcVsForecastsPath"]
    path_to_forecast = config['Paths']["SavedEcForecastsPath"]
    date_col = config['General']['DateColumn']

    # Use glob to find all CSV files in the directory
    csv_files = glob.glob(os.path.join(path_to_forecast, '*.csv'))
    # Process each CSV file
    for csv_file in csv_files:
        try:
            # Extract station ID from the file name, assuming it's formatted as '{ID}_saved_forecast.csv'
            # This step may need adjustment based on your actual file naming conventions
            station_id = os.path.basename(csv_file).split('_')[0]

            logger.info(f"Processing station {station_id}")

            # Load the current and vs forecasts using the derived station ID
            current_forecast = load_most_recent_forecast(path_to_forecast, f'{station_id}{config["Filename_extension"]["EcForecast"]}', date_col)
            vs_forecast = load_vs_forecast(path_to_vs, f'{station_id}{config["Filename_extension"]["EcVs"]}', date_col)

            # Combine forecasts and process them
            combined_forecast = combine_past_and_current_forecast(vs_forecast, current_forecast, date_col)
            combined_forecast = fill_missing_hours(combined_forecast, date_col)

            # Save the combined forecast
            save_forecast(combined_forecast, path_to_vs, f'{station_id}{config["Filename_extension"]["EcVs"]}')

        except Exception as e:
            logger.error(f"Error processing station {csv_file}: {e}")

# Main execution --------------------------------------

if __name__ == "__main__":
    config = load_config('ec_config.json')
    variables = config['General']
    forecast_variables = [variables['temp_col'], variables['hr_col'], variables['rain_col'], variables['rad_col']]
    main(config)