import csv
import os
import pandas as pd
from wunderground_pws import WUndergroundAPI, units

#TODO : Make it so that it loops over various stations and each aer stored as individual csv files
#TODO : Check for anomalies in the data (missing data, impossible values, missing dates)
#TODO : Check wether it would be necessary to compute hourly precipitation
#TODO : Hourly mean the data (temperature, wind, pressure, dewpoint). Get the last value of the hour for precipitation
#TODO : Make the history call go back more than 7 days
#TODO : Create error cases for wether : file is already open
#TODO : add an extra column that mentions if the data is from historical or current data


def load_existing_data(filepath):
    """Load existing data from a CSV file into a pandas DataFrame."""
    return pd.read_csv(filepath)

def extract_data_from_json(history):
    """Extract data from JSON structure and return as a list of dictionaries."""
    data_to_save = []
    for observation in history['observations']:
        row = {
            'obsTimeLocal': observation['obsTimeLocal'],
            'tempAvg': observation['metric_si']['tempAvg'],
            'windspeedAvg': observation['metric_si']['windspeedAvg'],
            'dewptAvg': observation['metric_si']['dewptAvg'],
            'pressureTrend': observation['metric_si']['pressureTrend'],
            'precipRate': observation['metric_si']['precipRate'],
            'precipTotal': observation['metric_si']['precipTotal']
        }
        data_to_save.append(row)
    return data_to_save

def resample_to_hour_avg(df:pd.DataFrame, datetime_col:str, cols_to_average:list):
    """
    Resample the DataFrame to get 15-minute averages for the specified columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with datetime measurements.
    - datetime_col (str): The name of the column in df that contains datetime objects.
    - cols_to_average (list): List of column names to calculate the averages for.

    Returns:
    - pd.DataFrame: A new DataFrame with 15-minute averages for the specified columns.
    """

    # First, ensure the datetime column is in datetime format and set it as the index
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df.set_index(datetime_col, inplace=True)

    # Now, resample and compute the mean for the specified columns
    resampled_df = df[cols_to_average].resample('H').mean()

    return resampled_df

def update_and_save_data(new_data, filepath):
    """Update CSV file with new data, remove duplicates, and sort by date."""
    if os.path.exists(filepath):
        existing_data_df = load_existing_data(filepath)
        combined_df = pd.concat([existing_data_df, new_data], ignore_index=True)
        combined_df.drop_duplicates(subset=['obsTimeLocal'], inplace=True)
        combined_df.sort_values(by='obsTimeLocal', inplace=True)
    else:
        combined_df = new_data
    
    combined_df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")


def main(history, csv_file_path):
    """Main function to process and save weather data."""
    new_data_list = extract_data_from_json(history)
    new_data_df = pd.DataFrame(new_data_list)
    cols_to_average = ['tempAvg', 'windspeedAvg', 'dewptAvg', 'pressureTrend']
    resampled_df = resample_to_hour_avg(new_data_df, 'obsTimeLocal', cols_to_average)
    # Add total precipitation
    resampled_df['precipTotal'] = new_data_df['precipTotal'].resample('H').last()
    update_and_save_data(resampled_df, csv_file_path)

# Define the CSV file name
csv_file_name = f'{os.getcwd()}\weather_data.csv'

wu = WUndergroundAPI(
    default_station_id='ISAINT6465',
    units=units.METRIC_SI_UNITS,
)
history = wu.hourly()
main(history, csv_file_name)

# Printing the data ------------------------------------------------------------------------------
# print('Current status of my weather station:')
# pprint(wu.current()['observations'][0])
# print('Summary of last 7 days:')
# pprint(wu.summary())
# print('Detailed hourly history for the last 7 days:')
# pprint(wu.hourly())
# print('History for 2/11/2023:')
# pprint(wu.history(date(day=11, month=2, year=2023)))
