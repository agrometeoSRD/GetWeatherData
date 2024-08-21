"""
File: combine_station_hrdps.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747

Description: Some .BRU dont have solar radiation. Use HRDPS solar radiation nowcast to fill in the missing data
- For code to work, first must have .BRU files and HRDPS solar radiation forecast
- .BRU files can be accessed by imported get_SM_data.py
- HRDPS files are locally saved in a folder. These are created by fisrt downloading ec_forecasts.py and then saving with save_ec_nowcast.py

- Output columns will be standardized to the universal column names (for those that have a universal name)

Notes
- There must be available nowcast files within the saved virtual station folder. To create these, see ec_forecasts.py and save_ec_nowcast.py

Created: 2024-02-27
"""

# Import statements
import os
import sys
import pandas as pd
import numpy as np
import datetime
import argparse
from getweatherdata.source.Observations.Stations.get_SM_data import download_and_process_data
from getweatherdata.utils.utils import load_config
from getweatherdata.utils.utils import standardize_columns
# Constants

# Functions


def read_quick_wu_online(station_name:str) -> pd.DataFrame:
    url = f"http://meteo.irda.qc.ca/4z/{station_name}.csv"
    df = pd.read_csv(url, sep=';',skiprows=1)
    # Overwrite the column names to be standardized
    df['Date'] = pd.to_datetime(df['DATE']) + pd.to_timedelta(df['TIME'].str.split(':').str[0].astype(int), unit='h')
    # drop 'DATE' and 'TIME' column in order to just have one column for date
    df = df.drop(columns=['DATE','TIME'])
    return df

# Load the .BRU files for selected stations
def load_saved_nowcast_csv(InFile : str) -> pd.DataFrame:
    if os.path.isfile(InFile):
        df = pd.read_csv(InFile, sep=';')
        df = standardize_columns(df)
        return df
    else: # return empty pandas dataframe
        print('No nowcast data not found.')
        return pd.DataFrame(columns=['Date'])

# Combine the .BRU and nowcast data
def concatenate_station_nowcast(station_df, nowcast_df):
    station_df_indexed = station_df.assign(Date=pd.to_datetime(station_df['Date'])).set_index('Date')
    nowcast_df_indexed = nowcast_df.assign(Date=pd.to_datetime(nowcast_df['Date'])).set_index('Date')
    # replace missing data from bru_df with that of nowcast_df
    combined_df = station_df_indexed.combine_first(nowcast_df_indexed).reset_index()
    return combined_df

# Save the combined data as .csv
def save_dataframe_to_csv(forecast_df: pd.DataFrame, save_path: str, filename: str):
    out = f"{save_path}\\{filename}.csv"
    print(f'Saving bru+nowcast to : {out}')
    forecast_df.to_csv(out, index=False, sep=';',na_rep=np.nan)


# Main execution ---------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Process weather forecast data.')
    parser.add_argument('--dat-file', default=r"C:\Users\sebastien.durocher\PycharmProjects\GetWeatherData\test\VStations.dat", help='Path to the .dat file with station information')
    return parser.parse_args()

def main(config,dat_file):
    # Load the configuration file
    save_path = config['Paths']["SavedEcVsForecastsPath"] # define as a variable the path to the data

    # set up parameters
    sel_year = datetime.datetime.now().strftime("%Y") # get current year (in string)

    # Load station info
    try:
        stations_info = pd.read_csv(dat_file, skiprows=2)
    except Exception as e:
        print(f"Error reading file {dat_file}: {e}")
        sys.exit(1)

    for i, station_info in stations_info.iterrows():
        try :
            sel_station_type = stations_info.loc[i,'Type']
            sel_station_name = stations_info.loc[i,'Name']
            sel_station_id = stations_info.loc[i,'ID']
            print(f'Processing station ID = {sel_station_id}')

            if sel_station_type == 'WU':
                station_df = standardize_columns(read_quick_wu_online(sel_station_name)) # get the wu data
            elif sel_station_type == 'FADQ':
                station_df = standardize_columns(download_and_process_data([sel_station_name], [sel_year])) # get the bru data
            else:
                print(f'Station Type not recognized for station ID = {sel_station_id}. Skipping to next station.')
            if not station_df.empty: # Only continue if .bru exists
                nowcast_df = load_saved_nowcast_csv(f'{save_path}\\{sel_station_id}{config["Filename_extension"]["EcVs"]}.csv') # get the nowcast
                combined_df = concatenate_station_nowcast(station_df, nowcast_df) # merge together
                save_dataframe_to_csv(combined_df, save_path, f"{sel_station_id}_bru_nowcast") # save as csv
            else:
                print('No station data found this file. Skipping to next station.')
        except :
            print('Some problem occured at the given station. Creation of bru+nowcast was unsuccessful.')
            pass

if __name__ == "__main__":
    args = parse_args()  # Parse command-line arguments
    config = load_config('ec_config.json')
    main(config,args.dat_file)  # Run the main function and pass the dat_file path to main()

# TODO : Check what happens if file exists for nowcast (like RIMpro nowcast) but doesnt exist in .BRU
# TODO : Check what happens if file exists for .BRU but doesnt exist for nowcast
# TODO : Check what happens if file exists for neither nowcast nor .BRU
# TODO : Remove absolute path and replace with something more universal / flexible
# TODO : Create a new script that does the same thing but for non .BRU files (like WU)
# TODO : Add some kind of smoothing when adding nowcast to .BRU
