"""
File: combine_bru_hrdps.py
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
import pandas as pd
import numpy as np
import datetime
from source.Observations.Stations.get_SM_data import download_and_process_data
from utils.utils import load_config
from utils.utils import invert_mapping
# Constants

# Functions
# Verification function that checks if the columns are standardized
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    universal_mapping = load_config('column_names.json') # load config file that contains mapping of column names
    standardize_columns_dict = invert_mapping(universal_mapping)
    # Apply the dictionnary map to replace columnns (if column are in the keys)
    return df.rename(columns=standardize_columns_dict)

# Load the .BRU files for selected stations
def load_saved_nowcast_csv(id, path_input):
    InFile = os.path.join(path_input, f"{id}_vs.csv") #id not the same as station name
    if os.path.isfile(InFile):
        df = pd.read_csv(InFile, sep=';')
        df = standardize_columns(df)
        return df
    else: # return empty pandas dataframe
        return pd.DataFrame()

# Combine the .BRU and nowcast data
def concatenate_bru_nowcast(bru_df, nowcast_df):
    bru_df_indexed = bru_df.assign(Date=pd.to_datetime(bru_df['Date'])).set_index('Date')
    nowcast_df_indexed = nowcast_df.assign(Date=pd.to_datetime(nowcast_df['Date'])).set_index('Date')
    # replace missing data from bru_df with that of nowcast_df
    combined_df = bru_df_indexed.combine_first(nowcast_df_indexed).reset_index()
    return combined_df

# Save the combined data as .csv
def save_dataframe_to_csv(forecast_df: pd.DataFrame, save_path: str, filename: str):
    out = f"{save_path}\\{filename}.csv"
    print(f'Saving bru+nowcast to : {out}')
    forecast_df.to_csv(out, index=False, sep=';',na_rep=np.nan)


# Main execution ---------------------------------------

def main():
    # Load the configuration file
    config = load_config('ec_config.json') # load config file that contains paths to folders
    save_path = config['Paths']["SavedEcVsForecastsPath"] # define as a variable the path to the data

    # set up parameters
    sel_year = datetime.datetime.now().strftime("%Y") # get current year (in string)
    # Load station info
    dat_file = "C:\\Users\\sebastien.durocher\\PycharmProjects\\GetWeatherData\\source\\Forecasts\\VStations.dat"
    stations_info = pd.read_csv(dat_file, skiprows=2)

    for i, _ in stations_info.iterrows():
        sel_station_name = stations_info.loc[i,'Name']
        sel_station_id = stations_info.loc[i,'ID']

        nowcast_df = load_saved_nowcast_csv(sel_station_id, save_path) # get the nowcast
        bru_df = standardize_columns(download_and_process_data([sel_station_name], [sel_year])) # get the bru data
        combined_df = concatenate_bru_nowcast(bru_df, nowcast_df) # merge together
        save_dataframe_to_csv(combined_df, save_path, f"{sel_station_id}_bru_nowcast") # save as csv

if __name__ == "__main__":
    main()

# TODO : Check what happens if file exists for nowcast (like RIMpro nowcast) but doesnt exist in .BRU
# TODO : Check what happens if file exists for .BRU but doesnt exist for nowcast
# TODO : Check what happens if file exists for neither nowcast nor .BRU
# TODO : Remove absolute path and replace with something more universal / flexible

