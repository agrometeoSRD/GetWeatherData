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

Created: 2024-02-27
"""

# Import statements
import os
import pandas as pd
import numpy as np
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
name_id_dict = {'Compton': 'COMPTN', 'Dunham': 'DUNHM'}

def main():
    # Load the configuration file
    config = load_config('ec_config.json') # load config file that contains paths to folders
    save_path = config['Paths']["SavedEcVsForecastsPath"] # define as a variable the path to the data
    # set up parameters
    sel_station = ['Compton']
    sel_years = ['2024']

    nowcast_df = load_saved_nowcast_csv(name_id_dict[sel_station[0]], save_path)
    bru_df = standardize_columns(download_and_process_data(sel_station, sel_years))
    combined_df = concatenate_bru_nowcast(bru_df, nowcast_df)
    save_dataframe_to_csv(combined_df, save_path, f"{name_id_dict[sel_station[0]]}_bru_nowcast")

if __name__ == "__main__":
    main()

# TODO : Create a function that will combine bru and hrdps for every available saved nowcast file
# TODO : Check what happens if file exists for nowcast (like RIMpro nowcast) but doesnt exist in .BRU
# TODO : Check what happens if file exists for .BRU but doesnt exist for nowcast
# TODO : Check what happens if file exists for neither nowcast nor .BRU
