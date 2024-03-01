"""
File: combine_bru_hrdps.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747

Description: Some .BRU dont have solar radiation. Use HRDPS solar radiation to fill in the missing data
- For code to work, first must have .BRU files and HRDPS solar radiation forecast
- .BRU files can be accessed by imported get_SM_data.py
- HRDPS files are locally saved in a folder. These are created by fisrt downloading ec_forecasts.py and then saving with save_ec_nowcast.py

Created: 2024-02-27
"""

# Import statements
import os
import pandas as pd
import json
from source.Observations.Stations.get_SM_data import download_and_process_data
from source.utils.utils import invert_mapping
# Constants

# Functions
# Check which stations have solar radiation


# Load the .BRU files for selected stations
def load_saved_nowcast_csv(id, path_input):
    InFile = os.path.join(path_input, f"{id}_vs.csv") #id not the same as station name
    return pd.read_csv(InFile, sep=';')

# Load the HRDPS solar radiation data for selected stations

# Combine the .BRU and HRDPS data

# Save the combined data as .csv
def standardize_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    with open('column_names.json', 'r') as file:
        universal_mapping = json.load(file)
        inverted_mapping = invert_mapping(universal_mapping)

    standardized_cols = {col: mapping.get(col, col) for col in df.columns}
    return df.rename(columns=standardized_cols)

# Main execution ---------------------------------------
path_to_ec_nowcasts = 'C:\\Scripts\\PycharmProjects\\GetWeatherData\\source\\Forecasts\\'
name_id_dict = {'Compton': 'COMPTN', 'Dunham': 'DUNHM'}

def main():
    sel_station = ['Compton']
    sel_years = ['2020']
    bru_df = download_and_process_data(sel_station, sel_years)
    nowcast_df = load_saved_nowcast_csv(name_id_dict[sel_station[0]], path_to_ec_nowcasts)

if __name__ == "__main__":
    pass
