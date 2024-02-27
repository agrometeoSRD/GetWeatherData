"""
File: get_SM_data.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description:
- Fetches weather station data from http meteoirda/CIPRA (using CIPRA and not 4z, because CIPRA its easier to use)
- Weather station data is stored as .BRU file. This has specific formatting that needs to be taken into account
- Code allows loading of multiple .BRU files and concatenates them into a single dataframe
- To find out which stations are available, use a specific file that stores the information (reseau_sm.csv)
- Stored as a pandas and saved following a specific weather station format

Notes
- By using CIPRA, we do not consider advance eastern time. Time is always UTC-5
- To test directly on the CLI
python script_name.py --stations "Compton" "Dunham" --years "2020" "2021" --save_path "./" --filename "test"

Created: 2024-02-21
"""

# Import statements
import sys
import argparse
import urllib
import json
from importlib import resources # avoids hardocding the path
import logging
import unidecode
import datetime
from typing import Dict,Union, Type, List, Any
import numpy as np
import pandas as pd

# Constants
# Configure basic logging
logging.basicConfig(level=logging.INFO,format='%asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# Functions
def load_config(config_path=None):
    try:
        logging.info("Configuration loaded successfully")
        if config_path is None:
            # Access the default configuration as a package resource
            with resources.open_text('source.Observations.Stations', 'config.json') as f:
                return json.load(f)
        else:
            # load configuration from a user-specified path
            with open(config_path, 'r') as f:
                return json.load(f)
    except FileNotFoundError:
        logging.error('Configuration file not found')

    except Exception as e:
        logging.error(f"An error occured while loading the configuration file: {str(e)}")


def set_column_types(config):
    column_types = {col: float for col in config["BRU_num_headers"]}
    column_types.update({col: str for col in config["BRU_date_headers"]})
    return column_types

def convert_time(df:pd.DataFrame) -> pd.DataFrame:
    df['Hour'] = df['Hour'].astype(str).str.zfill(4) # pad with 0s
    df = (df.assign(Datetime = df['Year'].astype(str) +'-'+ df['Day'].astype(str) +'-'+ df['Hour'])
            .assign(Datetime = lambda df: pd.to_datetime(df['Datetime'], format='%Y-%j-%H%M'))
            .drop(columns=['Year', 'Day', 'Hour'])
          )
    # palce datetime as first column
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    return df

def define_station_names(station_names: list[str]) -> list[str]:
    # Function that checks to make sure names are properly formatted for the url

    # Make sure Station_names is list and names within are strings
    if isinstance(station_names, list) == False and all(isinstance(x, str) for x in station_names) == False:
        raise TypeError('Hey! Input station_names should be a list of strings.')

    # Replace blank spaces with underscores for names
    station_names = [name.replace(' ', '_') for name in station_names]
    # remove accents
    station_names = [unidecode.unidecode(el) for el in station_names]

    return station_names

def define_years(years: Union[str, List[str]] = 'all') -> List[str]:
    if years == 'all':
        return list(map(str, range(2000, int(datetime.datetime.now().year) + 1)))
    elif isinstance(years, list) and all(isinstance(x, str) for x in years):
        return years
    else:
        raise TypeError('Input years must be a list and the years within must be strings')

def create_url(station_name:str,year:str) -> str:
# will look like something like this :  url = f"http://http://meteo.irda.qc.ca//Cipra//{year}//{station_id}.BRU"
    url = f"http://meteo.irda.qc.ca/Cipra/{year}/{station_name}.BRU"
    print("URL: ",url)
    return url

def fetch_data(url:str,BRU_headers:list[str],column_types:Dict[Any, Type[float]]) -> pd.DataFrame:
    """
    Fetches weather station data from http://meteoirda.qc.ca/CIPRA
    """

    try:
        df = pd.read_csv(url, names=BRU_headers, dtype=column_types,engine='python')
    except urllib.error.HTTPError as e:
        df = pd.DataFrame(columns=BRU_headers)
        if e.code == 404:
            logger.error(f"HTTP Error 404: File Not Found. Returning empty dataframe.")
        else:
            raise Exception(f"Undocumented error occured: {str(e)}")
    return df

def process_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Processes the weather station data
    """
    #TODO : Add more processing

    # filter variables if specified

    # create date column
    df = convert_time(df)

    # convert -991 to nan
    df = df.replace(-991, np.nan)  # Convert -991 to nan

    return df

def download_and_process_data(station_names: List[str], years: List[str], config: Dict = None):
    if config is None:
        config = load_config()

    df_list = []
    BRU_headers = config["BRU_date_headers"] + config["BRU_num_headers"]
    for station in station_names:
        for year in years:
            url = create_url(station, year)
            df = fetch_data(url, BRU_headers, set_column_types(config))
            df_list.append(df)

    df_all_stations = pd.concat(df_list)
    df_all_stations = process_data(df_all_stations)
    return df_all_stations

def save_data(df:pd.DataFrame,save_path: str, filename: str) -> None:
    """
    Saves the weather station data
    """
    #TODO : Add more saving options

    # Save as csv
    out = f"{save_path}\\{filename}.csv"
    logger.info(f'Saving station file to : {out}')
    df.to_csv(out, index=False,na_rep=np.nan)

# Main execution ---------------------------------------
def main(args):
    config=load_config()
    df_all_stations = download_and_process_data(args.stations,args.years,config)
    save_data(df_all_stations,args.save_path,args.filename)

def parser_args(args=None):
    parser = argparse.ArgumentParser(description="Download and process weather station data.")
    # the + specifies that there can be multiple arguments and to store those as a list
    parser.add_argument("--stations", nargs="+",default=['Compton','Dunham'], help="List of station names")
    parser.add_argument("--years", nargs="+", default=['2020','2021'],help="List of years")
    parser.add_argument("--save_path", default='./',help="Path to save the output CSV")
    parser.add_argument("--filename", default='Compton_station',help="Filename for the output CSV (no extension)")
    return parser.parse_args(args)


if __name__ == "__main__":
    args = parser_args()
    main(args)
