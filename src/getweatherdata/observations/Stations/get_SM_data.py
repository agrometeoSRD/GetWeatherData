"""
File: get_SM_data.py
Author: Sebastien Durocher
Python: 3.10
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/agrometeoSRD

Description:
    This script fetches and processes weather station data from the IRDAMETEO server.
    On the server, the weather data is stored in `.BRU` files. This is a specific format that needs to be modified for proper extraction and processing.

    The script allows for the loading and concatenation of multiple `.BRU` files into a single pandas DataFrame.
    It also provides functionality to retrieve a list of available stations, select specific stations and years of interest, and save the processed data in a standardized format.

Features:
    - Downloads weather station data that was downloaded from SM server
    - Supports fetching data for multiple stations and years, combining them into a single dataset.
    - Converts time data to Eastern Standard Time (UTC-5) without considering Daylight Saving Time (DST).
        (In other words this script gets station data from /CIPRA/ and converts it to EST)
    - Processes weather data by creating appropriate datetime columns, handling missing data, and converting units (e.g., solar radiation).
    - Includes functionality to find the nearest station to a given coordinate or retrieve stations within a specific geographic area (not fully tested).

Usage:
    - The script can be executed from the command line with specific stations and years provided as arguments.
      Example:  python -m src.getweatherdata.observations.Stations.get_SM_data --stations "Compton" "Dunham" --years "2020" "2021" --save_path "./" --filename "test"
        Note : must be within the location of the script to execute this command
        Note2 : See end of description for more examples
    - The output will be saved as a CSV file, with a "standardized" format

Notes:
    - The data is fetched from the cipra directory, which does not account for Daylight Saving Time, and all times are assumed to be in UTC-5. A conversion is done here to convert into local times
    - The script automatically handles the conversion of special values (e.g., -991) to NaN for cleaner data processing.
    - Ensure that the correct station names and years are provided, as the script relies on these to construct the URLs for data retrieval.
    - The output CSV file will contain columns like Date, Temperature, Humidity, Precipitation, Solar Radiation, etc., based on the `.BRU` file format.
        - For more infromation on the .BRU file format, see : https://meteo.irda.qc.ca/Format%20fichiers%20meteo.pdf

TODO:
    - Implement additional data processing steps to handle specific edge cases or anomalous data points.
    - Add more options for saving data, such as different formats (e.g., NetCDF, Excel) or customizable output structures.
    - Integrate a progress bar to monitor the download and processing of large datasets.

Dependencies:
    - The script has been tested with Python 3.10 and may require adjustments for compatibility with other versions.

Inputs when running from main:
    - The input is a list of station names and years for which data should be retrieved, provided via command-line arguments.
    - The configuration file (`sm_config.json`) should be available within the project directory, as it defines the columns and types expected in the `.BRU` files.

Outputs:
    - The processed weather data is saved as a CSV file, with a standardized format suitable for further analysis or reporting.

Example CLI Usage:
    - The script can be executed using a command line interface, with station names and years passed as arguments.
       python -m src.getweatherdata.observations.Stations.get_SM_data --stations "Compton" "Dunham" --years "2020" "2021" --save_path "./" --filename "test"
       python -m src.getweatherdata.observations.Stations.get_SM_data --stations "Baie-Comeau" --years "2005" "2020" --save_path "C://tmptest//" --filename "Baie-Comeau_2005_2020"
        In this example, Baie-Comeau station didnt exist in 2005, therefore only 2020 data was obtained

Created: 2024-02-21
"""

import os
from ...utils.utils import load_config
import argparse
import urllib
import logging
import unidecode
import datetime
from typing import Dict,Union, Type, List, Any
import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Constants
# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Functions
# def load_config(config_path=None):
#     try:
#         if config_path is None:
#             # Access the default configuration as a package resource
#             with resources.open_text('source.observations.Stations', 'sm_config.json') as f:
#                 return json.load(f)
#         else:
#             # load configuration from a user-specified path
#             with open(config_path, 'r') as f:
#                 return json.load(f)
#     except FileNotFoundError:
#         logging.error('Configuration file not found')
#
#     except Exception as e:
#         logging.error(f"An error occured while loading the configuration file: {str(e)}")
#

def set_column_types(config):
    column_types = {col: float for col in config["BRU_num_headers"]}
    column_types.update({col: str for col in config["BRU_date_headers"]})
    return column_types


def convert_to_eastern(df, datetime_col='Date'):
    """
    Converts the datetime column of a pandas DataFrame from Eastern Time to
    Eastern Daylight Time or Eastern Standard Time as appropriate,
    considering daylight saving adjustments.

    Parameters:
    df (pandas.DataFrame): The dataframe containing the datetime column.
    datetime_col (str): The name of the datetime column in the dataframe.

    Returns:
    pandas.DataFrame: A new dataframe with the converted datetime column.
    """

    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Localize the timezone to EST (ignoring DST)
    df[datetime_col] = df[datetime_col].dt.tz_localize('EST', ambiguous='infer')

    # Convert to Eastern Time (automatically handles DST)
    df[datetime_col] = df[datetime_col].dt.tz_convert('America/New_York')

    # Remove timezone information
    df[datetime_col] = df[datetime_col].dt.tz_localize(None)

    return df

def get_SM_data_file() -> pd.DataFrame:
    """
    Load list of all station files that contain names and coordinates of all stations.
    returns a dataframe with the station names, coordinates, and other metadata.

    Note: These files may become outdated eventually. See https://www.agrometeo.org/weather/help for updated files.
    """

    # Define the specific date
    specific_date = datetime.datetime(2024, 8, 1) # update to the current time when the file is updated
    current_date = datetime.datetime.now()

    # Calculate the difference in months
    difference_in_months = (current_date.year - specific_date.year) * 12 + current_date.month - specific_date.month

    # Check if the difference is greater than or equal to 6 months
    if difference_in_months >= 6:
        logging.warning("It has been more than 6 months since station metadata has been uploaded to meteo.irda. Consider updating the files in the URLs.")


    am_station_file_url = "http://meteo.irda.qc.ca/reseau_agrometeo.csv"
    pomme_station_file_url = "http://meteo.irda.qc.ca/reseau_pommiers.csv"
    am_station_file = pd.read_csv(am_station_file_url,encoding='latin1')
    pomme_station_file = pd.read_csv(pomme_station_file_url,encoding='latin1')
    # merge based on ID, NOM, Lat, Lon, Réseau
    station_file = pd.concat([am_station_file,pomme_station_file]).reset_index(drop=True)
    return station_file


def get_stations_within_area(area_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Finds weather stations within a specified geographic area.

    Note : this function is not fully tested

    Parameters:
    area_gdf (geopandas.GeoDataFrame): A GeoDataFrame representing the geographic area of interest.
                                       The GeoDataFrame should have a geometry column with polygon geometries.

    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame containing the weather stations that fall within the specified area.
                            The returned GeoDataFrame includes all columns from the original stations data,
                            except for the geometry and index_right columns.
    """
    # Retrieve the list of all weather stations
    stations_df = get_SM_data_file()
    # Convert station coordinates to Point objects
    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=gpd.points_from_xy(stations_df.Lon, stations_df.Lat),
        crs='EPSG:4326')

    # Find stations within the area
    stations_within_area = (gpd.sjoin(stations_gdf, area_gdf, predicate='within')
                            .drop(columns=['geometry','index_right']))
    return stations_within_area

def find_nearest_station(lat: float, lon: float):
    """
    Finds the nearest weather station to a given latitude and longitude.
    Output is a series of metadata containing name, latitude, longitude, types of variables, etc. for the nearest station

    Parameters:
    lat (float): The latitude of the point of interest.
    lon (float): The longitude of the point of interest.

    Returns:
    geopandas.GeoSeries: A GeoSeries containing the information of the nearest weather station, including its distance from the input point.

    Notes:
    - A warning may occur because a projected coordinate system is used instead of a reference coordinate system. However, this is acceptable for the current level of precision.
    """

    # Retrieve the list of all weather stations
    stations_df = get_SM_data_file()

    # Convert station coordinates to Point objects
    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=gpd.points_from_xy(stations_df.Lon, stations_df.Lat),
        crs='EPSG:4326'
    )

    # Create a Point object for the input latitude and longitude
    input_point = Point(lon, lat)
    # Calculate the distance from the input point to each station
    stations_gdf['distance'] = stations_gdf['geometry'].distance(input_point)

    # Find the station with the minimum distance
    nearest_station = stations_gdf.loc[stations_gdf['distance'].idxmin()]

    # Return the nearest station's information
    return nearest_station


def convert_time(df:pd.DataFrame) -> pd.DataFrame:
    df['Hour'] = df['Hour'].astype(str).str.zfill(4) # pad with 0s
    df = (df.assign(Date =df['Year'].astype(str) + '-' + df['Day'].astype(str) + '-' + df['Hour'])
            .assign(Date = lambda df: pd.to_datetime(df['Date'], format='%Y-%j-%H%M',errors='coerce'))
            .drop(columns=['Year', 'Day', 'Hour'])
          )
    # palce datetime as first column
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]
    # convert normal time to local (normal / eastern)
    df = convert_to_eastern(df, 'Date')
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

def rename_columns(df:pd.DataFrame,config:Dict) -> pd.DataFrame:
    pass

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

    # convert solar radiation kj/h*m2 into W/m2
    df['InSW'] = (df['InSW'] * 1000 / 3600).round(3)

    return df

def download_and_process_data(station_names: List[str], years: List[str], config: Dict = None):
    if config is None:
        config = load_config('sm_config.json')

    df_list = []
    # Replace blank spaces with underscores for names. Remove accents as well
    station_names = [unidecode.unidecode(name.replace(' ', '_')) for name in station_names]

    BRU_headers = config["BRU_date_headers"] + config["BRU_num_headers"]
    for station in station_names:
        for year in years:
            url = create_url(station, year)
            df = fetch_data(url, BRU_headers, set_column_types(config))
            df['name'] = station
            df = process_data(df)
            df_list.append(df)

    df_all_stations = pd.concat(df_list).dropna(subset=['Date']).reset_index(drop=True)
    return df_all_stations

def save_data(df: pd.DataFrame, save_path: str, filename: str) -> None:
    """
    Saves the weather station data
    """
    # Check if the directory exists, if not, create it
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Save as csv
    out = f"{save_path}\\{filename}.csv"
    logger.info(f'Saving station file to : {out}')
    df.to_csv(out, index=False, na_rep=np.nan)

# Main execution ---------------------------------------
def main(args):
    config = load_config('sm_config.json')
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
