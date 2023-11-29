#!/usr/bin/env
"""
Creation date: 2023-03-29
Creator : sebastien durocher
Python version : 3.10

Description:
- Use the MSC API (oswlib) to get data from the geomet server.
- Input requires a spatial coordinate (From a station file) and output will save forecast for this coordinate within a csv file.

Updates:

Notes:
    - Unlike all past iterations of getting forecast, now the forecast script doesn't transform the data into RIMpro format
"""
#TODO : Create automatic file to get acces to station data. Add the necessary errors to make sure the data is there
#TODO : Expand the forecast to include RDPS and GDPS
#TODO : Create a seperate script that would accept the forecast data and convert it into RIMpro
#TODO : Apply asyncio to the script
#TODO : Create some kind of secondary dictionnary that associates each variable with its corresponding forecast variable, that way the user can easily specify the variables that he wants and then the code will get the corresponding layers

# imports
import os
import re
import warnings
from datetime import datetime, timedelta
from dateutil import tz

import numpy as np
import pandas as pd
from owslib.util import ServiceException
from owslib.wms import WebMapService

warnings.filterwarnings("ignore")

# new imports
import aiohttp
import asyncio
import pytz

def request(layer : str, time : datetime.date, coor : list) -> list:
    '''

    @param layer:
    @param time:
    @param coor:
    @return: list containing floats
    '''
    info = []
    pixel_value = []
    for timestep in time:
        # WMS GetFeatureInfo query
        print(timestep, layer)
        try:
            info.append(wms.getfeatureinfo(layers=[layer],
                                           srs='EPSG:4326',
                                           bbox=tuple(coor),
                                           size=(100, 100),
                                           format='image/jpeg',
                                           query_layers=[layer],
                                           info_format='text/plain',
                                           xy=(1, 1),
                                           feature_count=1,
                                           time=str(timestep.isoformat()) + 'Z'
                                           ))
            # Probability extraction from the request's results
            text = info[-1].read().decode('utf-8')
            pixel_value.append(str(re.findall(r'value_0\s+\d*.*\d+', text)))
            try:
                pixel_value[-1] = float(
                    re.sub('value_0 = \'', '', pixel_value[-1])
                        .strip('[""]')
                )
            except ValueError:
                print(
                    f'Problem with the extract data (most likely empty output) at time = {timestep} and layer = {layer}')
                print('Returning empty float instead')
                pixel_value[-1] = [np.nan]
        except ServiceException:
            print(f'Request could not be made for some reason at time = {timestep} and layer = {layer}')
            pixel_value.append(np.nan)

    return pixel_value

def get_forecast_times(layer:str,target_timezone='America/Montreal'):
    """
    Generate a list of forecast times for a given layer. Output two lists, once with time in local timezone and the other in UTC.
    """
    def convert_timezone(dt, tz):
        # Convert the datetime object to UTC
        utc_dt = dt.astimezone(pytz.utc)
        # Convert the UTC datetime object to the specified timezone
        tz_dt = utc_dt.astimezone(tz)
        return tz_dt

    # Extract start time, end time, and interval
    start_time, end_time, interval_str = wms[layer].dimensions['time']['values'][0].split('/')
    interval = int(re.findall(r'\d+', interval_str)[0])  # Possible formats are : 'PT1H', 'PT3H' or 'PT6H'

    # Parse times
    iso_format = '%Y-%m-%dT%H:%M:%SZ' # ISO 8601 format that's used by wms
    start_time = datetime.strptime(start_time, iso_format).replace(tzinfo=tz.UTC)
    end_time = datetime.strptime(end_time, iso_format).replace(tzinfo=tz.UTC)

    # Create list of times
    time_list_utc = [start_time]
    while time_list_utc[-1] < end_time:
        time_list_utc.append(time_list_utc[-1] + timedelta(hours=interval))

    # Convert list to local time
    au_tz = pytz.timezone(target_timezone)
    time_list_local = [convert_timezone(atime, au_tz).replace(tzinfo=None) for atime in time_list_utc]

    return time_list_local, time_list_utc

# ===============================
# Code start    =================
# ===============================
# Setup wms
address = "geo.weather.gc.ca/geomet?service=WMS"  # for operational use # "http://collaboration.cmc.ec.gc.ca/rpn-wms" # for experimental use #
Version = "1.3.0"
wms = WebMapService('https://geo.weather.gc.ca/geomet?SERVICE=WMS' +
                    '&REQUEST=GetCapabilities',
                    version=Version,
                    timeout=300)

# Setup the paths
Path_To_Script = r"C:\\Users\\sebastien.durocher\\PycharmProjects\\GetWeatherData\\source\\Forecasts"
# Path_To_Script = os.getcwd()
# Open file that contains path for station coordinates
paths = list()
with open(Path_To_Script + '\\Chemins_Acces.txt') as f:
    lines = f.readlines()
    for line in lines:
        li = line.strip()
        if not li.startswith("#"):
            paths.append(line.split('\n')[0])

InFile = paths[0]  # First line must be path + filename for station coordinates
Path_Output = paths[1]  # Second line must be path for outputing station files

RDPS_varlist = ['RDPS.ETA_TT', 'RDPS.ETA_HR', "RDPS.ETA_PR"]
variables = ['TT', 'HR', 'PR']
Forecast_Variables_List = ["AIRTEMP", "HR", "RAIN"]

# Check if station file names exists or not (if it doesn't, make error message)
try:
    file = open(InFile)
except Exception as e:
    print('VStations file does not exist. Check the name of file or create a new one.')

# Start reading for all stations
Stations_info = pd.read_csv(InFile, skiprows=2)
# Create box with variables by getting 2nd longitude to the east and 2nd latitude to the north
# Box order is the following : Lon1, Lat1, Lon2, Lat2
Stations_info['Lon2'] = Stations_info['Lon'] + 0.1
Stations_info['Lat2'] = Stations_info['Lat'] - 0.1


# Get RDPS
def run_rdps(coor, nb_timestep=None):
    # print("Requesting RDPS data...")
    time_local, time_utc = get_forecast_times(RDPS_varlist[0])  # setup time as a base
    pixel_value_dict_rdps = {layer: request(layer, time_utc[:nb_timestep],coor) for layer in RDPS_varlist}
    RDPS_df = pd.DataFrame.from_dict(pixel_value_dict_rdps, orient='index').transpose()
    RDPS_df['Date'] = time_local[:nb_timestep]
    RDPS_df['RDPS.ETA_PR'] = RDPS_df['RDPS.ETA_PR'].diff()
    return RDPS_df
def process_request(arg):
    # print('Inside function')
    # print(arg)
    # Doesn't work with interpreter. Multprocess (which is supposed to work with interpreter) doesn't work
    info = pd.DataFrame(arg).T
    print(f'Acquiring weather forecast for {arg.iloc[0]}')
    coor = (info[['Lon', 'Lat2', 'Lon2', 'Lat']].iloc[0].tolist())

    # Can add nb_timesteps to define how far we want to go. Must add as function argument
    nb_timestep = 24

    RDPS_df = run_rdps(coor, nb_timestep)

    return RDPS_df

Stations_info.apply(process_request, axis=1)


# def run_rdps(coor, nb_timestep=None):
#     # print("Requesting RDPS data...")
#     time_local, time_utc = get_forecast_times(RDPS_varlist[0])  # setup time as a base
#     pixel_value_dict_rdps = {layer: request(layer, time_utc[:nb_timestep],coor) for layer in RDPS_varlist}
#     RDPS_df = pd.DataFrame.from_dict(pixel_value_dict_rdps, orient='index').transpose()
#     RDPS_df['Date'] = time_local[:nb_timestep]
#     RDPS_df['RDPS.ETA_PR'] = RDPS_df['RDPS.ETA_PR'].diff()
#     return RDPS_df

class BaseForecastSource:
    def fetch_data(self, layer, time, coordinates):
        raise NotImplementedError
    def transform_to_dataframe(self, parsed_data):
        raise NotImplementedError
class RDPSForecastSource(BaseForecastSource):

    def __init__(self):
        self.nb_timestep = None
    def fetch_data(self, layer, time, coordinates):
        time_local, time_utc = get_forecast_times(layer)  # setup time as a base
        self.time_local = time_local  # Storing for later use in transform_to_dataframe
        pixel_value_dict_rdps = {layer: request(layer, time_utc[:self.nb_timestep],coordinates) for layer in RDPS_varlist}

    def transform_to_dataframe(self, pixel_value_dict_rdps):
        # RDPS-specific transformation logic
        time_local, time_utc = get_forecast_times(RDPS_varlist[0])  # setup time as a base
        RDPS_df = pd.DataFrame.from_dict(pixel_value_dict_rdps, orient='index').transpose()
        RDPS_df['Date'] = time_local[:self.nb_timestep]
        RDPS_df['RDPS.ETA_PR'] = RDPS_df['RDPS.ETA_PR'].diff()
