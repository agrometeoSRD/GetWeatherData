# Note : this script is not working as expected. Loading time speed has not increased.
# Not recommended to use this script to download forecast data (see ec_forecasts.py)

# imports
import os
import re
import warnings
from datetime import datetime, timedelta
from functools import reduce
import time
import pytz
import pandas as pd
from owslib.util import ServiceException
from owslib.wms import WebMapService

warnings.filterwarnings("ignore")
import aiohttp
import asyncio
import logging
import configparser

wms_url = 'https://geo.weather.gc.ca/geomet/?SERVICE=WMS&REQUEST=GetFeatureInfo'
wms = WebMapService(wms_url, version='1.3.0', timeout=300)
common_var_names = ['TT', 'HR', 'PR', 'N4']  # These are the common variable names between the different forecast models
rain_col = "RAIN [mm]"
temp_col = "AIRTEMP [C]"
hr_col = "HR [%]"
rad_col = "GLOBALRAD [Wm2]"
forecast_variables = [temp_col, hr_col, rain_col, rad_col]

import concurrent.futures
import functools  # at the top with the other imports
async def request(session:aiohttp.ClientSession,layer: str, times:list, coor: list) -> list:
    pixel_values = []
    for timestep in times:
        print(timestep, layer)
        try :
            loop = asyncio.get_event_loop()
            response_object = await loop.run_in_executor(
                None,
                functools.partial(
                    wms.getfeatureinfo,
                    layers=[layer],
                    srs='EPSG:4326',
                    bbox=tuple(coor),
                    size=(100, 100),
                    format='image/jpeg',
                    query_layers=[layer],
                    info_format='text/plain',
                    xy=(1, 1),
                    feature_count=1,
                    time=timestep.isoformat() + 'Z'
                )
            )
            url = wms.request
            print(url)
            async with session.get(url) as resp:
                if resp.status == 200:
                    text = await resp.text()
                    value_str = re.search(r'value_0\s+\d*.*\d+', text)
                    if value_str:
                        pixel_values.append(re.sub('value_0 = \'', '', value_str.group()).strip('[""]'))
                else:
                    print(f'Request could not be made for some reason at time = {timestep} and layer = {layer}')
                    pixel_values.append(float('nan'))

        except ServiceException:
            print(f'Request could not be made for some reason at time = {timestep} and layer = {layer}')
            pixel_values.append(float('nan'))

    return pixel_values


# Extraction of temporal information from metadata
def time_parameters(layer):
    start_time, end_time, interval = (wms[layer]
                                      .dimensions['time']['values'][0]
                                      .split('/')
                                      )
    iso_format = '%Y-%m-%dT%H:%M:%SZ'
    start_time = datetime.strptime(start_time, iso_format)
    end_time = datetime.strptime(end_time, iso_format)
    interval = int(re.sub(r'\D', '', interval))
    return start_time, end_time, interval

def setup_time(layer):
    '''
    This is what's proposed by ECCC in their tutorial
    :param layer:
    :return:
    '''
    # Setup time
    au_tz = pytz.timezone('America/Montreal')
    # get numerical time zone based on location (in this example, the local time zone would be UTC-05:00):
    start_time, end_time, interval = time_parameters(layer)

    # Calculation of date and time for available predictions
    # (the time variable represents time at UTC±00:00)
    time_utc = [start_time]
    while time_utc[-1] < end_time:
        time_utc.append(time_utc[-1] + timedelta(hours=interval))

    # Convert time to local time zone
    time_local = [t.replace(tzinfo=pytz.utc).astimezone(au_tz).replace(tzinfo=None) for t in time_utc]

    return time_local, time_utc


async def run_hrdps(session, coor: list, nb_timestep: dict,date_col:str) -> pd.DataFrame:
    print('Getting HRDPS')
    HRDPS_varlist = ['HRDPS.CONTINENTAL_TT', 'HRDPS.CONTINENTAL_HR', 'HRDPS.CONTINENTAL_PR', 'HRDPS.CONTINENTAL_N4']
    time_local, time_utc = setup_time(HRDPS_varlist[0])

    pixel_value_dict_HRPDS = await asyncio.gather(*[request(session, layer, time_utc[:nb_timestep['HRDPS']], coor) for layer in HRDPS_varlist])
    hrdps_df = pd.DataFrame.from_dict(dict(zip(HRDPS_varlist,pixel_value_dict_HRPDS)), orient='index').transpose()
    hrdps_df[date_col] = time_local[:nb_timestep['HRDPS']]
    hrdps_df['HRDPS.CONTINENTAL_PR'] = hrdps_df['HRDPS.CONTINENTAL_PR'].diff().clip(lower=0)
    hrdps_df['HRDPS.CONTINENTAL_N4'] = (hrdps_df['HRDPS.CONTINENTAL_N4'] / 3600).diff().clip(lower=0)

    return hrdps_df

async def run_rdps(session, coor: list, nb_timestep: dict,date_col:str):
    print('Getting RDPS')
    RDPS_varlist = ['RDPS.ETA_TT', 'RDPS.ETA_HR', 'RDPS.ETA_PR', 'RDPS.ETA_N4']
    time_local, time_utc = setup_time(RDPS_varlist[0])

    pixel_value_dict_rdps = await asyncio.gather(*[request(session, layer, time_utc[nb_timestep['HRDPS']:nb_timestep['RDPS']], coor) for layer in RDPS_varlist])
    rdps_df = pd.DataFrame.from_dict(dict(zip(RDPS_varlist,pixel_value_dict_rdps)), orient='index').transpose()
    rdps_df[date_col] = time_local[nb_timestep['HRDPS']:nb_timestep['RDPS']]
    rdps_df['RDPS.ETA_PR'] = rdps_df['RDPS.ETA_PR'].diff().clip(lower=0)
    rdps_df['RDPS.ETA_N4'] = (rdps_df['RDPS.ETA_N4'] / 3600).diff().clip(lower=0)  # remove possible negative values
    return rdps_df

async def run_gdps(session, coor: list, nb_timestep: dict,date_col:str):
    print('Getting GDPS')
    GDPS_varlist = ['GDPS.ETA_TT', 'GDPS.ETA_HR', 'GDPS.ETA_PR', 'GDPS.ETA_N4']
    time_local, time_utc = setup_time(GDPS_varlist[0])
    start_idx = [idx for idx,dt in enumerate(time_utc) if dt == (time_utc[0] + timedelta(hours=nb_timestep['RDPS']))][0]
    # pixel_value_dict_GDPS = await asyncio.gather(*[request(session, layer, time_utc[start_idx:], coor) for layer in GDPS_varlist], return_exceptions=True)
    pixel_value_dict_GDPS = []
    for layer in GDPS_varlist:
        result = await request(session, layer, time_utc[start_idx:], coor)
        pixel_value_dict_GDPS.append(result)

    gdps_df = pd.DataFrame.from_dict(dict(zip(GDPS_varlist, pixel_value_dict_GDPS)), orient='index').transpose()
    gdps_df[date_col] = time_local[start_idx:]
    gdps_df['GDPS.ETA_PR'] = gdps_df['GDPS.ETA_PR'].diff().clip(lower=0)
    gdps_df['GDPS.ETA_N4'] = (gdps_df['GDPS.ETA_N4'] / (3*3600)).diff().clip(lower=0)

    return gdps_df

# Similar changes for run_hrdps and run_rdps...

async def process_request(session, station_info: pd.Series,date_col:str) -> dict:
    print(f'Acquiring forecast for station : {station_info["Name"]}')
    coor = [station_info['Lon'], station_info['Lat2'], station_info['Lon2'], station_info['Lat']]
    # timesteps_dict = {'HRDPS': 48, 'RDPS': 84, 'GDPS': 120}
    timesteps_dict = {'HRDPS': 48, 'RDPS': 3, 'GDPS': 10}

    # hrdps_df = pd.DataFrame()
    # rdps_df = pd.DataFrame()
    gdps_df = await asyncio.gather(run_gdps(session,coor,timesteps_dict,date_col))
    # hrdps_df, rdps_df, gdps_df = await asyncio.gather(
    #     run_hrdps(session, coor,timesteps_dict, date_col),
    #     run_rdps(session, coor,timesteps_dict, date_col),
    #     run_gdps(session, coor,timesteps_dict, date_col)
    # )

    # return {'RDPS': rdps_df, 'GDPS': gdps_df, 'HRDPS': hrdps_df}
    return gdps_df

async def main(config_path):
    start_time = time.time()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load configuration
    config = configparser.ConfigParser()
    config.read(config_path)

    path_to_script = config.get('Paths', 'EcScriptPath')
    path_to_save = config.get('Paths', 'SavePath')
    date_col = config.get('General', 'DateColumn')

    # Load station info
    InFile = os.path.join(path_to_script, 'VStations_test.dat')
    try:
        Stations_info = pd.read_csv(InFile, skiprows=2)
    except Exception as e:
        logger.error(f"Error reading file {InFile}: {e}")
        return

    # Additional processing
    Stations_info['Lon2'] = Stations_info['Lon'] + 0.1
    Stations_info['Lat2'] = Stations_info['Lat'] - 0.1

    connector = aiohttp.TCPConnector(force_close=True)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [process_request(session, row, date_col) for _, row in Stations_info.iterrows()]
        results = await asyncio.gather(*tasks)

    # Process results...
    print('Here is the forecast results')
    print(results)

    elapsed_time = time.time() - start_time
    logger.info(f"Script completed in {elapsed_time} seconds")

if __name__ == "__main__":
    # Define path to configuration file
    config_file_path = f'C:\\Users\\{os.getenv("USERNAME")}\\PycharmProjects\\GetWeatherData\\source\\Forecasts\\config.ini'
    asyncio.run(main(config_file_path))