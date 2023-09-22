# Currently only know one fast way to get data, and that's with env_canada_local
# Input requires a single coordinate. Does not work with a geopandas. In that case would need to get centroid of the geodataframe

import asyncio
# imports
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.spatial import distance

from env_canada_local.ec_historical import ECHistoricalRange, get_historical_stations

def get_station_coords():
    # Input requires a single distance.
    eccc_stations = pd.DataFrame(asyncio.run(get_historical_stations(coordinates, start_year=1970,
                                                                     end_year=2000, radius=200, limit=10))).T

    # drop stations that dont have data between 1979 and 2000
    eccc_stations = eccc_stations[(eccc_stations['dlyRange'].str.split('|').str[0] < '1979') & (
        (eccc_stations['dlyRange'].str.split('|').str[1] > '2000'))]