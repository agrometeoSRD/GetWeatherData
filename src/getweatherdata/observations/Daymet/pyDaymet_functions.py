#!/usr/bin/env
"""
Creation date: 2022-03-23
Creator : sebastien durocher 
Python version : ''

Description:

Updates:

Notes:

To-do:
 - Find a way to integrate timerange, kinda not clear how dates are supposed to be integrated in this.
"""

# imports
import geopandas as gpd
import pandas as pd
import os
import pydaymet as daymet

def create_geodataframe():
    # dataframe must have : id, start, end and geometry
    # id is the filename for saving the data
    # NO NEED TO DEFINE VARIABLES HERE
    # `time_scale``: (optional) Time scale, either ``daily`` (default), ``monthly`` or ``annual``
    north = 49.62
    south = 44.356
    east = -64.64
    west = -79.467
    timescale = 'monthly'
    startyear = 2000
    endyear   = 2020
    startmonth = '05-01'
    endmonth = '09-31'
    # start_dates =
    df = pd.DataFrame({'id':'test','time_scale':timescale,
                      'Latitude':[north,north,south,south],'Longitude':[east,west,east,west]})
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
    # define crs
    gdf = gdf.set_crs("epsg:4326")
    return gdf

def save_geodataframe(gdf):
    # output must be either .shp or .gpkg
    ext = "gpkg"
    outpath = "D:\\Weather_Grid\\Daymet\\"
    fname = f"geo.{ext}"
    gdf.to_file(outpath+fname)

def execute_commandline():
    script_path = ""
    type = 'geometry' # coords for pixel and geometry for area
    file = "D:\\Weather_Grid\\Daymet\\geo.gpkg"
    os.system(f"{script_path} {type} {file} -v prcp -v tmin -v tmax")

def get_data():
    geometry = create_geodataframe().geometry
    var = ["prcp", "tmin"]
    dates = ("2000-01-01", "2000-06-30")
    daily = daymet.get_bygeom(geometry, dates, variables=var, pet="priestley_taylor", snow=True)
    monthly = daymet.get_bygeom(geometry, dates, variables=var, time_scale="monthly")