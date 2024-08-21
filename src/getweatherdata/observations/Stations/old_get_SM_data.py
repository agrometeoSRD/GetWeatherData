import os,glob
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import Union
import pathlib


##% Old code

# open csv file from the same folder as this script
# TODO : maybe put this in a function or something
station_filename = r"reseau_agro.csv"
p = pathlib.Path(__file__).with_name(station_filename)
# Check if file doesn't exist
if not p.exists():
 raise FileNotFoundError(f'File not found: {p}')

# Read geopolygon file
# TODO : place this line somewhere else, because don't always have to load a geopolygon. Also make it less "hardcoded"
gpd.io.file.fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
polygon_gdf = gpd.read_file(r"C:\Users\sebastien.durocher\OneDrive - IRDA\Chargé de projets\Petits projets\Marc-Olivier\Projet 10033\BV_BaieMissisquoi_Uni.kml")

def get_station_coords_within_area(csv_filepath: str, polygon_gdf: gpd.GeoDataFrame, lat_col: str ='Lat', lon_col: str='Lon') -> Union[pd.DataFrame, None]:
# lat_col = 'Lat'
# lon_col = 'Lon'
    # Read CSV file
     df = pd.read_csv(csv_filepath,sep=';')

     # Convert DataFrame to GeoDataFrame
     geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
     geo_df = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

     # Filter out points within the multipolygon
     points_in_polygon = gpd.sjoin(geo_df, polygon_gdf, how="inner", op="within")

     return points_in_polygon

ans = get_station_coords_within_area(p, polygon_gdf)

def convert_time(inputime):
    try:
        mod_time = inputime[:-3] + ''.join(str(inputime[-3:]).split('00')[:])
        # Need to do this because split takes the first 00 it sees, meaning at 10:00, it will take the first 2 zeros.
        mod_time = str(datetime.datetime.strptime(mod_time, '%Y-%j-%H'))
    except ValueError:
        print('Doesn''t work for ', inputime)
    return mod_time
def load_data():
    # This was made specifically for Marc-Olivier. Not universal. Waiting for update on the M.

    path = "D:\\observations\\Stations\\Marc-Olivier\\"
    myHeaders = ['Year', 'Day', 'Hour', 'Num', 'TMax', 'TMin', 'TMoy', 'HR', 'PR', 'PR_start', 'InSW', 'Tair_5',
              'TGr_5', 'TGr_10',
              'TGr_20', 'TGr_50', 'Wind', 'WindDir', 'Pressure', 'Mouillure_feuil', 'Mouillure_feuil_thres',
              'no_data', 'no_data2']
    # Load all .BRU files within that path as pandas csv. Add a new colum named "filename" with the name of the file
    df_l = []
    for f in glob.glob(os.path.join(path,"*.BRU")):
      df_p = pd.read_csv(f,encoding='latin-1',names=myHeaders)
      df_p['Name'] = f.split('\\')[-1].split(' ')[0]
      df_l.append(df_p)

    Station_df = pd.concat(df_l).reset_index(drop=True)
    Station_df = Station_df.assign(
     Date=Station_df.Year.astype(str) + '-' + Station_df.Day.astype(int).astype(str) + '-' +
          Station_df.Hour.astype(int).astype(str))  # Merge time together
    Station_df['Date'] = Station_df['Date'].apply(convert_time)
    Station_df = Station_df.drop(columns=['Year', 'Day', 'Hour'])  # Remove first 3 columns
    Station_df['Date'] = pd.to_datetime(Station_df['Date'])
    Station_df = Station_df.sort_values(by=['Name', 'Date']).reset_index(drop=True)
    Station_df = Station_df.replace(-991, np.nan)  # Convert -991 to nan

    # compute daily mean for TMax, TMin, TMoy, HR, Wind, WindDir, Pressure
    mean_cols = ['TMoy', 'HR', 'Wind', 'WindDir', 'Pressure']
    mean_dict = {col: 'mean' for col in mean_cols}
    # compute daily sums for Pr and InSW
    sum_cols = ['PR', 'InSW']
    sum_dict = {col: 'sum' for col in sum_cols}
    # for min and max temp, select min and max of hourly mean temperature within the day
    min_dict = {'TMin':'min'}
    max_dict = {'TMax':'max'}
    # join dictionaries
    agg_dict = {**mean_dict, **sum_dict,**min_dict,**max_dict}
    Station_df[mean_cols] = Station_df[mean_cols].astype(float)
    Station_df[sum_cols] = Station_df[sum_cols].astype(float)
    Station_df_daily = Station_df.groupby(['Name', pd.Grouper(key='Date', freq='D')]).agg(agg_dict).reset_index()

    # Some strings in column Name have .BRU at the end. Remove it
    Station_df_daily['Name'] = Station_df_daily['Name'].str.replace('.BRU', '')

    # drop missing nans
    Station_df_daily = Station_df_daily.dropna()

    # sort by dates
    Station_df_daily = Station_df_daily.sort_values(by=['Name', 'Date']).reset_index(drop=True)

    # round mean_cols and sum_cols
    Station_df_daily[mean_cols] = Station_df_daily[mean_cols].round(2)
    Station_df_daily[sum_cols] = Station_df_daily[sum_cols].round(2)

    # Get unique of Station_df['Name'] and write csv files for each station
    for station in Station_df_daily['Name'].unique():
        Station_df_daily[Station_df_daily['Name'] == station].to_csv(f"C:\\temp\\{station}.csv", index=False)
