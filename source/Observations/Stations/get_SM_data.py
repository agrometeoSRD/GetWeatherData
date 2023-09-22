import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from typing import Union
import pathlib


# open csv file from the same folder as this script
# TODO : maybe put this in a function or something
station_filename = r"reseau_agro.csv"
p = pathlib.Path(__file__).with_name(station_filename)
with p.open('r') as f:
    print(f.read())
# Create condition if the file doesn't exist
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

