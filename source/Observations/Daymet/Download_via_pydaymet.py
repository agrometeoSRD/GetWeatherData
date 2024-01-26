import pydaymet as daymet
import os
import json
from shapely.geometry import Polygon
import geopandas as gpd

# geometry = Polygon([[-69.77, 45.07], [-69.31, 45.07], [-69.31, 45.45], [-69.77, 45.45], [-69.77, 45.07]])
geometry_file = f"C:\\Users\\{os.getenv('USERNAME')}\\OneDrive - IRDA\\GIS\\PAVICS\\RegionAgricolesQC.geojson"
gdf = gpd.read_file(geometry_file)
# combine all geometries into a single MultiPolygon
geometry = gdf.unary_union
boundary = geometry.bounds

vars = ["srad","tmin","tmax","prcp"]
for var in vars:
    for yr in range(1989, 2020):
        raw_daymet = daymet.get_bygeom(geometry, yr, variables=var,region='na',time_scale="daily")
        raw_daymet.to_netcdf(f"{output_folder_temp}/{yr}.nc",mode = "w")
        raw_daymet.close()