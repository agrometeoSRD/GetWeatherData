"""
File: load_areas.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description: This is a description of what the script does
Created: 2024-04-02
"""

# Import statements
from utils.utils import load_config
import geopandas as gpd

# Constants

# Functions
def load_basemaps(config):
    path_fadq = config['Paths']["QC_GIS_path"]
    munic_shp_filename = "\FADQ_munics_s_ForMap.shp"
    fadq_gdf = gpd.read_file(path_fadq + munic_shp_filename)
    # define crs as epsg 4326
    fadq_gdf = fadq_gdf.to_crs(epsg=4326)

    # not implemented yet
    # shp_Munic_Path = r"C:\\Users\\sebastien durocher\\OneDrive - IRDA\\Projet FADQ\\GIS_Region\\QC_Administrative_Regions\\SHP\\"
    # shp_Munic_Filename = "munic_s.shp"
    # munic_gdf = gpd.read_file(shp_Munic_Path + shp_Munic_Filename) # Not loaded, but could be used instead of fadq_df

    # not implemented yet
    # shp_Water_Path = r"C:\\Users\\sebastien durocher\\OneDrive - IRDA\\Projet FADQ\\GIS_Region\\QC_Administrative_Regions\\Reseau_National_Hydrog\\"
    # shp_Water_Filename = "Slice_hydro_s.shp"
    # water_gdf = gpd.read_file(shp_Water_Path + shp_Water_Filename)

    return fadq_gdf


# Main execution ---------------------------------------

if __name__ == "__main__":
    config = load_config('ec_config.json')
    load_basemaps(config)
    pass
