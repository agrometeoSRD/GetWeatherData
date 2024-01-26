"""
File: Get_Daymet_Area.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description: Load daymet from two sources : local (download from internet if not avaiable) and planetary computer (microsoft)
Created: 2024-01-26
"""

# todo : add options to run as module or cli

# Import statements ------------------------------------------------------
import json
import urllib
import pystac
import fsspec
import xarray as xr

# Constants --------------------------------------------------------------

# Functions --------------------------------------------------------------

# Read request file (input : years, lat, lon, variables)

# Define boundary box to have matching coordinates between planetary and local

# Download from website (for local)
def define_parameters():
    # %% Downloads daily data of Daymet from ORNL DAAC
    STARTYEAR = 1980
    ENDYEAR   = 1985
    # NO_NAME = "NULL"
    # YEAR_LINE = "years:"
    # VAR_LINE  = "variables:"
    DAYMET_VARIABLES = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp'] # Not sure if tmean could work
    # DAYMET_VARIABLES = ['tmin', 'prcp']
    # DAYMET_VARIABLES = ['tmean'] # N
    DAYMET_YEARS     = [str(year) for year in range(STARTYEAR, ENDYEAR + 1)]
    north = 55
    south = 44.991
    east = -63.551
    west = -79.579
    s_stride = 1 #default is 1 to get all data
    t_stride = 1 #default is 1 to get all data
    format = "netcdf"
    return params

def test_url():
    url = create_url('dayl', 1980, 56, 44.991, -63.551, -65, 1, 1, 'netcdf')
    print(url)

def create_url(variable, year, north, south, east, west, s_stride, t_stride, format) -> str:
    """Create the url for the DAYMET data."""
    base_url = "https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/2129/"
    version_str = f"daymet_v4_daily_na_{variable}_{year}.nc?var=lat&var=lon&"
    var_area_str = (f"var={variable}&north={north}&west={west}&east={east}&south={south}"
                    f"&disableProjSubset=on&horizStride={s_stride}&")
    time_str = f"time_start={year}-01-01T12%3A00%3A00Z&time_end={year}-12-31T12%3A00%3A00Z&timeStride={t_stride}&"
    return f"{base_url}{version_str}{var_area_str}{time_str}accept={format}"

# Save download configuration file from website
def save_config(config_file, config_data):
    """Save configuration data to a JSON file."""
    with open(config_file, 'w') as file:
        json.dump(config_data, file)
    print(f"Configuration saved to {config_file}")

def read_config(config_file):
    """Read configuration data from a JSON file."""
    if os.path.exists(config_file):
        with open(config_file, 'r') as file:
            return json.load(file)
    else:
        return {}
def download_daymet():
    params = define_parameters()
    OutPath = "E:\\Weather_Grid\\Daymet\\Daily\\"
    config_file = "path_to_your_config_file.json"  # Define the path to your config file

    config_data = read_config(config_file)
    for DAYMETVAR in params['DAYMET_VARIABLES']:
        for YEAR in params['DAYMET_YEARS']:
            print(f"Requesting daily data for {DAYMETVAR} ; {YEAR}")

            # Update paths
            daymet_download_url = create_url(DAYMETVAR, YEAR, north, south, east, west, s_stride, t_stride, format)
            OutName = f'DAYMET{Project_Name}_{DAYMETVAR}_{YEAR}.nc'

            # Check if the URL is already in the config file
            if daymet_download_url in config_data:
                print(f"Data for {DAYMETVAR} {YEAR} already downloaded.")
                continue

            # Download data
            urllib.request.urlretrieve(DAYMET_FULL_PTH, OutPath + OutName)
            print(f"Saving to :", OutPath + OutName)

            # save configuration
            config_data[daymet_download_url] = {'filename': OutName}
            save_config(config_file, config_data)



# Load from planetary computer
def load_daymet_planetary():
    # Load Daymet data with Microsoft computer thing (must import aiohttp and zarr)
    # Tutorial : https://planetarycomputer.microsoft.com/dataset/daymet-daily-na#Example-Notebook
    # Adress : https://planetarycomputer.microsoft.com/dataset/group/daymet#north_america
    # TODO : make this to load on GetWeatherData
    print('Loading Daymet dataset')
    url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/daymet-daily-na"
    collection = pystac.read_file(url)
    asset = collection.assets["zarr-https"]
    store = fsspec.get_mapper(asset.href)
    ds = xr.open_zarr(store, **asset.extra_fields["xarray:open_kwargs"])
    # ds = ds.sel(nv=0,drop=True)
    return ds


# Load from local




# Main execution ---------------------------------------

if __name__ == "__main__":
    pass
