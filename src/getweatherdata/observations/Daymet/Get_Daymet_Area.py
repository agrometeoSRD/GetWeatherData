"""
File: Get_Daymet_Area.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description: Load daymet from two sources : local (download from internet if not avaiable) and planetary computer (microsoft)
- Default download area is that of Quebec agricole
Created: 2024-01-26
"""

# todo : add options to run as module or cli
# todo : add dimensions of the data to the log file (lat, lon, time)

# Import statements ------------------------------------------------------
import os
import json
import urllib
import pystac
import fsspec
import xarray as xr


# Constants --------------------------------------------------------------

# Functions --------------------------------------------------------------

# define metadata parameters (variables, years, coordinates)
def define_parameters(variables: list, start_year: int, end_year: int, dimensions: dict = None) -> dict:
    available_variables = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']

    # Check if all variables exist in the list of available variables
    if not all(var in available_variables for var in variables):
        raise ValueError("One or more variables do not exist in the list of available variables.")

    # If dimensions is a dictionary, check if it has the correct keys
    if isinstance(dimensions, dict) and not all(key in dimensions for key in ['north', 'south', 'east', 'west']):
        raise ValueError("Dimensions must be a dictionary with keys 'north', 'south', 'east', and 'west'.")

    if not dimensions:
        print("No dimensions provided. Using default values for Quebec agricole.")
        dimensions = {
            'north': 55,
            'south': 44,
            'east': -64,
            'west': -80
        }

    params = {
        'daymet_variables': variables,
        'years': [str(year) for year in range(start_year, end_year + 1)],
        'north': dimensions['north'],
        'south': dimensions['south'],
        'east': dimensions['east'],
        'west': dimensions['west'],
        's_stride': 1,
        't_stride': 1,
        'format': "netcdf"
    }
    return params

# Read request file (input : years, lat, lon, variables)

# Define boundary box to have matching coordinates between planetary and local

# Download from website (for local)

# def test_url():
#     url = create_url('dayl', 1980, 56, 44.991, -63.551, -65, 1, 1, 'netcdf')
#     print(url)

def create_url(variable, year, params):
    base_url = "https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/2129/"
    url_params = {
        'var': f"{variable}&north={params['north']}&west={params['west']}&east={params['east']}&south={params['south']}",
        'horizStride': params['s_stride'],
        'time_start': f"{year}-01-01T12%3A00%3A00Z",
        'time_end': f"{year}-12-31T12%3A00%3A00Z",
        'timeStride': params['t_stride'],
        'accept': params['format']
    }
    url_params_str = "&".join([f"{key}={value}" for key, value in url_params.items()])
    return f"{base_url}daymet_v4_daily_na_{variable}_{year}.nc?{url_params_str}"


# Save download configuration file from website
def save_log(config_file, config_data):
    """Save config data to a JSON file."""
    with open(config_file, 'w') as file:
        json.dump(config_data, file)
    print(f"Log saved to {config_file}")


def read_config(log_file):
    """Read log data from a JSON file."""
    if os.path.exists(log_file):
        with open(config_file, 'r') as file:
            return json.load(file)
    else:
        return {}


def download_daymet(params, outpath, config_file):
    config_data = read_config(config_file)
    for variable in params['daymet_variables']:
        for year in params['years']:
            print(f"Requesting daily data for {variable} ; {year}")

            daymet_download_url = create_url(variable, year, params)
            output_name = f'DAYMET_{variable}_{year}.nc'

            if daymet_download_url in config_data:
                print(f"Data for {variable} {year} already downloaded.")
            else:
                urllib.request.urlretrieve(daymet_download_url, os.path.join(outpath, output_name))
                print(f"Saving to: {os.path.join(outpath, output_name)}")

            # save log by having the following description : {url : {filename : output_name}}
            # example of output name : DAYMET_dayl_2020.nc
            config_data[daymet_download_url] = {'filename': output_name, 'north':params['north'], 'south':params['south'],
                                             'east':params['east'], 'west':params['west']}
            save_log(config_file, config_data)


# download_daymet(define_parameters(), "E:\\Weather_Grid\\Daymet\\Daily\\", "config.json")

# Load from planetary computer
def load_daymet_from_planetary():
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


# Main execution ---------------------------------------
# todo : find a way to batch download multiple variables, different dimensions or different years
if __name__ == "__main__":
    dimensions={'north': 46, 'south': 44, 'east': -73, 'west': -75}
    config = define_parameters(variables = ['prcp','tmax','tmin'],start_year=2019,end_year=2021,dimensions=dimensions)
    output_path = "D:\\observations\\Daymet\\Test\\"
    config_file = "config.json"
    download_daymet(config, output_path, config_file)
