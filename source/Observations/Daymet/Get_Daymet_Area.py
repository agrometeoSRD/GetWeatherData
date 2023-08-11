#! /usr/bin/env python
# Load daymet in an area. Output is netcdf

# imports
import os
import json
import urllib.request
import argparse
from collections import defaultdict

#TODO : add detailed description of what the code does.
#TODO : add dask multiple interpreter
#TODO : something to make shape_path more flexible to different OS and users

# # Try to accelerate the process with dask (doesn't work)
# from dask.distributed import Client
# from dask.distributed import LocalCluster
# cluster = LocalCluster()
# client = Client(cluster)

#%% Alternative with Microsoft PlanetaryComputer
import os
import xarray as xr
import pystac
import fsspec
from clisops.core import subset

shape_path = f"C:\\Users\\{os.getenv('USERNAME')}\\OneDrive - IRDA\\GIS\\RegionAgricolesQC.geojson"

def get_url(frequency: str) -> str:
    """
    Get the appropriate URL based on the desired frequency.

    Parameters:
    - frequency: Either 'daily', 'monthly', or 'annual'

    Returns:
    The URL as a string.
    """
    base_url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/daymet-{}-na"
    if frequency not in ['daily', 'monthly', 'annual']:
        raise ValueError("Frequency should be either 'daily', 'monthly', or 'annual'")
    return base_url.format(frequency)

def subset_by_shape(ds: xr.Dataset, shape_path: str) -> xr.Dataset:
    """
    Subset the given dataset by a shapefile.

    Parameters:
    - ds: Input xarray Dataset
    - shape_path: Path to the shapefile (.geojson)

    Returns:
    An xarray Dataset subsetted by the shape
    """
    return subset.subset_shape(ds, shape=shape_path)

def load_dataset(frequency: str, subsetting = True) -> xr.Dataset:
    """
    Load the dataset from the given frequency.

    Parameters:
    - frequency: Either 'daily', 'monthly', or 'annual'
    - Subsetting : Boolean. Default to true cause always want to subset with ag geojson since smaller.

    Returns:
    An xarray Dataset
    """
    url = get_url(frequency)
    collection = pystac.read_file(url)
    asset = collection.assets["zarr-https"]
    store = fsspec.get_mapper(asset.href)
    ds = xr.open_zarr(store, **asset.extra_fields["xarray:open_kwargs"])
    if subsetting == True:
        ds = subset_by_shape(ds, shape_path)
    return ds.sel(nv=0, drop=True)

def main(frequency: str):
    ds = load_dataset(frequency)
    print(ds)

if __name__ == "__main__":
    freq = input("Enter the frequency (daily, monthly, annual): ")
    main(freq)

# #%% Functions
# def create_url(variable, year, north, south, east, west, s_stride, t_stride, format):
#     """Create the url for the DAYMET data."""
#     base_url = "https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/2129/"
#     version_str = f"daymet_v4_daily_na_{variable}_{year}.nc?var=lat&var=lon&"
#     var_area_str = (f"var={variable}&north={north}&west={west}&east={east}&south={south}"
#                     f"&disableProjSubset=on&horizStride={s_stride}&")
#     time_str = f"time_start={year}-01-01T12%3A00%3A00Z&time_end={year}-12-31T12%3A00%3A00Z&timeStride={t_stride}&"
#     return f"{base_url}{version_str}{var_area_str}{time_str}accept={format}"
#
# def save_config(args, out_path):
#     """Save the configuration to a file."""
#     config_file = os.path.join(out_path, "config.txt")
#
#     with open(config_file, 'w') as f:
#         json.dump(vars(args), f)
#
# def check_config(args, out_path):
#     """Check if configuration file exists and matches current args."""
#     config_file = os.path.join(out_path, "config.txt")
#
#     # If the config file doesn't exist, return False
#     if not os.path.isfile(config_file):
#         return False
#
#     with open(config_file, 'r') as f:
#         config = json.load(f)
#
#     # Compare the previous configuration to the current args
#     # If they are not the same, return False
#     return config == vars(args)
#
# def save_file(url, out_file):
#     """Save the DAYMET data to a file."""
#     print(f"Saving to :", out_file)
#     urllib.request.urlretrieve(url, out_file)
#
# def main(args):
#     """Main function to download DAYMET data."""
#     # variables = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
#     variables = ['dayl']
#     save_config_dict = defaultdict()
#     for variable in variables:
#         for year in range(args.start_year, args.end_year + 1):
#             print(f"Requesting daily data for {variable} ; {year}")
#             url = create_url(variable, year, args.north, args.south, args.east, args.west,
#                              args.s_stride, args.t_stride, args.format)
#
#             out_name = f'DAYMET_{variable}_{year}.nc'
#             out_file = os.path.join(args.out_path, out_name)
#
#             # save out_file with corresponding url for the configuration
#             save_config_dict[out_file] = url
#
#             save_file(url, out_file)
#
#     save_config()
#
# #%% Main
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Download DAYMET data. Default bounding box is that of PAVICS")
#     parser.add_argument('--start-year', type=int, default=1980, help='Start year')
#     parser.add_argument('--end-year', type=int, default=1981, help='End year')
#     parser.add_argument('--north', type=float, default=56, help='North boundary')
#     parser.add_argument('--south', type=float, default=44.991, help='South boundary')
#     parser.add_argument('--east', type=float, default=-63.551, help='East boundary')
#     parser.add_argument('--west', type=float, default=-65, help='West boundary')
#     parser.add_argument('--s-stride', type=int, default=1, help='Spatial stride')
#     parser.add_argument('--t-stride', type=int, default=1, help='Time stride')
#     parser.add_argument('--format', type=str, default='netcdf', help='Format of the data')
#     parser.add_argument('--out-path', type=str, default='./', help='Output directory')
#     args = parser.parse_args()
#
#     main(args)
#
# # Output example on the terminal :
# # python script.py --start-year 1980 --end-year 1985 --north 55 --south 44.991 --east -63.551 --west -79.579 --s-stride 1 --t-stride 1 --format netcdf --out-path ./output
#
# # Small output example on the terminal :
# # python script.py --start-year 1980 --end-year 1980 --north 55 --south 54 --east -63.551 --west -64 --s-stride 1 --t-stride 1 --format netcdf --out-path D:\Observations\Daymet
#
# def map_test_plotting():
#     import matplotlib.pyplot as plt
#     computed_val = ds_poly.tmax.isel(time=100).compute()
#     fig,axe = plt.subplots()
#     computed_val.plot(ax=axe)
#     plt.show()
#
# # Test plotting time series to see the result
# def test_plotting():
#     lon = [-75.4, -85, -65.5]  # Longitude
#     lat = [46.67, 41, 55.3]  # Latitude
#     ds_gridpoint = subset.subset_gridpoint(ds_poly, lon=lon, lat=lat)
#     ds_gridpoint_seltime = subset.subset_time(ds_gridpoint,start_date="2010")
#     print(ds_gridpoint_seltime)
#     # time series to visualize
#     ds_gridpoint_seltime.tmax.isel(time=slice(0, 365)).plot.line(x="time", figsize=(10, 4))
