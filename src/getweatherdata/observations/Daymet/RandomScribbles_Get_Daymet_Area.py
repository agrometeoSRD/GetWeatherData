#! /usr/bin/env python
# Load daymet in an area.
# Two methods are used : planetary computer and local download. If planetary computer is not available, then local download is used.


#TODO : create a reproducible array function (having issues because daymet works in x,y with lat,lon being in 2d... abandoning the idea for now. Maybe just best to create a small slice of the dataset)
#TODO : add detailed description of what the code does.
#TODO : add dask multiple interpreter
#TODO : something to make shape_path more flexible to different OS and users

# imports
import os
import json
import urllib.request
import argparse
from collections import defaultdict


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

# def create_reproducible_array(
#     dimensions  : dict,
#     missing_pct : float = 0.1,
#     random      : bool  = False,
#     random_seed : int   = None,
#         ) -> xr.DataArray:
#
#     import numpy as np
#
#     numpy_rng    = np.random.default_rng(random_seed)
#     # numpy_rng = np.random.default_rng(config['data_kwargs'].get('random_seed', None))
#
#     keys, shapes = zip(*dimensions.items())
#     # dimensions = config['data_kwargs']['dimensions']
#     # coords = {key: np.linspace(start, end, size) for key, (start, end, size) in dimensions.items()}
#     # shapes = tuple(dimensions[key][2] for key in dimensions)
#
#     if random: a = numpy_rng.random(shapes, dtype='float32')
#     else:      a = np.arange(np.prod(shapes), dtype='float32').reshape(shapes)
#     missing_idxs = i = numpy_rng.integers(0, a.size, int(a.size * missing_pct))
#     a.ravel()[i] = np.nan
#
#     return xr.DataArray(a, coords=dict(zip(keys, map(np.arange, shapes))))


# def generate_dataset():
#     import pandas as pd
#     import numpy as np
#     y_vals = 5
#     x_vals = 10
#     time_vals = pd.date_range('1980-01-01', '1980-12-31', freq='D')
#
#     # coordinates
#     y = np.linspace(4.984e+06, -3.09e+06, y_vals)
#     x = np.linspace(-4.56e+06, 3.253e+06, x_vals)
#     lon,lat = np.meshgrid(x,y)
#
#     # lat = np.linspace(10, 60, y_vals)
#     # lon = np.linspace(-130, 60, x_vals)
#     coords = {'time':time_vals,'lat':lat,'lon':lon}
#
#     # create artificial data based on dimension size (ideally would be based on values that make somewhat sense)
#     prcp = np.random.rand(len(time_vals), y_vals, x_vals)
#     swe = np.random.rand(len(time_vals), y_vals, x_vals)
#     tmax = np.random.rand(len(time_vals), y_vals, x_vals) * 40 - 20  # Temp between -20 to 20
#     tmin = np.random.rand(len(time_vals), y_vals, x_vals) * 40 - 20  # Temp between -20 to 20
#     vp = np.random.rand(len(time_vals), y_vals, x_vals)
#
#     # add condition for missing percentage if want to deal with missing values
#     random_seed = 42
#     missing_pct = config['data_kwargs'].get('missing_pct', 0.1)
#     numpy_rng    = np.random.default_rng(random_seed)
#     missing_idxs = i = numpy_rng.integers(0, prcp.size, int(prcp.size * missing_pct))
#     prcp.ravel()[i] = np.nan
#
#     # Create the dataset
#     ds_sample = xr.Dataset(
#             {
#                 'prcp': (['time', 'y', 'x'], prcp),
#                 'swe': (['time', 'y', 'x'], swe),
#                 'tmax': (['time', 'y', 'x'], tmax),
#                 'tmin': (['time', 'y', 'x'], tmin),
#                 'vp': (['time', 'y', 'x'], vp),
#                 'lambert_conformal_conic': 16,
#                 'time_bnds': time_vals,
#             },
#             coords={
#                 'lat': (['y', 'x'], lat),
#                 'lon': (['y', 'x'], lon),
#                 'time': time_vals,
#                 # 'y': y,
#                 # 'x': x
#             },
#             attrs={
#                 'Conventions': 'CF-1.6',
#                 'Version_data': 'Daymet Data Version 4.0',
#                 'Version_software': 'Daymet Software Version 4.0',
#                 'citation': 'Please see http://daymet.ornl.gov/ for current Daymet data citations.',
#                 'references': 'Please see http://daymet.ornl.gov/ for current information on Daymet references.',
#                 'source': 'Daymet Software Version 4.0',
#                 'start_year': '1980'
#             }
#         )

# # Use this example from the website
# # Small configuration to verify correctness
#  small_config = {
#      'data_kwargs'   : {
#          'dimensions' : { # Dimensions of the DataArray
#              'time'      : ("1980-01-01", "1980-12-31", 365),
#              'latitude'  : (10, 60, 5),
#              'longitude' : (-130, -60, 10),
#          },
#          'random_seed' : 42,
#      },
#      'chunks' : { # Initial data chunking
#          'time'      : 2,
#          'latitude'  : 2,
#          'longitude' : 2,
#      },
#      'depth' : { # 2 x 3 x 3 window
#          'time'      : 1, # 1 lookback step = window depth  2
#          'latitude'  : 1, # 1 adjacent lats = window height 3
#          'longitude' : 1, # 1 adjacent lons = window width  3
#      },
#  }
# config = small_config
# data   = create_reproducible_array(**config['data_kwargs'])

# Try modified example with pretend daymet
# daymetconfig = {
#     'data_kwargs': {
#         'dimensions': {
#             'time': ("1980-01-01", "1980-12-31", 365),
#             'latitude': (10, 60, 5),
#             'longitude': (-130, -60, 10),
#         },
#         'random_seed': 42,
#         'random': True,
#     },
#     'chunks': {
#         'time': 2,
#         'latitude': 2,
#         'longitude': 2,
#     },
#     'depth': {
#         'time': 1,
#         'latitude': 1,
#         'longitude': 1,
#     },
# }

# def get_url(frequency: str) -> str:
#     """
#     Get the appropriate URL based on the desired frequency.
#
#     Parameters:
#     - frequency: Either 'daily', 'monthly', or 'annual'
#
#     Returns:
#     The URL as a string.
#     """
#     base_url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/daymet-{}-na"
#     if frequency not in ['daily', 'monthly', 'annual']:
#         raise ValueError("Frequency should be either 'daily', 'monthly', or 'annual'")
#     return base_url.format(frequency)

# TODO : Transform function into class and make it so that we can get either : daily data, yearly data, climate summaries
def create_url(variable, year, north, south, east, west, s_stride, t_stride, format) -> str:
    """Create the url for the DAYMET data."""
    base_url = "https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/2129/"
    version_str = f"daymet_v4_daily_na_{variable}_{year}.nc?var=lat&var=lon&"
    var_area_str = (f"var={variable}&north={north}&west={west}&east={east}&south={south}"
                    f"&disableProjSubset=on&horizStride={s_stride}&")
    time_str = f"time_start={year}-01-01T12%3A00%3A00Z&time_end={year}-12-31T12%3A00%3A00Z&timeStride={t_stride}&"
    return f"{base_url}{version_str}{var_area_str}{time_str}accept={format}"

# Create a test function to see if the url works
def test_url():
    url = create_url('dayl', 1980, 56, 44.991, -63.551, -65, 1, 1, 'netcdf')
    print(url)


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


def load_daymet(years:list,variables:str):
    # inputs : years and variables
    # output : xarray dataset
    planetary_available_years = list(range(1980,2020))
    # if any of the years are in the planetary computer, then use planetary computer
    # Note that planetary computer loads up all the years, so time selecting will be done after loading
    if any(year in planetary_available_years for year in years):
        ds_plany = load_daymet_planetary()


def main(frequency: str):
    ds = load_dataset(frequency)
    print(ds)

if __name__ == "__main__":
    freq = 'daily'
    main(freq)

#################################################################
# #%% Functions

def save_config(args, out_path):
    """Save the configuration to a file."""
    config_file = os.path.join(out_path, "config.txt")

    with open(config_file, 'w') as f:
        json.dump(vars(args), f)

def check_config(args, out_path):
    """Check if configuration file exists and matches current args."""
    config_file = os.path.join(out_path, "config.txt")

    # If the config file doesn't exist, return False
    if not os.path.isfile(config_file):
        return False

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Compare the previous configuration to the current args
    # If they are not the same, return False
    return config == vars(args)

def save_file(url, out_file):
    """Save the DAYMET data to a file."""
    print(f"Saving to :", out_file)
    urllib.request.urlretrieve(url, out_file)

def main(args):
    """Main function to download DAYMET data."""
    # variables = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    variables = ['dayl']
    save_config_dict = defaultdict()
    for variable in variables:
        for year in range(args.start_year, args.end_year + 1):
            print(f"Requesting daily data for {variable} ; {year}")
            url = create_url(variable, year, args.north, args.south, args.east, args.west,
                             args.s_stride, args.t_stride, args.format)

            out_name = f'DAYMET_{variable}_{year}.nc'
            out_file = os.path.join(args.out_path, out_name)

            # save out_file with corresponding url for the configuration
            save_config_dict[out_file] = url

            save_file(url, out_file)

    save_config()

#%% Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DAYMET data. Default bounding box is that of PAVICS")
    parser.add_argument('--start-year', type=int, default=1980, help='Start year')
    parser.add_argument('--end-year', type=int, default=1981, help='End year')
    parser.add_argument('--north', type=float, default=56, help='North boundary')
    parser.add_argument('--south', type=float, default=44.991, help='South boundary')
    parser.add_argument('--east', type=float, default=-63.551, help='East boundary')
    parser.add_argument('--west', type=float, default=-65, help='West boundary')
    parser.add_argument('--s-stride', type=int, default=1, help='Spatial stride')
    parser.add_argument('--t-stride', type=int, default=1, help='Time stride')
    parser.add_argument('--format', type=str, default='netcdf', help='Format of the data')
    parser.add_argument('--out-path', type=str, default='./', help='Output directory')
    args = parser.parse_args()

    main(args)

def create_url(variable, year, north, south, east, west, s_stride, t_stride, format):
    """Create the url for the DAYMET data."""
    base_url = "https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/2129/"
    version_str = f"daymet_v4_daily_na_{variable}_{year}.nc?var=lat&var=lon&"
    var_area_str = (f"var={variable}&north={north}&west={west}&east={east}&south={south}"
                    f"&disableProjSubset=on&horizStride={s_stride}&")
    time_str = f"time_start={year}-01-01T12%3A00%3A00Z&time_end={year}-12-31T12%3A00%3A00Z&timeStride={t_stride}&"
    return f"{base_url}{version_str}{var_area_str}{time_str}accept={format}"

def save_config(args, out_path):
    """Save the configuration to a file."""
    config_file = os.path.join(out_path, "config.txt")

    with open(config_file, 'w') as f:
        json.dump(vars(args), f)

def check_config(args, out_path):
    """Check if configuration file exists and matches current args."""
    config_file = os.path.join(out_path, "config.txt")

    # If the config file doesn't exist, return False
    if not os.path.isfile(config_file):
        return False

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Compare the previous configuration to the current args
    # If they are not the same, return False
    return config == vars(args)

def save_file(url, out_file):
    """Save the DAYMET data to a file."""
    print(f"Saving to :", out_file)
    urllib.request.urlretrieve(url, out_file)

def main(args):
    """Main function to download DAYMET data."""
    # variables = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
    variables = ['dayl']
    save_config_dict = defaultdict()
    for variable in variables:
        for year in range(args.start_year, args.end_year + 1):
            print(f"Requesting daily data for {variable} ; {year}")
            url = create_url(variable, year, args.north, args.south, args.east, args.west,
                             args.s_stride, args.t_stride, args.format)

            out_name = f'DAYMET_{variable}_{year}.nc'
            out_file = os.path.join(args.out_path, out_name)

            # save out_file with corresponding url for the configuration
            save_config_dict[out_file] = url

            save_file(url, out_file)

    save_config()

#%% Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DAYMET data. Default bounding box is that of PAVICS")
    parser.add_argument('--start-year', type=int, default=1980, help='Start year')
    parser.add_argument('--end-year', type=int, default=1981, help='End year')
    parser.add_argument('--north', type=float, default=56, help='North boundary')
    parser.add_argument('--south', type=float, default=44.991, help='South boundary')
    parser.add_argument('--east', type=float, default=-63.551, help='East boundary')
    parser.add_argument('--west', type=float, default=-65, help='West boundary')
    parser.add_argument('--s-stride', type=int, default=1, help='Spatial stride')
    parser.add_argument('--t-stride', type=int, default=1, help='Time stride')
    parser.add_argument('--format', type=str, default='netcdf', help='Format of the data')
    parser.add_argument('--out-path', type=str, default='./', help='Output directory')
    args = parser.parse_args()

    main(args)

# # Output example on the terminal :
# # python script.py --start-year 1980 --end-year 1985 --north 55 --south 44.991 --east -63.551 --west -79.579 --s-stride 1 --t-stride 1 --format netcdf --out-path ./output
#
# # Small output example on the terminal :
# # python script.py --start-year 1980 --end-year 1980 --north 55 --south 54 --east -63.551 --west -64 --s-stride 1 --t-stride 1 --format netcdf --out-path D:\observations\Daymet
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
