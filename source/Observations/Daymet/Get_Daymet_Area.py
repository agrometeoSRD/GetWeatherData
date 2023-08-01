#! /usr/bin/env python
# Load daymet in an area. Output is netcdf

# imports
import os
import json
import urllib.request
import argparse
from collections import defaultdict

#TODO : add detailed description of what the code does. Explain why config file is necessary
#TODO : Introducing error cases in case something doesn't work. Probably config file in wrong place could be an issue

#%% Functions
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

# Output example on the terminal :
# python script.py --start-year 1980 --end-year 1985 --north 55 --south 44.991 --east -63.551 --west -79.579 --s-stride 1 --t-stride 1 --format netcdf --out-path ./output

# Small output example on the terminal :
# python script.py --start-year 1980 --end-year 1980 --north 55 --south 54 --east -63.551 --west -64 --s-stride 1 --t-stride 1 --format netcdf --out-path D:\Observations\Daymet

#%% script alternative : get data (without downloading it) with siphon.catalogue
import xarray as xr
import dask

#1 ) Use Siphon to access a THREDDS catalog
url = 'https://thredds.daac.ornl.gov/thredds/catalog/ornldaac/2129/catalog.xml'
from siphon.catalog import TDSCatalog
cat = TDSCatalog(url)
dataset_list = list(cat.datasets.keys())
# print(list(cat.datasets.keys()))
# https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/2129/daymet_v4_daily_na_dayl_1980.nc?var=lat&var=lon&var=dayl&north=56&west=-65&east=-63.551&south=44.991&disableProjSubset=on&horizStride=1&time_start=1980-01-01T12%3A00%3A00Z&time_end=1980-12-31T12%3A00%3A00Z&timeStride=1&accept=netcdf

# filter list based on the following criteria
location = 'Continental North America'
years = list(range(2000, 2002))  # convert the years to strings for comparison
variables = ["daily maximum temperature", "vapor pressure"] # list of possible variables

# loop through dataset_list (list of strings), find the index of the datasets that match the criteria
print('Finding datasets that match the criteria')
idx_list = []
for idx,dataset in enumerate(dataset_list):
    if location in dataset and any(variable in dataset for variable in variables) and int(dataset[-5:-1]) in years:
        idx_list.append(idx)

# loop through cat.datasets that are within idx_list. open each dataset has xarray dataset and save into one single list
# xr_ds_list = []
# for idx in idx_list:
#     xr_ds_list.append(xr.open_dataset(cat.datasets[idx].access_urls["OPENDAP"], chunks="auto"))

print('Opening datasets')
dsets = [cat.datasets[idx].remote_access(use_xarray=True).reset_coords(drop=True).chunk({'time':1}) for idx in idx_list]

ds = xr.combine_by_coords(dsets,combine_attrs='drop')
ds = ds.sel(nv=0,drop=True)

# Testing dataset with PAVICS single pixel extraction (see if we actually get data)
def test_singlepoint_pixel(ds):
    from clisops.core import subset
    import pyproj
    proj_daymet = "+proj=lcc +lat_0=42.5 +lon_0=-100 +lat_1=25 +lat_2=60 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs" #custom daymet crs
    latlon_daymet_transform = pyproj.Transformer.from_crs("EPSG:4326", proj_daymet, always_xy=True)


    lon = [-75.4, -85, -65.5]  # Longitude
    lat = [46.67, 41, 55.3]  # Latitude
    ds_gridpoint = subset.subset_gridpoint(ds, lon=lon, lat=lat)
    print(ds_gridpoint)

    # Plot first year of tasmax data
    a = ds.tmax.isel(x=slice(0,3),y=slice(0,1),time=slice(0, 365))

#%% Alternative with Microsoft PlanetaryComputer
import pystac
import fsspec
from clisops.core import subset
import xarray as xr

# Tutorial : https://planetarycomputer.microsoft.com/dataset/daymet-daily-na#Example-Notebook
# Adress : https://planetarycomputer.microsoft.com/dataset/group/daymet#north_america
url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/daymet-daily-na"
collection = pystac.read_file(url)
asset = collection.assets["zarr-https"]
store = fsspec.get_mapper(asset.href)
ds = xr.open_zarr(store, **asset.extra_fields["xarray:open_kwargs"])
ds = ds.sel(nv=0,drop=True)

# Downscale to fit PAVICS grid
# Subset over the polygon (LONG LOADING TIME AND ERROR IS PRODUCED
tmp = ds.isel(time = 5000)
ds_poly = subset.subset_shape(
    ds, shape="E:\\GIS\\PAVICS\\RegionAgricolesQC.geojson"
)  # use path to layer

# OLD SCRIPT : Read OneNote//Météo//Daymet for description on what's going on with everything
#
#
# STARTYEAR = 1980
# ENDYEAR   = 1985
# # NO_NAME = "NULL"
# # YEAR_LINE = "years:"
# # VAR_LINE  = "variables:"
# DAYMET_VARIABLES = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp'] # Not sure if tmean could work
# # DAYMET_VARIABLES = ['tmin', 'prcp']
# # DAYMET_VARIABLES = ['tmean'] # N
# DAYMET_YEARS     = [str(year) for year in range(STARTYEAR, ENDYEAR + 1)]
# north = 55
# south = 44.991
# east = -63.551
# west = -79.579
# s_stride = 1 #default is 1 to get all data
# t_stride = 1 #default is 1 to get all data
# format = "netcdf"
#
# #%% Daily data
# OutPath = "E:\\Weather_Grid\\Daymet\\Daily\\"
# Project_Name = '' # if adding a project name, must add an _ at the beggining
# for DAYMETVAR in DAYMET_VARIABLES:
#     for YEAR in DAYMET_YEARS:
#         print(f"Requesting daily data for {DAYMETVAR} ; {YEAR}")
#         time_start = f"{YEAR}-01-01T12%3A00%3A00Z" # Not sure what that last part is
#         time_end = f"{YEAR}-12-31T12%3A00%3A00Z"
#
#         # Update paths
#         DAYMET_BASE_URL = f"https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/2129/"
#         DAYMET_VERSION_STR = f"daymet_v4_daily_na_{DAYMETVAR}_{YEAR}.nc?var=lat&var=lon&"
#         DAYMET_VAR_AREA_STR = f"var={DAYMETVAR}&north={north}&west={west}&east={east}&south={south}" \
#                               f"&disableProjSubset=on&horizStride={s_stride}&"
#         DAYMET_TIME_STR = f"time_start={time_start}&time_end={time_end}&timeStride={t_stride}&"
#         DAYMET_FULL_PTH = f"{DAYMET_BASE_URL}{DAYMET_VERSION_STR}{DAYMET_VAR_AREA_STR}{DAYMET_TIME_STR}accept={format}"
#         OutName = f'DAYMET{Project_Name}_{DAYMETVAR}_{YEAR}.nc'
#         print(f"Saving to :", OutPath + OutName)
#         urllib.request.urlretrieve(DAYMET_FULL_PTH, OutPath + OutName)
# stophere
#
# #%% #monthly climate summaries
# # From https://daac.ornl.gov/DAYMET/guides/Daymet_V3_Monthly_Climatology.html
# # These single month summary data products are produced for each individual month within a calendar year
# # and cover the same period of record as the Daymet V3 daily data
# # OutPath = f"F:\\Weather_Grid\\Daymet\\Means\\"
#
# OutPath = "D:\\Weather_Grid\\Daymet\\Monthly\\"
# DAYMET_BASE_URL = f"https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/1855/"
# for DAYMETVAR in DAYMET_VARIABLES:
#     for YEAR in DAYMET_YEARS:
#         if DAYMETVAR == 'prcp':
#             DAYMET_VAR_STR = f"daymet_v4_prcp_monttl_na_{YEAR}.nc?var={DAYMETVAR}&"
#         else:
#             DAYMET_VAR_STR = f"daymet_v4_{DAYMETVAR}_monavg_na_{YEAR}.nc?var={DAYMETVAR}&"
#         print(f"Requesting data for {DAYMETVAR} ; {YEAR}")
#         time_start = f"{YEAR}-04-01T12%3A00%3A00Z" # Not sure what that last part is
#         time_end = f"{YEAR}-10-30T12%3A00%3A00Z"
#
#         DAYMET_AREA_STR = f"north={north}&west={west}&east={east}&south={south}&disableLLSubset=on&disableProjSubset=on&horizStride={s_stride}&"
#         DAYMET_TIME_STR = f"time_start={time_start}&time_end={time_end}&timeStride={t_stride}&"
#         DAYMET_FULL_PTH =  f"{DAYMET_BASE_URL}{DAYMET_VAR_STR}{DAYMET_AREA_STR}{DAYMET_TIME_STR}accept={format}"
#         print(f"Saving to :", OutPath + f'DAYMET_{DAYMETVAR}_Monthly{YEAR}.nc')
#         urllib.request.urlretrieve(DAYMET_FULL_PTH, OutPath + f'DAYMET_{DAYMETVAR}_Monthly{YEAR}.nc')
# #https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/1345/daymet_v3_prcp_monttl_1982_pr.nc4?
# # north=19.9381&west=-67.9927&east=-64.1195&south=16.8443&disableLLSubset=on&disableProjSubset=on&horizStride=1&
# # time_start=1982-01-16T12%3A00%3A00Z&time_end=1982-12-16T12%3A00%3A00Z&timeStride=1&accept=netcdf
#
# #%% Concatenate multiple files into one
# # Doesn't work for large files (tested on > 700mb)
# import xarray as xr
# import glob
#
# def get_digit(el):
#     return ''.join(filter(str.isdigit,el)) # returns a string
#
# def number_sanity_check(number):
#     # Code is only suppose to accept 4 numbers to get a year. If numbers are in the path or somewherelse, an error will be raised
#     if len(number) != 4 :
#         raise ValueError(f'There are more numbers than there are year numbers ({number})')
#     else:
#         return True
#
# InPath = "D:\\Weather_Grid\\Daymet\\Monthly\\"
# OutPath = "F:\\Projet_CRIBIQ\\Daymet\\"
# Suffix = 'CRIBIQ'
# # This is what will distinguish between every other files (see DAYMET_VARIABLES)
# # We will only select the first 30 years
# sel_years = list(range(2010,2021))
# for param_name in DAYMET_VARIABLES:
#     print(f"Loading all datasets for {param_name}")
#     all_files_with_Specificparam = [f for f in glob.glob(f'{InPath}DAYMET_{param_name}*.nc', recursive=True)]
#     # select only files that have requested years (yeras must be the only digits)
#     all_files_with_Specificparam = [el for el in all_files_with_Specificparam if get_digit(el).isdigit() and number_sanity_check(get_digit(el))]
#     all_files_with_Specificparam = [el for el in all_files_with_Specificparam if int(get_digit(el)) in sel_years]
#     # Insanity check to make sure we only have years as the only digits in the path
#     out_fname = f"{OutPath}DAYMET_{param_name}_{Suffix}.nc"
#     mdf = xr.open_mfdataset(all_files_with_Specificparam)
#     mdf_selYears = mdf.sel(time=mdf.time.dt.year.isin(sel_years))
#     print(f'Saving as {out_fname}')
#     mdf_selYears.to_netcdf(out_fname)
#
# # Special case for temperature : combine tmax and tmin together and create a output file that has tmax,tmin and tmean
# # tmean is the daily mean between the two (not the best I know, but whatever)
# ds_tmax = xr.open_dataset(f'{OutPath}DAYMET_tmax_{Suffix}.nc')
# ds_tmin = xr.open_dataset(f'{OutPath}DAYMET_tmin_{Suffix}.nc')
# ds_temp = xr.merge([ds_tmax,ds_tmin])
# ds_temp = ds_temp.assign(tmean = (ds_temp.tmax + ds_temp.tmin)/2)
# ds_temp.to_netcdf(f'{OutPath}DAYMET_temp_{Suffix}.nc')
#
# # ext = "daymet_v3_tmax_monavg_1990_hi.nc4?"
# # ext = "daymet_v3_tmin_monavg_2004_na.nc4?"
# # ext = "daymet_v3_prcp_monttl_2011_na.nc4?"