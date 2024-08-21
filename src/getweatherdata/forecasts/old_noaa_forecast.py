"""
File: old_noaa_forecast.py
Author: sebastien.durocher
Python : 3.11
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description:
- Essentially just this : https://github.com/blaylockbk/Herbie
- fast herbie faster than anything i could ever comne up with
- Currently only been "tested" with HRRR and the surface
- Loading multiple variables at the same time can very likely lead to a crash
Created: 2024-07-08
"""
#todo : allow options to either run stand alone, run with cli or run from another function

# Import statements
from herbie import Herbie
from herbie import fast
import xarray as xr
import pandas as pd
from clisops.core import subset
import xclim as xc

import time

# Constants

# Functions
def create_time_range(start_date:str='2021-01-01 00:00',end_date:str='2021-01-10 00:00'):
  # To use if we want to create a date range with hourly frequency
  date_range = pd.date_range(start=start_date, end=end_date, freq='h')
  # Convert the date range to a list of strings in the desired format
  list_times = date_range.strftime('%Y-%m-%d %H:%M').tolist()
  return list_times

def standardize_noaa_units(ds):
    # apply changes to meteorological units of noaa models to make them work with units used in the library
    # if condition for if temperature is the variable, meaning that temperature unit is in kelvins
    # if 'TMP' in ds._attrs['search']:
        # ds =
    pass


def get_forecast_data(list_times, model='hrrr', product='sfc', fxx=[0], variable:list=['TMP:2 m']):
    """
    Fetches forecast data for a specified variable using the Herbie package and return into an xarray

    Parameters:
        list_times (list): List of forecast times.
        model (str): Model name (default is 'hrrr').
        product (str): Product name (default is 'sfc').
        fxx (list): Forecast hour (default is [0]). Should always be 0
        variable (str): Variable name to fetch (default is 'SNOWC').

    Returns:
        xarray.Dataset: Dataset containing the forecast data for the specified variable.
    """
    variable = '|'.join(variable)
    FH = fast.FastHerbie(list_times, model=model, product=product, fxx=fxx)
    try:
      ds = FH.xarray(variable)
      # rename variable to package standards (see column_names.json in config folder)


    except FileNotFoundError as e:
      raise ValueError(
                  f"An error occurred while fetching the variable '{variable}'. Please check the list of available variables at "
                  "https://mesowest.utah.edu/html/hrrr/zarr_documentation/html/zarr_variables.html\n"
                  f"Original error message: {e}"
              )
def turn_to_dask(xarray):
  return xarray.to_dask_dataframe()

def process_dask(dask_dfs:list):
  # To use if multiple dask dataframes that have been loaded from the herbie forecast
  # Concatenation currently only works with two dataframes
  if len(dask_dfs) > 1:
    raise ValueError(f"You're trying to process more than two dask dataframes at the same time and sadly the code wasn't built for that")
  combined_dask_df = dd.concat([dask_dfs[0],dask_dfs[1]])

  # drop unnecessary columns
  cols_to_drop = ['y','x','step','surface','valid_time','gribfile_projection'] # drop columns that arent really necessary
  combined_dask_df = combined_dask_df.drop(columns = cols_to_drop)

  # turn to optimal 100mb partitions
  print('Repartitioning to 100mb partitions...')
  combined_dask_df = combined_dask_df.repartition(partition_size="100MB")

  return combined_dask_df

def save_to_csv(xarray,longitudes:list,latitudes:list):
    xarray_gridpoints = subset.subset_gridpoint(xarray, lon=longitudes, lat=latitudes)
    xarray_gridpoints.to_dask_dataframe()

def save_to_netcdf():
  pass


# Main execution ---------------------------------------




# Testing grounds ---------------------------------------
list_times = create_time_range(start_date='2022-01-01 00:00', end_date='2022-01-02 00:00')

print('Fast herbie')
start = time.time()
FH = fast.FastHerbie(list_times,model='hrrr',product='sfc',fxx=[0])
snow_cover_ds = FH.xarray('SNOWC') # get snow cover values (%)
end = time.time()
print(end-start)

start = time.time()
snow_height_ds = FH.xarray("SNOD")
end = time.time()
print(end-start)

# convert to dask dataframe and concatenante together
import dask.dataframe as dd
cols_to_drop = ['y','x','step','surface','valid_time','gribfile_projection'] # drop columns that arent really necessary
snow_cover_daskdf = snow_cover_ds.to_dask_dataframe()
snow_height_daskdf = snow_height_ds.to_dask_dataframe()
snow_daskdf = dd.concat([snow_cover_daskdf,snow_height_daskdf])
# drop unnecessary columns
snow_daskdf = snow_daskdf.drop(columns = cols_to_drop)
# turn to optimal 100mb partitions
print('Repartitioning to 100mb partitions...')
snow_daskdf = snow_daskdf.repartition(partition_size="100MB")


