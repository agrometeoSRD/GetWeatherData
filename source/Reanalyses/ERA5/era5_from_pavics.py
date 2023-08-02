#!/usr/bin/env
"""
Creation date: 2023-07-27
Creator : sebastien.durocher
Python version : 3.10

Description:
- Taken from PAVICS tutorial, extract era5 data (https://pavics.ouranos.ca/climate_analysis.html#a)

Updates:

Notes:

"""

# %% imports
from xclim import atmos
from dask import compute
from clisops.core import subset
import pandas as pd
import xarray as xr
from xclim import units
from siphon.catalog import TDSCatalog

# %% get url and metadata
def get_meta():
    url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/reanalyses/catalog.html?dataset=datasets/reanalyses/day_ERA5-Land_NAM.ncml"
    cat = TDSCatalog(url)
    # Create Catalog
    cat = TDSCatalog(url)

    # List of datasets
    print(f"Number of datasets: {len(cat.datasets)}")

    # Access mechanisms - here we are interested in OPENDAP, a data streaming protocol
    cds = cat.datasets[0]
    print(f"Access URLs: {tuple(cds.access_urls.keys())}")

    return cds

# %% Open with xarray

def open_cds(cds):
    ds = xr.open_dataset(cds.access_urls["OPENDAP"], chunks="auto")

    # convert units to celcius
    ds['tas'] = units.convert_units_to(ds['tas'], target='degC')
    ds['tasmax'] = units.convert_units_to(ds['tasmax'], target='degC')
    ds['tasmin'] = units.convert_units_to(ds['tasmin'], target='degC')
