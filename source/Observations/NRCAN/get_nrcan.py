#!/usr/bin/env
"""
Creation date: 2023-08-02
Creator : the_l
Python version : 3.10

Description:
- Inspired from the PAVICS tutorial, we will extract NRCAN v2 data for a list of specific lat and lon indices
- File is formatted to work as an external packages for other scripts that need data

Notes:
"""
#TODO : Handle exceptions


import xarray as xr
from xclim import units
from siphon.catalog import TDSCatalog

def get_data():
    # get url and metadata
    url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/gridded_obs/catalog.html?dataset=datasets/gridded_obs/nrcan_v2.ncml"
    # url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/simulations/bias_adjusted/cmip5/ouranos/cb-oura-1.0/catalog.xml"  # TEST_USE_PROD_DATA

    # Create Catalog
    cat = TDSCatalog(url)

    # List of datasets
    print(f"Number of datasets: {len(cat.datasets)}")

    # Access mechanisms - here we are interested in OPENDAP, a data streaming protocol
    cds = cat.datasets[0]
    print(f"Access URLs: {tuple(cds.access_urls.keys())}")

    # Open with xarray

    ds = xr.open_dataset(cds.access_urls["OPENDAP"], chunks="auto")

    # convert units to celcius
    ds['tasmax'] = units.convert_units_to(ds['tasmax'], target='degC')
    ds['tasmin'] = units.convert_units_to(ds['tasmin'], target='degC')

    # Compute mean daily temperature
    ds['tasmean'] = (ds['tasmin'] + ds['tasmax']) / 2
    ds['tasmean'].attrs = ds['tasmax'].attrs

    return ds

