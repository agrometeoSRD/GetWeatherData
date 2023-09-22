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

# imports
import os
import xarray as xr
from xclim import units
from siphon.catalog import TDSCatalog
from clisops.core import subset
from source.utils import subsetting

shape_path = f"C:\\Users\\{os.getenv('USERNAME')}\\OneDrive - IRDA\\GIS\\RegionAgricolesQC.geojson"

# get url and metadata
def get_meta():
    url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/reanalyses/catalog.html?dataset=datasets/reanalyses/day_RDRSv2.1_NAM.ncml"
    cat = TDSCatalog(url)
    # Create Catalog
    cat = TDSCatalog(url)

    # List of datasets
    print(f"Number of datasets: {len(cat.datasets)}")

    # Access mechanisms - here we are interested in OPENDAP, a data streaming protocol
    cds = cat.datasets[0]
    print(f"Access URLs: {tuple(cds.access_urls.keys())}")

    return cds

#Open with xarray
def open_cds(cds, do_subset = True) -> xr.Dataset:
    ds = xr.open_dataset(cds.access_urls["OPENDAP"], chunks="auto")

    # convert units to celcius
    ds['tas'] = units.convert_units_to(ds['tas'], target='degC')
    ds['tasmax'] = units.convert_units_to(ds['tasmax'], target='degC')
    ds['tasmin'] = units.convert_units_to(ds['tasmin'], target='degC')
    if do_subset == True:
        print('-> Subsetting with RegionAgricoleQC.geojson...')
        ds = subsetting.subset_by_shape(ds, shape_path)
    return ds



def get_era_sample(ds):
    # from the dataset, only select a sample to make it easier for testing
    ds_sub = subset.subset_time(ds, start_date="2020-01-01", end_date="2021-01-01")
    # subset box by specifying the longitude and latitude boundaries
    lon_bnds = [-74.4, -72.9]
    lat_bnds = [45.3, 45.9]
    ds1 = subset.subset_bbox(ds_sub, lon_bnds=lon_bnds, lat_bnds=lat_bnds)
    return ds1

def main():
    cds = get_meta()
    era_ds = open_cds(cds)
    print(era_ds)

if __name__ == "__main__":
    main()
