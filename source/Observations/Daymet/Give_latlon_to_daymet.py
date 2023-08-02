#!/usr/bin/env
"""
Creation date: 2021-07-21
Creator : sebastien durocher 
Python version : ''

Description:
- After many hours of failing to get an answer, I decided to create a blank netcdf that only has lat and lon coordinates
I have no idea how this file managed to get lat and lon coordinates for Daymet, but since I don't know how to
repeat the exercise, I will simply use that to give lat and lon coordinates to any new file

This only works for AgriFusion. I'll have to find the answer eventually...

Updates:

Notes:
- The files are in netcdf
"""

import xarray as xr
import pathlib
import numpy as np
# Create the blank file (if it didn't already exist)

blank_filename = "F:\\Projet_CRIBIQ\\Daymet\\Blank_latlon.nc"
if pathlib.Path(blank_filename).exists():
    print('Exists')
latlon_file = xr.open_dataset("F:\\Projet_CRIBIQ\\Daymet\\DAYMET_precipitation_AF_fields.nc")
latlon_file = latlon_file.assign_coords({'lat':latlon_file.lat,'lon':latlon_file.lon})
lat = np.array(latlon_file.lat[0,:,:])
lon = np.array(latlon_file.lon[0,:,:])

# This is what needs to be done
# ds_temperature.assign_coords(lat=(('y','x'),lat))
# ds_temperature.assign_coords(lon=(('y','x'),lon))
