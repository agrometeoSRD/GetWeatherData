#!/usr/bin/env
"""
Creation date: 2020-12-16
Creator : sebastien durocher 
Python version : ''

Description:
    - Load files of the various climate sources and perform daily, weekly, monthly, seasonal or normal means.

    - Can't really do annual because we don't have all the whole year
        - We can say a production cycle to equate the mean/sum of all the months
        - And by seasonal we can split dates by temperature interval, altough that wouldn't really be seasonal anymore

    - Create output for each of these files

Updates:

Notes:
    - 2022-04-7 : a lot of things here a cool and a lot of things here need to be updated, but I didn't have enough time
"""
import csv
# imports
import glob
import os
import pathlib
import sys
from collections import defaultdict
from shutil import move

import numpy as np
import pandas as pd
import xarray as xr
from osgeo import gdal, osr

def define_proj(in_file, reference='EPSG:4326'):
    # Create temporary file, cause gdal warp doesn't like warping on same file
    tmp_file = '\\'.join(in_file.split('\\')[:-1]) + '\\tmp.nc'
    print(f"Warping to {reference}")
    if 'daymet' in in_file.lower():
        kwargs = {
            'srcSRS': '+proj=lcc +datum=WGS84 +a=6378137 +lat_1=25 +lat_2=60 +lat_0=42.5 +lon_0=-100 +x_0=0 +y_0=0 +units=km +no_defs'}
        gdal.Warp(tmp_file, in_file, dstSRS=reference, dstNodata=-9999, **kwargs)
    else:
        gdal.Warp(tmp_file, in_file, dstSRS=reference, dstNodata=-9999)
    move(tmp_file, in_file)

def array_to_geotiff(array, output_file, Type, var):
    # Write array into the corresponding extension
    # Taken from https://gis.stackexchange.com/questions/307926/numpy-array-to-raster-file-geotiff
    if Type == 'Daily':
        Original_file = 'F:\\Weather_Grid\\NRCAN\\clip_DailyMap.tif'
    else:
        Original_file = 'F:\\Weather_Grid\\NRCAN\\clip_MonthlyMap.tif'

    GDAL_DATA_TYPE = gdal.GDT_Float32
    GEOTIFF_DRIVER_NAME = r'GTiff'
    ds_first = gdal.Open(Original_file)  # Use the boundary array to get coordinate information
    geotransform = ds_first.GetGeoTransform()
    wkt = ds_first.GetProjection()

    # Create gtif file
    driver = gdal.GetDriverByName(GEOTIFF_DRIVER_NAME)
    # output_file = f"{Path}{int(year)}\\{var}_Mean" + '.tif'

    dst_ds = driver.Create(output_file,
                           array.shape[1],  # band.XSize,
                           array.shape[0],  # band.YSize,
                           1,
                           GDAL_DATA_TYPE)

    # Convert nan to -999
    array = np.nan_to_num(array, nan=-9999)
    # writting output raster
    dst_ds.GetRasterBand(1).WriteArray(array)
    # setting nodata value
    dst_ds.GetRasterBand(1).SetNoDataValue(-999)
    # setting extension of output raster
    # top left x, w-e pixel resolution, rotation, top left y, rotation, n-s pixel resolution
    dst_ds.SetGeoTransform(geotransform)
    # setting spatial reference of output raster
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dst_ds.SetProjection(srs.ExportToWkt())
    # Close output raster dataset
    ds = None
    dst_ds = None
    # re-open new dataset and warp to give it a coordinate system (only for daily pcp)
    if Type == 'Daily' and var == 'pcp':
        define_proj(output_file)
    print(f'Array was successfully saved as geotiff {output_file}')

# Monthly and yearly mean
def month_stuff(year, el):
    doy = el.split('\\')[-1].split('_')[-1][:-4]
    return pd.to_datetime(str(year) + '-' + doy, format='%Y-%j').month

def do_mean(in_array, var):
    if var == 'pcp':  # special case where pcp is monthly sum
        mean_array = np.sum(in_array, axis=2)
    else:
        mean_array = np.mean(in_array, axis=2)
    return mean_array

def transform_coordinates(coords):
    # Taken from https://stackoverflow.com/questions/47727389/finding-the-closest-ground-pixel-on-an-irregular-grid-for-given-coordinates
    """ Transform coordinates from geodetic to cartesian

    Keyword arguments:
    coords - a set of lan/lon coordinates (e.g. a tuple or
             an array of tuples)
    """
    # WGS 84 reference coordinate system parameters
    A = 6378.137 # major axis [km]
    E2 = 6.69437999014e-3 # eccentricity squared

    coords = np.asarray(coords).astype(np.float)

    # is coords a tuple? Convert it to an one-element array of tuples
    if coords.ndim == 1:
        coords = np.array([coords])

    # convert to radiants
    lat_rad = np.radians(coords[:,0])
    lon_rad = np.radians(coords[:,1])

    # convert to cartesian coordinates
    r_n = A / (np.sqrt(1 - E2 * (np.sin(lat_rad) ** 2)))
    x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
    y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
    z = r_n * (1 - E2) * np.sin(lat_rad)

    return np.column_stack((x, y, z))

def define_timerange(timeA,dt=5):
    timeB = timeA - pd.to_timedelta(dt,unit='days')
    return [timeB,timeA]

# region Setup paths and variables
months = [4, 5, 6, 7, 8, 9]
years = list(range(2000, 2018))
# endregion

# region Daymet
# %%
# Daymet already offers monthly mean for its data, so we'll jsust do normal means + ag season means and call it a day
def DAYMET_Process():
    Path_DAYMET = "F:\\Weather_Grid\\Daymet\\Means\\"
    Daymet_files = list(map(str, pathlib.Path(os.path.expanduser(Path_DAYMET)).rglob('*Monthly*.nc')))
    Daymet_files = [file for file in Daymet_files if file[-7:-3].isdigit() and int(file[-7:-3]) in years]
    for file in Daymet_files:
        ds = xr.open_dataset(file)
        AgSeason_array = ds.copy()
        var = list(ds.variables)[0]
        print(file)
        if 'prcp' in var:
            tmp = ds[var].sum(axis=0)
        else:
            tmp = ds[var].mean(axis=0)
        #  save AgSeason
        AgSeason_array[var] = tmp  # Assign new values to variable
        AgSeason_array = AgSeason_array.drop_vars('time')  # Drop time column cause only one dataset
        year = int(ds.time.dt.year[0])
        fname = f"DAYMET_{var}_AgSeason{year}"
        AgSeason_array.to_netcdf(f"{Path_DAYMET}{fname}.nc")
        # Give projection
        define_proj(f"{Path_DAYMET}{fname}.nc")

    # Normal mean
    # mds = xr.open_mfdataset(f'{Path_DAYMET}*tmax*.nc', concat_dim='cases', preprocess=preproc)
    vars = ['tmax', 'tmin', 'prcp']
    print('Computing monthly normals from monthly data')
    for var in vars:
        print(f"Working with {var}")
        mds = xr.open_mfdataset(f'{Path_DAYMET}*{var}_Monthly*.nc', concat_dim='cases', combine='by_coords')
        mds_normal = mds.groupby('time.month').mean()
        fname = f"DAYMET_{var}_NormalsMonthly"
        mds_normal.to_netcdf(f"{Path_DAYMET}{fname}.nc")
        define_proj(f"{Path_DAYMET}{fname}.nc")

# DAYMET_Process()

# endregion

#%% region XClim

import xclim as xc
import Functions_WeatherManipulations as fcw
import matplotlib.pyplot as plt
from weather_params_standard import *
import pandas as pd
from mapping import *


# (Specific for microbiome project) compute precipitation 7 and 10 days before the sample date --------------
Workdir = r"C:\Users\sebastien durocher\OneDrive - IRDA\Nouveaux_Projets\Microbiome"
filename = "\data_820700_40sites_SRougeriev20220404.xlsx"
df = pd.read_excel(Workdir + filename)
coordinate_dict = dict(zip(['lat2','long2'], Standard_coordinate_format))
df.rename(columns=coordinate_dict,inplace=True)
dc = 'SamplingDate'
df[dc] = pd.to_datetime(df[dc])

path = "D:\\Weather_Grid\\Daymet\\Daily\\"
files = fcw.get_filenames(path,'.nc')
filenames = files.create_filelist(path,'prcp')
print('opening dataset')
stophere
ds = xr.open_mfdataset(filenames,parallel=True,chunks= {'time':-1,'y': 'auto', 'x':'auto'})
# tmp = ds_raw.isel(x=0,y=0)
# tmpdf = tmp['prcp'].to_dataframe()
# ds_raw = xr.open_mfdataset(filenames,parallel=True)
print('Resampling')
# ds= ds.resample(time='D').mean(keep_attrs=True) # Problem with time. Resample to fit with xclim understanding
# ds = ds.chunk({'time': -1, 'x': 'auto', 'y': 'auto'})
# Select small window for testing purposes
# ds = ds.isel(y = slice(0,10),x = slice(0,10))
lat = ds.sel(time='2018-04-01').lat
lon = ds.sel(time='2018-04-01').lon

# get ds index coordinates for each sample
from scipy import spatial
# reshape and stack coordinates
coords = np.column_stack((lat.values.ravel(),
                          lon.values.ravel()))
ground_pixel_tree = spatial.cKDTree(transform_coordinates(coords))
shpe = (881,1201)
idx_df = pd.DataFrame(columns=['x','y'])
for i in range(0,df.shape[0]):
    row_df = df.iloc[i]
    rome = tuple(row_df[['lat','lon']])
    index = ground_pixel_tree.query(transform_coordinates(rome))
    index = np.unravel_index(index[1], shpe)
    idx_df = idx_df.append(pd.Series(index,index=['x','y']),ignore_index=True)

#%% region compute indices from daily data
#
def add_ds_to_df(row,ds):
    out = ds.isel(x = row['x'],y=row['y']).compute()
    out = out.values.mean()
    return out

def add_xclim_indice(df,colname,ds,xcfc):
    df[colname] = np.nan
    ds_xclim = xcfc(ds['prcp'],freq='MS')
    ds_xclim = ds_xclim.groupby('time.year').max()
    vals = idx_df.apply(lambda x : add_ds_to_df(x,ds_xclim),1)
    df[colname] = vals
    return df

df = add_xclim_indice(df,'max1day',ds,xc.indicators.atmos.max_1day_precipitation_amount)
# df['max1day'] = np.nan
# prcp_max1day_ds = xc.indicators.atmos.max_1day_precipitation_amount(ds['prcp'],freq='MS')
# prcp_max1day_ds = prcp_max1day_ds.groupby('time.year').max()
# valz = idx_df.apply(lambda x : add_ds_to_df(x,prcp_max1day_ds),1)
# df['max1day'] = valz

# prcp_maxconsecutivedys_ds = xc.indicators.atmos.maximum_consecutive_wet_days(ds['prcp'],freq='MS')

# Precipitation seasonality
# (The standard deviation of the precipitation estimates expressed as a percentage of the mean of those estimates.)
# xc.indicators.anuclim.P15_PrecipSeasonality(ds['prcp']).groupby('time.year').median()
# this works, but the output is in % and we're getting very small values. Maybe doesn't work with daily data, or it "fucks up" because we don't have annual data

# Only index that somewhat matters
# qté de précipitation durant les jours de pluie
qte_pluie = xc.indicators.cf.sdii(ds['prcp'],freq='MS').groupby('time.year').median()
# endregion

#%% region compute accumulated past rainfall
# (Already computed, load the csv instead)

# construct KD-tree
# There are a lot of assumptions here. Difficult to make on global scale
# But the transform coordinate is good
# output is in mm

time_ranges = [7,14,30]
df[['prcp_7','prcp_14','prcp_30']] = np.nan
for seltime in time_ranges:
    for i in range(0,df.shape[0]):
        print(i)
        row_df = df.iloc[i]
        times = define_timerange(row_df[dc],dt=seltime)
        p1 = ds.sel(time=slice(times[0],times[1]))

        rome = tuple(row_df[['lat','lon']])
        index = ground_pixel_tree.query(transform_coordinates(rome))
        index = np.unravel_index(index[1], shpe)
        sel_ds = p1.isel(x=index[1],y=index[0])

        df.loc[i,f"prcp_{seltime}"] = sel_ds['prcp'].sum().values
# save data
workdir = 'C:\\temp\\'
df.to_csv(workdir+"microbiome_pluiepassée.csv",index=None)
# endregion

# compute gini index ------------------------------------------------------------------
def compute_gini(tmp):
    # assumes one index at the time
    tmpdf = tmp.to_dataframe()
    bins = np.arange(0.1,tmpdf['prcp'].max(),1).tolist()
    bins.append(round(tmpdf['prcp'].max(),2))
    names = [f"{bins[idx-1]}-{bins[idx]}" for idx,nb in enumerate(bins)]
    names = names[1:]
    tmpdf['prcpbins'] = pd.cut(tmpdf['prcp'], bins, labels=names)
    gb = tmpdf.groupby('prcpbins')
    counts = gb.size().to_frame(name='counts')
    gb = counts.join(gb.agg({'prcp': 'sum'}).rename(columns={'prcp': 'prcp_sum'}))
    cumsumdf = gb[['counts','prcp_sum']].cumsum().rename(columns={'counts':'counts_cumsum','prcp_sum':'prcp_cumsum'})
    cumsum_pdf = gb/gb.sum(axis=0)
    cumsum_pdf = 100*cumsum_pdf[['counts','prcp_sum']].cumsum().rename(columns={'counts':'counts_cumsum_p','prcp_sum':'prcp_cumsum_p'})
    lorenz_curve_df = pd.concat([gb,cumsumdf,cumsum_pdf],axis=1)
    # compute coefficients
    X = lorenz_curve_df['counts_cumsum_p']
    Y = lorenz_curve_df['prcp_cumsum_p']
    N = lorenz_curve_df.shape[0]
    lna_num = ((X*X).sum())*(np.log(Y).sum()) + (X.sum())*((X*np.log(X)).sum()) -\
          ((X*X).sum())*(np.log(X).sum()) - (X.sum())*((X*np.log(Y)).sum())
    lna_den = N*((X*X).sum())-(X.sum()**2)
    lna = lna_num/lna_den
    b_num = N*((X*np.log(Y)).sum()) +  (X.sum())*((np.log(X)).sum()) - (N)*((X*np.log(X)).sum()) - (X.sum())*(np.log(Y).sum())
    b_den = N*((X*X).sum())-(X.sum()**2)
    b = b_num/b_den
    # define area under curve
    A100 = (np.exp(lna)/b)*(np.exp(b*100))*(100-(1/b))
    A0 = (np.exp(lna)/b)*(np.exp(0))*(0-(1/b))
    A = A100 - A0
    Gini = (5000 - A)/5000

    return Gini

# df['cum_perc'] = 100*cumsumdf[['counts_cumsum','prcp_cumsum']]/gb[['counts','prcp_sumZZ']].sum()
# Compute the shannon diversity index ----------------------------------------------------------------
# ds_tmp = ds.sel(time=slice('2018-05-23','2018-05-30'))
df['GI'] = np.nan
df['SDI'] = np.nan
df['AWDR'] = np.nan
df['PPT'] = np.nan
for i in range(0,df.shape[0]):
    print(i)
    row_df = df.iloc[i]

    rome = tuple(row_df[['lat','lon']])
    index = ground_pixel_tree.query(transform_coordinates(rome))
    index = np.unravel_index(index[1], shpe)
    ds_tmp = ds.isel(y=index[1],x=index[0]).chunk({'time':-1}).compute()

    PPT = ds_tmp['prcp'].groupby('time.year').sum()
    P = xr.apply_ufunc(lambda x,t : x/t, ds_tmp['prcp'].groupby('time.year'),PPT,dask='allowed',vectorize=True)
    n = ds_tmp.groupby('time.year').count() # number of days used in each calculation
    n = n.rename_dims(dims_dict={'year':'time'})
    SDI = (-(P*np.log(P)).resample(time = '1Y').sum())/np.log(n)
    SDI = SDI.drop('year')
    # data = np.array(SDI['prcp'].mean(dim='time'))
    # save index

    # Compute abundant and well-distributed rainfall
    AWDR = PPT*SDI.rename_dims(dims_dict={'time':'year'})

    df.loc[i,'PPT'] = PPT.values.mean()
    df.loc[i,'SDI'] = SDI['prcp'].values.mean()
    df.loc[i,f"AWDR"] = AWDR['prcp'].values.mean()
    df.loc[i,f"GI"] = compute_gini(ds_tmp['prcp'])


# endregion


# Alternative to open the netcdf files
import netCDF4 as nc
ds = nc.Dataset(filenames[0])