"""
File: tmp.py
Author: sebastien.durocher
Email: sebastien.durocher@example.com
Github: https://github.com/sebastien.durocher
Description: read all csvs and convert them into a single xarray dataset
Created: 2024-01-16
"""

import pandas as pd
import os
import numpy as np
import xarray as xr
import glob
from functools import reduce

def compute_growing_dd(min_temp, max_temp, threshold: float):
    """Compute growing degree days for a given threshold. Returns daily DD"""
    return (((min_temp + max_temp) / 2) - threshold).clip(lower=0)  # clip(lower=0) sets negative values to 0

def gini(array: np.ndarray) -> float:
    # to input : an array that contains the daily time series data of interest

    # Taken from https://github.com/oliviaguest/gini/blob/master/gini.py
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    type(array)
    array = np.asarray(array).flatten()
    array[np.isnan(array)] = 0
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array = array + 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

# Get a list of all CSV file names
csv_files = glob.glob(r'C:\Users\sebastien.durocher\PycharmProjects\GetWeatherData\source\Observations\Daymet\data\*.csv')
daymet_meta = xr.open_dataset(r"D:\Observations\Daymet\Daymet_metadata.nc")
essaq_microbiome_file = f"C:\\Users\\{os.getenv('USERNAME')}\\OneDrive - IRDA\\Chargé de projets\\Petits projets\\Microbiome\\microbiome_indices.xlsx"
essaq_microbiome_original_df = pd.read_excel(essaq_microbiome_file)

start_date = ''

# Loop over the CSV files
essaq_microbiome_with_indices_list = []
i = 0
for file in csv_files:
    print(f"Processing file {i} of {len(csv_files)}")
    siteid = file.split('\\')[-1][:-4].upper()
    target_row = essaq_microbiome_original_df[essaq_microbiome_original_df.Site_id == siteid]
    target_sampling_date = target_row.SamplingDate.values[0]
    target_sampling_year = pd.to_datetime(target_sampling_date).year

    # Read the CSV file into a DataFrame
    df_meta = pd.read_csv(file, nrows=0)
    df_data = pd.read_csv(file,skiprows=6)
    df_data.rename(columns={'tmax (deg c)':'tmax','tmin (deg c)':'tmin','prcp (mm/day)':"prcp"}, inplace=True)
    # get five years of data before the target sampling year
    df_data = df_data[(df_data.year <= target_sampling_year) & (df_data.year >= target_sampling_year - 5)]
    # Get the column name
    col_name = df_meta.columns[0]
    # Split the column name by spaces
    parts = col_name.split()
    # The latitude and longitude values are at the second and fifth positions in the list
    latitude = float(parts[1])
    longitude = float(parts[3])

    df_data['time'] = pd.to_datetime(df_data['year'].astype(str) + df_data['yday'].astype(str), format='%Y%j')
    df_data.set_index('time', inplace=True)

    # compute gdd -------------------------------------------------------
    gdd_df = df_data.copy()
    gdd_df = gdd_df[(gdd_df.index.month >= 4) & (gdd_df.index.month <= 10)]
    gdd_result = gdd_df.groupby(gdd_df.index.year).apply(lambda group: compute_growing_dd(group['tmin'], group['tmax'], 5)).groupby(level=0).sum().reset_index().rename(columns={0:'GDD(5)','time':'year'})

    # compute seasonal precipitation -----------------------------------
    PPT_df = df_data.copy()
    PPT_df = PPT_df[(PPT_df.index.month >= 4) & (PPT_df.index.month <= 10)]
    PPT_df = PPT_df['prcp'].resample('Y').sum().reset_index()
    PPT_df['year'] = PPT_df['time'].dt.year
    PPT_df = PPT_df.drop(columns=['time']).rename(columns={'prcp':'PPT'})

    # Compute sdi (with pandas) -----------------------------------------
    sdi_df = df_data.copy()
    # Set 'time' as the index
    # Resample the DataFrame to a monthly frequency and sum the 'prcp' values
    monthly_precip = sdi_df['prcp'].resample('M').sum()
    # Group the monthly precipitation data by year and transform each value
    pi = monthly_precip.groupby(monthly_precip.index.year).apply(lambda x: x / x.sum())
    pi.index = pi.index.rename(['year', 'time'])
    n = 12
    SDI = -1*((pi*np.log(pi)).groupby(level = 'year').sum())/np.log(n)
    SDI = SDI.reset_index().rename(columns={'prcp':'SDI'})

    # compute AWDR
    precip_yearly = sdi_df['prcp'].resample('Y').sum()
    AWDR = precip_yearly*SDI['SDI'].values
    AWDR = (
        AWDR.reset_index()
        .rename(columns={'prcp':'AWDR'})
        .assign(year=lambda df: df['time'].dt.year)
        .drop(columns=['time'])
    )

    #compute gini (with pandas)--------------------------------------------------
    gini_df = df_data.copy()
    gini_df = gini_df[(gini_df.index.month >= 4) & (gini_df.index.month <= 10)]
    gini_result = gini_df.groupby(gini_df.index.year)['prcp'].apply(gini)
    gini_result = gini_result.reset_index().rename(columns={'prcp':'gini','time':'year'})

    # Restructure everything so that it can fit nicely into a single pandas dataframe
    df_list = [gini_result, AWDR, SDI,PPT_df, gdd_result]
    indices_frame = reduce(lambda  left,right: pd.merge(left,right,on=['year'],
                                                how='outer'), df_list)

    target_row = target_row.merge(indices_frame,how='cross')
    essaq_microbiome_with_indices_list.append(target_row)

    i += 1

essaq_microbiome_with_indices = pd.concat(essaq_microbiome_with_indices_list)
essaq_microbiome_with_indices.drop(columns=['prcp_7','prcp_14','prcp_30','GI','SDI_x','AWDR_x','PPT_x'],inplace=True)
essaq_microbiome_with_indices.to_csv(r"C:\Users\sebastien.durocher\PycharmProjects\GetWeatherData\source\Observations\Daymet\essaq_microbiome_with_indices.csv",index=False)