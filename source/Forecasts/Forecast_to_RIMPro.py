#!/usr/bin/env
"""
Creation date: 2023-12-13
Creator : the_l
Python version : 3.10

Description:

Notes:
    - Will probably not stay within this subfolder
    - Input takes forecast data from a csv file and converts the parameters to RIMPro format
    - The output format should look like this
Compton
DATE	TIME	AIRTEMP	AIRHUM	RAIN	LW1
2022-02-24	23:00:00	-12.291052	65.531517	0.0	-991
2022-02-25	00:00:00	-12.24201	61.508823	0.0	-991
2022-02-25	01:00:00	-12.102331	58.036999	0.0	-991

With filename
FRP_COMPTON.CSV

"""
# TODO : automatic RIMpro forecast folder creation  when it doesn't exist

# imports
import os
import numpy as np
import pandas as pd

# Define variables
input_forecast_headers = ["AIRTEMP", "HR", "RAIN","GLOBALRAD"]
rimpro_headers = ['DATE', 'TIME', 'AIRTEMP', 'AIRHUM', 'RAIN', 'GLOBALRAD'] # Note : DATE and TIME should not change


# Convert to RIMPro format
def to_RIMpro_format(df):
    '''
    Following columns must be present : all Forecast_VariableS_List and 'Date' column


    :param df: pandas dataframe
    :return: pandas dataframe
    '''
    df_ForRIMpro = df[input_forecast_headers + ['Date']]

    df_ForRIMpro[['DATE', 'TIME']] = df_ForRIMpro["Date"].astype(str).str.rsplit(' ', n=1,expand=True)  # expand date
    df_ForRIMpro = df_ForRIMpro[['DATE', 'TIME'] + input_forecast_headers]  # shift columns
    df_ForRIMpro.columns = rimpro_headers
    df_ForRIMpro['DATE'] = df_ForRIMpro['DATE'].str.replace('-', '/')

    # Convert all values to strings
    df_ForRIMpro = df_ForRIMpro.astype(str)
    return df_ForRIMpro

# Save as csv
def write_df_to_RIMpro_csv(df : pd.DataFrame, path_output : str, staname : str, ID : str):
    # Saving will overwrite previous file if it already existed
    OutFile = path_output + "\\FRP_" + ID + ".csv"
    with open(OutFile, 'w+') as OF:  # Writes down to file
        print('Writting forecast file for station: ' + staname)
        OF.write(staname + "\n")
        df.to_csv(OF, index=False, sep=';',lineterminator='\n')

# %% Read station information and
path_to_station_file = r"C:\Scripts\PycharmProjects\GetWeatherData\source\Forecasts"
path_to_forecasts = r"C:\Scripts\PycharmProjects\GetWeatherData\source\Forecasts\saved_forecasts"
path_to_rimpro = r"C:\Scripts\PycharmProjects\GetWeatherData\source\Forecasts\rimpro_forecasts"
InFile = os.path.join(path_to_station_file, 'VStations_test.dat')
Stations_info = pd.read_csv(InFile, skiprows=2)

# load saved csv file of every forecast
def load_saved_csv(id, path_input):
    '''
    :param ID: string
    :param Path_Input: string
    :return: pandas dataframe
    '''
    InFile = f"{path_input}\\{id}_saved_forecast.csv"
    df = pd.read_csv(InFile, sep=';')
    return df

# Loop over all stations found in the station file
for index, station in Stations_info.iterrows():
    # convert to RIMpro format
    df = load_saved_csv(station['ID'], path_to_forecasts)
    df_ForRIMpro = (to_RIMpro_format(df)
                    .replace('nan', np.nan)
                    .fillna(-991)) # replace any missing nans with -991 (The rimpro equivalent for nans)

    # print(df_ForRIMpro)
    write_df_to_RIMpro_csv(df_ForRIMpro, path_to_rimpro, station['Name'],station['ID'])

