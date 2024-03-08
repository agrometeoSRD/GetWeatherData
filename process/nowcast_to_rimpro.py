#!/usr/bin/env
"""
Creation date: 2024-02-21
Creator : the_l
Python version : 3.10

Description:
NOTE : OUTDATE. SEE TO_RIMPRO.PY
- This is a direct copy of forecast_to_rimpro.py
- 2024-02-21 : as of now, the only difference is the different paths

Notes:
    - Will probably not stay within this subfolder
    - Input takes forecast data from a csv file and converts the parameters to RIMPro format
    - The output format should look like this
    - MUST HAVE A COLUMN FOR LEAF WETNESS (LF1) EVEN IF ITS NOT USED (JUST -991)

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
import sys
import os
import numpy as np
import pandas as pd
from utils.utils import load_config

# load saved csv file of every forecast
def load_saved_csv(id, path_input):
    InFile = os.path.join(path_input, f"{id}_vs.csv")
    return pd.read_csv(InFile, sep=';')

# Convert to RIMPro format
def to_RIMpro_format(df):
    '''
    Following columns must be present : all Forecast_Variables_List and 'Date' column
    '''
    df_ForRIMpro = (df[forecast_variables + ['Date']]
                    .assign(
        **df['Date'].astype(str).str.rsplit(' ', n=1, expand=True).rename(columns={0: 'DATE', 1: 'TIME'}))
                    .loc[:, ['DATE', 'TIME'] + forecast_variables]
                    .apply(lambda x: x.round(3) if x.name in forecast_variables else x)
                    .rename(columns=dict(zip(['DATE', 'TIME'] + forecast_variables, rimpro_headers)))
                    .astype(str)
                    .assign(TIME = lambda x: x['TIME'].str[:-3])
                    )
    # add -991 to the LW1 column
    df_ForRIMpro['LW1'] = '-991'

    return df_ForRIMpro


# Save as csv
def write_df_to_RIMpro_csv(df, path_output, staname, ID):
    OutFile = os.path.join(path_output, f"FRP_{ID}_vs.csv")
    with open(OutFile, 'w+') as OF:
        print('Writting forecast file for station: ' + staname)

        OF.write(staname + "\n")
        df.to_csv(OF, index=False, sep=';', lineterminator='\n')



# %% Read station information and
def process_forecasts(config):
    
    path_to_rimpro = config['Paths']["SavedRIMproPath"]
    path_to_vs_save = config['Paths']["SavedEcVsForecastsPath"]

    # Load station info
    # InFile = os.path.join(path_to_script, 'VStations_p1.dat') uncomment for deployment
    InFile = os.path.join(config['Paths']['TestPath'], 'vs_stations_test.dat')
    try:
        stations_info = pd.read_csv(InFile, skiprows=2)
    except Exception:
        sys.exit(1)

    for _, station in stations_info.iterrows():
        # convert to RIMpro format
        df = load_saved_csv(station['ID'], path_to_vs_save)
        df_ForRIMpro = (to_RIMpro_format(df).replace('nan', np.nan).fillna(-991))  # replace any missing nans with -991 (The rimpro equivalent for nans)
        write_df_to_RIMpro_csv(df_ForRIMpro, path_to_rimpro, station['Name'], station['ID'])

if __name__ == "__main__":
    # Define path to configuration file
    config = load_config('ec_config.json')
    # Define variables
    rain_col = "RAIN [mm]"
    rimpro_headers = ['DATE', 'TIME', 'AIRTEMP', 'AIRHUM', 'RAIN', 'GLOBALRAD']  # Note : DATE and TIME should not change
    variables = config['General']
    forecast_variables = [variables['temp_col'], variables['hr_col'], variables['rain_col'], variables['rad_col']]
    process_forecasts(config)