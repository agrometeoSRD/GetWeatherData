"""
File: to_rimpro.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Created: 2024-03-08

Description:
- Convert specific csv files to RIMPro format
- Currently only supports the following files : ec forecasts, ec virtual station nowcasts and bru+nowcast

Note
- Must specify the .csv file extension in naming.

- The output format should look like this
Compton
DATE	TIME	AIRTEMP	AIRHUM	RAIN	LW1
2022-02-24	23:00:00	-12.291052	65.531517	0.0	-991
2022-02-25	00:00:00	-12.24201	61.508823	0.0	-991
2022-02-25	01:00:00	-12.102331	58.036999	0.0	-991

- To use with the CLI. Must be in same directory as the python script. Here's an example.
python to_rimpro.py --source ec_forecasts --suffix _saved_forecast.csv
"""

# Import statements
import argparse
import sys
import os
import glob
import numpy as np
import pandas as pd
from utils.utils import load_config

# Constants

# Functions
def load_saved_csv(id, path_input, file_suffix):
    InFile = os.path.join(path_input, f"{id}{file_suffix}.csv")
    # Read file as pandas, but throw error if file does not exist (print file name to check if properly written)
    try:
        df = pd.read_csv(InFile, sep=';')
        return df
    except Exception as e:
        print(f"Failed to load file: {InFile} : {e}")
        return

def to_RIMpro_format(df, forecast_variables, rimpro_headers):
    df_ForRIMpro = (df[forecast_variables + ['Date']]
                    .assign(**df['Date'].astype(str).str.rsplit(' ', n=1, expand=True).rename(columns={0: 'DATE', 1: 'TIME'}))
                    .loc[:, ['DATE', 'TIME'] + forecast_variables]
                    .apply(lambda x: x.round(3) if x.name in forecast_variables else x)
                    .rename(columns=dict(zip(['DATE', 'TIME'] + forecast_variables, rimpro_headers)))
                    .astype(str)
                    .assign(TIME=lambda x: x['TIME'].str[:-3]))
    df_ForRIMpro['LW1'] = '-991'
    return df_ForRIMpro

def write_df_to_RIMpro_csv(df, path_output, staname, ID, file_suffix):
    OutFile = os.path.join(path_output, f"FRP_{ID}{file_suffix}.csv")
    with open(OutFile, 'w+') as OF:
        print(f'Writing file to RIMpro : {OutFile}')
        OF.write(staname + "\n")
        df.to_csv(OF, index=False, sep=';', lineterminator='\n')

def define_file_path(config, file_suffix):
    if 'ec_forecasts' in file_suffix:
        return config['Paths']["SavedEcForecastsPath"]
    elif 'ec_vs_forecasts' in file_suffix:
        return config['Paths']["SavedEcVsForecastsPath"]
    elif 'bru_nowcast' in file_suffix:
        return config['Paths']["SavedEcVsForecastsPath"]
    else:  # raise error saying invalid source file
        print('Invalid source file')
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Process forecasts and convert to RIMPro format.")
    parser.add_argument("--source", help="Define file location (ec_forecasts, ec_vs_forecasts, bru_nowcast). File location must exist",default='bru_nowcast')
    parser.add_argument("--suffix", help="Specify suffix to load the correct file (always assume csv). File must exist before being converted to RIMpro.",default='_bru_nowcast')
    return parser.parse_args()

def process_forecasts(config:dict, file_suffix:str, source:str):
    source_path = define_file_path(config, source)
    path_to_rimpro = config['Paths']["SavedRIMproPath"]
    variables = config['General']
    forecast_variables = [variables['temp_col'], variables['hr_col'], variables['rain_col'], variables['rad_col']]
    rimpro_headers = ['DATE', 'TIME', 'AIRTEMP', 'AIRHUM', 'RAIN', 'GLOBALRAD']  # Note : DATE and TIME should not change

    reading_csv_files = glob.glob(os.path.join(source_path, f'*{file_suffix}.csv'))
    # Process each CSV file
    for csv_file in reading_csv_files:
        try:
            station_id = os.path.basename(csv_file).split('_')[0]

            df = load_saved_csv(station_id, source_path, file_suffix)
            df_ForRIMpro = to_RIMpro_format(df, forecast_variables, rimpro_headers).replace('nan', np.nan).fillna(-991)
            write_df_to_RIMpro_csv(df_ForRIMpro, path_to_rimpro, station_id, station_id, file_suffix)
        except Exception as e:
            print(f"Error processing station {csv_file}: {e}")

# Main execution ---------------------------------------
if __name__ == "__main__":
    # path examples :` config['Paths']["SavedEcForecastsPath"] or config['Paths']["SavedEcVsForecastsPath"]
    args = parse_args()
    config = load_config('ec_config.json')
    process_forecasts(config, args.suffix, args.source)

# TODO : Add condition if folder location doesn't exist
# TODO : Add condition if file doesn't exist
