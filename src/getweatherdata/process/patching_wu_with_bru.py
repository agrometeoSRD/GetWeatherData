"""
File: patching_wu_with_bru.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description: If personal WU station is missing, then this script will fetch the data from the BRU station.
Steps
 1. Define and load the WU station that needs patching
 2. Define and load the BRU station that will be used to patch the WU station
 3. Get matching columns between both stations. Fill with nans if no match
 4. Save to new WU station
Created: 2024-04-12
"""

# Import statements
import pandas as pd
import os
from ..utils.utils import load_config

# Constants

# Functions

# Main execution ---------------------------------------
config = load_config('ec_config.json')
wu_station_name = "IHEMMI29.csv"
bru_station_name = "FRP_HEMGFD_bru_nowcast.csv"
wu_df = pd.read_csv(f"{config['Paths']['SavedRIMproPath']}\\{wu_station_name}",sep=';',skiprows=1)
bru_df = pd.read_csv(f"{config['Paths']['SavedRIMproPath']}\\{bru_station_name}",sep=';',skiprows=1)
# remove all bru_df after 2024-04-24 24:00:00
bru_df_past = bru_df[bru_df['DATE'] <= '2024-04-24']
matching_columns = wu_df.columns.intersection(bru_df_past.columns)
new_wu_df = bru_df_past[matching_columns].copy()
new_wu_df['TotalRain'] = '0.0'
new_wu_df.columns = ['DATE', 'TIME', 'AIRHUM', 'AIRTEMP', 'RAIN', 'TotalRain','LW1']
new_wu_df['LW1'] = '-991'
new_wu_df[['AIRHUM','AIRTEMP','RAIN']] = new_wu_df[['AIRHUM','AIRTEMP','RAIN']].astype(str)

OutFile = os.path.join(config['Paths']['SavedRIMproPath'], f"IHEMMI29_n.csv")
staname = "Hemmingford"
with open(OutFile, 'w+') as OF:
    print(f'Writing file to RIMpro : {OutFile}')
    OF.write(staname + "\n")
    new_wu_df.to_csv(OF, index=False, sep=';', lineterminator='\n')

# compare the two data files
# merge 'DATE' and 'TIME' column together
# only select data between april 20th to april 30th
bru_df_sel = bru_df[(bru_df['DATE'] >= '2024-04-20') & (bru_df['DATE'] <= '2024-04-30')]
wu_df_sel = wu_df[(wu_df['DATE'] >= '2024-04-20') & (wu_df['DATE'] <= '2024-04-30')]
bru_df_sel['DATETIME'] = pd.to_datetime(bru_df_sel['DATE']) + pd.to_timedelta(bru_df_sel['TIME'].str.split(':').str[0].astype(int), unit='h')
wu_df_sel['DATETIME'] = pd.to_datetime(wu_df_sel['DATE']) + pd.to_timedelta(wu_df_sel['TIME'].str.split(':').str[0].astype(int), unit='h')
# plot time series of temperature data
import matplotlib.pyplot as plt
plt.plot(bru_df_sel['DATETIME'],bru_df_sel['RAIN'],label='BRU')
plt.plot(wu_df_sel['DATETIME'],wu_df_sel['RAIN'],label='WU')
plt.legend()
plt.show()

# compute mean difference between relative humidity
diff = wu_df_sel['AIRHUM'].astype(float).reset_index(drop=True) - bru_df_sel['AIRHUM'].astype(float).reset_index(drop=True)

