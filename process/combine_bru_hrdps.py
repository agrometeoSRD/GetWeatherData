"""
File: combine_bru_hrdps.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747

Description: Some .BRU dont have solar radiation. Use HRDPS solar radiation to fill in the missing data
- For code to work, first must have .BRU files and HRDPS solar radiation forecast
- .BRU files can be accessed by imported get_SM_data.py
- HRDPS files are locally saved in a folder. These are created by fisrt downloading ec_forecasts.py and then saving with save_ec_nowcast.py

Created: 2024-02-27
"""

# Import statements
from source.Observations.Stations.get_SM_data import download_and_process_data

# Constants

# Functions
# Check which stations have solar radiation

# Load the .BRU files for selected stations

# Load the HRDPS solar radiation data for selected stations

# Combine the .BRU and HRDPS data

# Save the combined data as .csv

# Main execution ---------------------------------------
def main():
    sel_station = ['Compton']
    sel_years = ['2020']
    bru_df = download_and_process_data(sel_station, sel_years)

if __name__ == "__main__":
    pass
