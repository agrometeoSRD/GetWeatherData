"""
File: utils.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description: This is a description of what the script does
Created: 2024-03-04
"""

# Import statements
import os
import configparser

# Constants

# Functions
def load_eccc_forecast_config_file():
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Join the script directory with the name of the configuration file
    config_file_path = os.path.join(script_dir, 'config.ini')

    # Check if the configuration file exists
    if not os.path.isfile(config_file_path):
        raise FileNotFoundError(f"Configuration file does not exist: {config_file_path}")

    config = configparser.ConfigParser()
    config.read(config_file_path)

    return config

# Main execution ---------------------------------------

if __name__ == "__main__":
    pass
