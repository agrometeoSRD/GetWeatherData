"""
File: utils.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description: functions that are applicable to various scripts in the projects
Created: 2024-03-04
"""

# Import statements
import os
from pathlib import Path
import json
import pandas as pd

# Constants

# Functions

def load_config(config_filename: str = "sm_config.json"):
    # First, try to find the config file in the current working directory
    cwd = Path.cwd()
    config_path = cwd / config_filename

    if not config_path.is_file():
        # If not found, look in the package directory
        package_dir = Path(__file__).parent.parent
        config_path = package_dir / 'config' / config_filename

    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file '{config_filename}' not found in current directory or package directory.")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # If 'Paths' in config, make them absolute based on the config file location
    if 'Paths' in config:
        for key, path in config['Paths'].items():
            absolute_path = (config_path.parent / path).resolve()
            config['Paths'][key] = str(absolute_path)

    return config

def get_package_directory():
    return Path(__file__).parent.parent

# You might want to keep this function for backwards compatibility
def get_project_root(current_directory: Path) -> Path:
    return get_package_directory()

def invert_mapping(mapping:dict):
    'takes a dictionnary (expected from column_names.json) and inverts it'
    inverted_map = {} # initialise the inverted map
    for standard, variations in mapping.items(): # loop over each key
        for variation in variations: # loop over its possible variation
            inverted_map[variation] = standard
    return inverted_map

# Verification function that checks if the columns are standardized
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    universal_mapping = load_config('column_names.json') # load config file that contains mapping of column names
    standardize_columns_dict = invert_mapping(universal_mapping)
    # Apply the dictionnary map to replace columnns (if column are in the keys)
    return df.rename(columns=standardize_columns_dict)

# Main execution ---------------------------------------

if __name__ == "__main__":
    pass
