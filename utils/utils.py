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

def get_project_root(current_directory: Path) -> Path:
    if (current_directory / 'config').exists():
        return current_directory
    parent_directory = current_directory.parent
    if parent_directory == current_directory:
        raise FileNotFoundError("Failed to find the project root directory.")
    return get_project_root(parent_directory)

def load_config(config_filename: str = "sm_config.json"):
    # Get the directory of the root folder
    project_root = get_project_root(Path(__file__).parent)

    # Get the path to the configuration file
    config_path = project_root / 'config' / config_filename

    # Check if the configuration file exists
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

    with open(config_path, 'r') as f:
         config = json.load(f)

    # Attach the paths from config to the root path of the project
    if 'Paths' in config:
        for key, path in config['Paths'].items():
            absolute_path = (project_root / path).resolve()
            config['Paths'][key] = str(absolute_path)

    return config

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
