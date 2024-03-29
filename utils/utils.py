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
from pathlib import Path
import json

# Constants

# Functions

def get_project_root(current_directory: Path) -> Path:
    if (current_directory / 'config').exists():
        return current_directory
    parent_directory = current_directory.parent
    if parent_directory == current_directory:
        raise FileNotFoundError("Failed to find the project root directory.")
    return get_project_root(parent_directory)

def load_config(config_filename: str = "config.json"):
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

# Main execution ---------------------------------------

if __name__ == "__main__":
    pass
