"""
File: microbiome_request.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description:
- This is not to share with the client. This is to be used internally to request the microbiome data from the ESSAQ project.
- The data will be used to create a report for the client.

Created: 2024-04-03
"""

# Import statements
import os
import pandas as pd

# Constants

# Functions
def read_essaq_microbiome():
    """
    This function reads the microbiome_indices.xlsx file from the ESSAQ project
    and returns a pandas DataFrame.
    """
    print('Opening microbiome_indices.xlsx file')
    essaq_microbiome_file = f"C:\\Users\\{os.getenv('USERNAME')}\\OneDrive - IRDA\\Chargé de projets\\Petits projets\\Microbiome\\microbiome_indices.xlsx"
    essaq_microbiome_original_df = pd.read_excel(essaq_microbiome_file)
    return essaq_microbiome_original_df

# Main execution ---------------------------------------

