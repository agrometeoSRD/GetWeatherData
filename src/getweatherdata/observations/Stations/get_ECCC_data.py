"""
File: get_ECCC_data.py
Author: sebastien.durocher
Python : 3.11
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description: Get weatherdata from env_canada local

Current status (2024-08-08) : Works but was not really tested

Created: 2024-03-08
"""

# meteorological_data_module.py

import pandas as pd
import asyncio
from env_canada import ECHistoricalRange
from env_canada.ec_historical import get_historical_stations
from datetime import datetime

# List of available meteorological variables
ENV_CANADA_STATION_VARIABLES = ['Longitude (x)', 'Latitude (y)', 'Station Name', 'Climate ID', 'Year',
                           'Month', 'Day', 'Data Quality', 'Max Temp (°C)', 'Max Temp Flag',
                           'Min Temp (°C)', 'Min Temp Flag', 'Mean Temp (°C)', 'Mean Temp Flag',
                           'Heat Deg Days (°C)', 'Heat Deg Days Flag', 'Cool Deg Days (°C)',
                           'Cool Deg Days Flag', 'Total Rain (mm)', 'Total Rain Flag',
                           'Total Snow (cm)', 'Total Snow Flag', 'Total Precip (mm)',
                           'Total Precip Flag', 'Snow on Grnd (cm)', 'Snow on Grnd Flag',
                           'Dir of Max Gust (10s deg)', 'Dir of Max Gust Flag',
                           'Spd of Max Gust (km/h)', 'Spd of Max Gust Flag']

class GetEnvCanadaHistoricalStationData:
   """
   A module to extract and manage meteorological data from ECCC

   This module provides the following functionalities:
   1. Allows the user to specify any desired time intervals and get the corresponding data.
   2. Allows the user to request the desired meteorological variables.
   3. Reads shapefiles or GeoJSON of specific point coordinates and uses the point coordinates as input.
   4. Allows local saving of data into CSV format.
   """

   def __init__(self, station_id: int, timeframe: str = "daily", start_date: datetime = None, end_date: datetime = None):
       """
       Initialize the MeteorologicalData object with the specified station ID, timeframe, and date range.

       Args:
           station_id (int): The ID of the meteorological station.
           timeframe (str, optional): The timeframe for the data, either "daily" or "hourly". Defaults to "daily".
           start_date (datetime, optional): The start date of the desired data range. Defaults to None.
           end_date (datetime, optional): The end date of the desired data range. Defaults to None.
       """
       self.station_id = station_id
       self.timeframe = timeframe
       self.start_date = start_date
       self.end_date = end_date
       self.data = None

   def get_station_data(self) -> pd.DataFrame:
       """
       Retrieve the meteorological data from the specified station and date range.

       Returns:
           pd.DataFrame: The meteorological data for the specified station and date range.
       """
       ec = ECHistoricalRange(station_id=self.station_id, timeframe=self.timeframe,
                              daterange=(self.start_date, self.end_date))
       self.data = ec.get_data()
       return self.data

   def get_available_variables(self) -> list:
       """
       Get a list of available meteorological variables.

       Returns:
           list: A list of available meteorological variables.
       """
       return ENV_CANADA_STATION_VARIABLES

   def extract_variable(self, variable: str) -> pd.DataFrame:
        """
        Extract a specific meteorological variable from the data, along with key metadata information.

        Args:
            variable (str): The name of the meteorological variable to extract.

        Returns:
            pd.DataFrame: A DataFrame containing the requested variable and key metadata information.
        """
        if variable not in ENV_CANADA_STATION_VARIABLES:
            raise ValueError(f"Invalid variable name: {variable}")

        # Get the index of the requested variable
        var_index = ENV_CANADA_STATION_VARIABLES.index(variable)

        # Get the flag variable index
        flag_index = var_index + 1 if (var_index + 1) < len(ENV_CANADA_STATION_VARIABLES) and ENV_CANADA_STATION_VARIABLES[var_index + 1].endswith('Flag') else None

        # Extract the key metadata columns
        metadata_columns = ['Longitude (x)', 'Latitude (y)', 'Station Name', 'Climate ID', 'Year', 'Month', 'Day', 'Data Quality']

        # Create a new DataFrame with the requested variable and metadata
        result_df = self.data[metadata_columns + [variable]]

        # Add the flag column if it exists
        if flag_index is not None:
            result_df[ENV_CANADA_STATION_VARIABLES[flag_index]] = self.data[ENV_CANADA_STATION_VARIABLES[flag_index]]

        return result_df

   def save_to_csv(self, file_path: str):
       """
       Save the meteorological data to a CSV file.

       Args:
           file_path (str): The file path to save the CSV file.
       """
       self.data.to_csv(file_path, index=False)

   @classmethod
   def get_stations_by_coordinates(cls, coordinates: list, start_year: int = 2022, end_year: int = 2024, radius: int = 200, limit: int = 100) -> pd.DataFrame:
       """
       Retrieve a list of meteorological stations based on the given coordinates.

       Args:
           coordinates (list): A list of latitude and longitude coordinates.
           start_year (int, optional): The start year for the station search.
           end_year (int, optional): The end year for the station search.
           radius (int, optional): The search radius in kilometers.
           limit (int, optional): The maximum number of stations to return.

       Returns:
           pd.DataFrame: A DataFrame containing the stations' information.
       """
       stations = pd.DataFrame(asyncio.run(get_historical_stations(coordinates, start_year=start_year,
                                                                   end_year=end_year, radius=radius, limit=limit))).T
       return stations


def main():
    # 1. Get a list of stations for a specific set of coordinates
    coordinates = ['45.289713', '-74.319667']
    md = GetEnvCanadaHistoricalStationData(station_id=None, start_date=None, end_date=None)
    stations = md.get_stations_by_coordinates(coordinates)
    print("Available stations:")
    print(stations)

    # 2. Get the meteorological data for a specific station and date range
    station_id = int(stations.iloc[0, 2])
    start_date = datetime(2022, 7, 1)
    end_date = datetime(2024, 6, 1)
    md = GetEnvCanadaHistoricalStationData(station_id=station_id, start_date=start_date, end_date=end_date)
    station_df = md.get_station_data()
    print("\nRetrieved meteorological data:")
    print(station_df.head())

    # 3. Extract a specific meteorological variable
    variable = 'Mean Temp (°C)'
    temperature_data = md.extract_variable(variable)
    print(f"\nExtracted {variable} data:")
    print(temperature_data.head())

    # 4. Save the data to a CSV file
    file_path = 'meteorological_data.csv'
    md.save_to_csv(file_path)
    print(f"\nData saved to {file_path}")

if __name__ == "__main__":
    main()
