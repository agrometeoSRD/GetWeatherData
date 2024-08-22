"""
File: noaa_forecast.py
Author: sebastien.durocher
Python : 3.11
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/agrometeoSRD
Status : Functional, but not tested

Description:
    This script extracts weather forecast data from the NOAA Open Data Dissemination (NODD) Program, focusing specifically on the High-Resolution Rapid Refresh (HRRR) model.
    It allows users to download, subset, and process HRRR data for specified variables and geographical coordinates. The script follows a tutorial provided by https://nbviewer.org/github/microsoft/AIforEarthDataSets/blob/main/data/noaa-hrrr.ipynb.

Features:
    - Extracts HRRR model data for specific time ranges and variables.
    - Subsets data to the Quebec region by default using a GeoJSON file.
    - Supports subsetting by custom geographical coordinates (GeoJSON, Shapefiles, or direct lat/lon input).
    - Combines data from multiple time steps into a single xarray Dataset.
    - Includes options to save the output as NetCDF or CSV files.

Usage:
    - Run `main()` to see a basic example of how to use the script to extract and process HRRR data.
    - Ensure that the AREA_REDUCER_PATH points to a valid GeoJSON file for subsetting the data to the Quebec region.
    - Modify the `variables` list and `time_list` in `main()` to customize the extraction process.

Created: 2024-08-05
"""

# Import statements
import utils.subsetting as subsetting_util
import os
from datetime import date, datetime, timedelta
import requests
import tempfile
import logging
from typing import List, Dict, Union, Tuple, Sequence
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import Point
from clisops.core import subset
import warnings

# Constants
CONFIG_FILE_NAME = "config.json" # this doesn't do anything, but eventually should be used to standardize variables and data format

# AREA_REDUCER is a geojon file that contains the boundaries of Quebec only, since HRRR is for all of NA
AREA_REDUCER_PATH = subsetting_util.get_shape_filename()
# Functions
class HRRRDataExtractor:
    def __init__(self, config_file: str = CONFIG_FILE_NAME):
        """
        Initialize the HRRR data extractor.

        :param config_file: Path to the JSON configuration file for variable mapping
        """
        #todo : create better condition for if subsetting to quebec should be an option or not (now it always is)
        print('Subsetting data to Quebec area only')

        #todo : add config file to standardize file naming
        # self.config = self.load_config(CONFIG_FILE_NAME)
        self.blob_container = "https://noaahrrr.blob.core.windows.net/hrrr"
        self.sector = "conus"
        self.product = "wrfsfcf"

        # Configure logger
        logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def get_hrrr_data(self, time_list: List[datetime],
                      variables: List[str], coordinates: Union[str, gpd.GeoDataFrame, pd.DataFrame, Dict[str, Sequence[float]]] = None) -> xr.Dataset:
        """
        Extract HRRR data for specified time range, variables, and coordinates.

        :param start_time: Start time for data extraction
        :param end_time: End time for data extraction
        :param variables: List of variables to extract
        :param coordinates: GeoJSON/Shapefile path or GeoDataFrame for subsetting
        :return: xarray Dataset containing extracted data
        """
        file_list = self._get_azure_file_list(time_list)

        # todo : Use multithreading to extract data from multiple files concurrently (or try with parrallel processing)
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     futures = [executor.submit(self._extract_single_file, file_info, variables, coordinates)
        #                for file_info in file_list]
        #     datasets = [future.result() for future in concurrent.futures.as_completed(futures)]

        # For now, no multithreading
        datasets = [self._extract_single_file(file_info, variables, coordinates) for file_info in file_list]

        # Combine all datasets that are seperated by time
        combined_dataset = xr.concat(datasets, dim='time')

        return combined_dataset

    def _get_azure_file_list(self, time_list: List[datetime]) -> List[Tuple[str, int, int]]:
        """
        Generate a list of Azure Blob Storage file paths and cycle/forecast hour info for the given list of times.

        :param time_list: List of datetime objects for which to generate file paths
        :return: List of tuples containing (file_path, cycle, forecast_hour)
        """
        file_list = []
        for time in time_list:
            date = time.date()
            hour = time.hour

            # Determine the most recent cycle
            if hour % 3 == 0:
                cycle = hour
            else:
                cycle = (hour // 3) * 3  # Round down to nearest 3-hour cycle

            # define the forecast hour as zero to get the analysis
            forecast_hour = 0

            file_path = f"{self.blob_container}/hrrr.{date:%Y%m%d}/{self.sector}/"
            file_path += f"hrrr.t{cycle:02d}z.{self.product}{forecast_hour:02d}.grib2"
            file_list.append((file_path, cycle, forecast_hour))

        # todo : implement eventually to integrate forecasts, but for now just analysis
        return file_list

    def _extract_single_file(self, file_info: Tuple[str, int, int], variables: List[str],
                            coordinates: Union[str, gpd.GeoDataFrame, pd.DataFrame, Dict[str, Sequence[float]]] = None) -> xr.Dataset:
        """
        Extract data from a single HRRR file.

        :param file_info: Tuple containing (file_path, cycle, forecast_hour)
        :param variables: List of variables to extract
        :param coordinates: Coordinates for subsetting. Can be one of:
                            - Path to a GeoJSON/Shapefile
                            - GeoDataFrame
                            - DataFrame with 'longitude' and 'latitude' columns
                            - Dictionary with 'lon' and 'lat' keys, each containing a sequence of floats
        :return: xarray Dataset containing extracted data
        """
        file_path, cycle, forecast_hour = file_info

        # Fetch the idx file
        idx_url = f"{file_path}.idx"
        r = requests.get(idx_url)
        idx = r.text.splitlines()

        # Extract byte ranges for requested variables
        byte_ranges = self._get_byte_ranges(idx, variables)

        # Fetch data for each variable
        datasets = []
        for variable, byte_range in byte_ranges.items():
            headers = {"Range": f"bytes={byte_range}"}
            resp = requests.get(file_path, headers=headers, stream=True)

            # Handles the temporary storage of fetched data. Creates a temporary file with a unique name (with prefixe tmp_ - avoids risk of file name collissions, useful for multithreading).
            # Doing this ensures that fetched data is stored securely and can be accessed later for further processing (like opening with xarray which is what we need)
            with tempfile.NamedTemporaryFile(prefix="tmp_", delete=False) as temp_file:
                temp_file.write(resp.content)
                self.logger.info(f"Temporary file created and written for variable '{variable}' with cycle {cycle} and forecast hour {forecast_hour}: {temp_file.name}")
            # Open the temporary file with xarray.
            # Setting idexpath to empty string tells cfgrib to open with out an associated .idx file (doesnt exist since just created a subfile of the file)
            ds = xr.open_dataset(temp_file.name, engine='cfgrib', backend_kwargs={'indexpath': ''},chunks='auto')
            datasets.append(ds)

        # Merge datasets for all variables
        merged_ds = xr.merge(datasets)

        # rename latitude and longitude coordinates to lat and lon (required for clisops)
        merged_ds = merged_ds.rename({'latitude':'lat','longitude':'lon'})

        # change longitude coordinate of merged_ds from 0-360 to -180-180
        merged_ds = merged_ds.assign_coords(lon=(merged_ds.lon + 180) % 360 - 180)

        # subset the data to Quebec area only
        merged_ds = subset.subset_shape(merged_ds, AREA_REDUCER_PATH)

        # Apply subsetting if coordinates are provided (only works with subset_gridpoint which requires geographic dataset)
        if coordinates is not None:
            merged_ds = self._subset_data(merged_ds, coordinates)

        return merged_ds

    def _subset_data(self, dataset: xr.Dataset,
                     coordinates: Union[str, gpd.GeoDataFrame, pd.DataFrame, Dict[str, Sequence[float]]]) -> xr.Dataset:
        """
        Subset the dataset based on the provided coordinates.

        :param dataset: xarray Dataset to subset
        :param coordinates: Coordinates for subsetting
        :return: Subsetted xarray Dataset
        """
        if isinstance(coordinates, str):
            # Load GeoJSON/Shapefile
            gdf = gpd.read_file(coordinates)
            lons, lats = gdf.geometry.x.tolist(), gdf.geometry.y.tolist()
        elif isinstance(coordinates, gpd.GeoDataFrame):
            lons, lats = coordinates.geometry.x.tolist(), coordinates.geometry.y.tolist()
        elif isinstance(coordinates, pd.DataFrame):
            if 'longitude' in coordinates.columns and 'latitude' in coordinates.columns:
                lons, lats = coordinates['longitude'].tolist(), coordinates['latitude'].tolist()
            else:
                raise ValueError("DataFrame must have 'longitude' and 'latitude' columns")
        elif isinstance(coordinates, dict):
            if 'lon' in coordinates and 'lat' in coordinates:
                lons, lats = coordinates['lon'], coordinates['lat']
            else:
                raise ValueError("Dictionary must have 'lon' and 'lat' keys")
        else:
            raise ValueError("Unsupported coordinate type")

        # Use clisops to subset the data
        subsetted_ds = subset.subset_gridpoint(dataset, lon=lons, lat=lats)
        return subsetted_ds

    def _get_byte_ranges(self, idx: List[str], variables: List[str]) -> Dict[str, str]:
        """
        Get byte ranges for requested variables from the idx file content.

        :param idx: List of strings representing idx file content
        :param variables: List of variables to extract
        :return: Dictionary of variable names and their corresponding byte ranges
        """
        byte_ranges = {}
        for variable in variables:
            var_idx = [l for l in idx if f":{variable}:" in l][0].split(":")
            line_num = int(var_idx[0])
            range_start = var_idx[1]

            next_line = idx[line_num].split(':') if line_num < len(idx) else None
            range_end = next_line[1] if next_line else None

            byte_ranges[variable] = f"{range_start}-{range_end}"

        return byte_ranges

    def save_to_netcdf(self, dataset: xr.Dataset, output_path: str):
        """
        Save the xarray Dataset to a NetCDF file.

        :param dataset: xarray Dataset to save
        :param output_path: Path to save the NetCDF file
        """
        dataset.to_netcdf(output_path)
        self.logger.info(f"Dataset successfully saved to {output_path}")

    def save_to_csv(self, dataset: xr.Dataset, output_path: str):
        """
        Save the xarray Dataset to a CSV file using pandas.

        :param dataset: xarray Dataset to save
        :param output_path: Path to save the CSV file
        """
        # Ensure the directory exists
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"\n Created directory: {output_dir}")

        # Save the dataset to CSV
        df = dataset.to_dataframe()
        df.to_csv(output_path)
        self.logger.info(f"HRRR CSV successfully saved to {output_path}")

# Main execution ---------------------------------------
def main():
    extractor = HRRRDataExtractor()

    # Example usage with a list of times
    time_list = [
        datetime(2024, 1, 1, 0, 0),
        datetime(2024, 1, 1, 6, 0),
        datetime(2024, 1, 2, 12, 0),
        datetime(2024, 1, 3, 18, 0),
    ]

    variables = ["SNOD","SNOWC"]  # variables are for "snow_height" and "snow_cover"

    # Use IRDA as an example
    locations = {
        "IRDA Saint-Bruno": (45.5522,-73.3506),
        "IRDA Québec": (46.792552,-71.310007)}
    # Create a GeoDataFrame for the locations
    gdf = gpd.GeoDataFrame(
        locations.keys(),
        geometry=[Point(lon, lat) for lat, lon in locations.values()],
        crs="EPSG:4326"
    )

    data = extractor.get_hrrr_data(time_list, variables, gdf)

    # extractor.save_to_netcdf(data, "output.nc") # to use if saving the entire netcdf
    extractor.save_to_csv(data, "C:\\temp_test\\output.csv") # to use if saving only certain coordinates

if __name__ == "__main__":
    main()
