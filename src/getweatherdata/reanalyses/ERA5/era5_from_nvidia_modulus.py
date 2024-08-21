"""
Author : Nvidia
File: era5_from_nvidia_modulus.py
Python : 3.11
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description :
- This is a lightly modified version of the scripts provided by Nvidia in their Modulus repository to download era5 data
- This script is used to download ERA5 data from the Copernicus Climate Data Store (CDS) and store it in Zarr format.
- Tutorial : https://docs.nvidia.com/deeplearning/modulus/modulus-core-v040/examples/weather/dataset_download/readme.html

Some definitions
- Mirroring the data = creating an exact copy from the source to a destination. Usually different from
downloading the data, because we maintain the structure, format and updates of the source data in the destination
- Zarr = A Python package providing an implementation of chunked, compressed, N-dimensional arrays.
Zarr is designed for use in parallel computing, with a particular focus on dask, a parallel computing library in Python.

Rights of use :
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

Created: 2024-08-14
"""


import os
import tempfile
import cdsapi
import xarray as xr
import datetime
import json
import dask
import calendar
from dask.diagnostics import ProgressBar
from typing import List, Tuple, Dict, Union
import urllib3
import logging
import numpy as np
import fsspec
import hydra
from omegaconf import DictConfig


urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class ERA5Mirror:
    """
    A class to manage downloading ERA5 datasets. The datasets are downloaded from the Copernicus Climate Data Store (CDS) and stored in Zarr format.

    Attributes
    ----------
    base_path : Path
        The path to the Zarr dataset.
    fs : fsspec.AbstractFileSystem
        The filesystem to use for the Zarr dataset. If None, the local filesystem will be used.
    """

    # initialize the class with a base path for storing the Zarr dataset. Optional filesystem object
    def __init__(self, base_path: str, fs: fsspec.AbstractFileSystem = None):
        # Get parameters
        self.base_path = base_path
        if fs is None: # if filesystem is not provided, default to local filesystem
            fs = fsspec.filesystem("file")
        self.fs = fs

        # Create the base path if it doesn't exist
        if not self.fs.exists(self.base_path):
            self.fs.makedirs(self.base_path)

        # Create metadata that will be used to track which chunks have been downloaded
        self.metadata_file = os.path.join(self.base_path, "metadata.json")
        self.metadata = self.get_metadata()

    # read metadata from a json file if it exists, or initialize an empty dictionary
    # This json file should contain : TBD
    def get_metadata(self):
        """Get metadata"""
        if self.fs.exists(self.metadata_file):
            with self.fs.open(self.metadata_file, "r") as f:
                try:
                    metadata = json.load(f)
                except json.decoder.JSONDecodeError:
                    metadata = {"chunks": []}
        else:
            metadata = {"chunks": []}
        return metadata

    def save_metadata(self):
        """Save metadata"""
        with self.fs.open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f)

    # important to check if chunk exist because... TBD
    def chunk_exists(self, variable, year, month, hours, pressure_level):
        """Check if chunk exists"""
        for chunk in self.metadata["chunks"]:
            if (
                chunk["variable"] == variable
                and chunk["year"] == year
                and chunk["month"] == month
                and chunk["hours"] == hours
                and chunk["pressure_level"] == pressure_level
            ):
                return True
        return False

    def download_chunk(
        self,
        variable: str,
        year: int,
        month: int,
        hours: List[int],
        pressure_level: int = None,
    ):
        """
        Download ERA5 data for the specified variable, date range, hours, and pressure levels.
        Constructs a temporary file to store the downloaded data and uses the cdsapi client to retrieve the data

        Parameters
        ----------
        variable : str
            The ERA5 variable to download, e.g. 'tisr' for solar radiation or 'z' for geopotential.
        year : int
            The year to download.
        month : int
            The month to download.
        hours : List[int]
            A list of hours (0-23) for which data should be downloaded.
        pressure_level : int, optional
            A pressure level to include in the download, by default None. If None, the single-level data will be downloaded.

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the downloaded data.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            # Get all days in the month
            days_in_month = calendar.monthrange(year, month)[1]

            # Make tmpfile to store the data
            output_file = os.path.join(
                tmpdir,
                f"{variable}_{year}_{month:02d}_{str(hours)}_{str(pressure_level)}.nc",
            )

            # start the CDS API client (maybe need to move this outside the loop?)
            c = cdsapi.Client(quiet=True)

            # Setup the request parameters
            request_params = {
                "product_type": "reanalysis",
                "variable": variable,
                "year": str(year),
                "month": str(month),
                "day": [f"{day:02d}" for day in range(1, days_in_month + 1)],
                "time": [f"{hour:02d}:00" for hour in hours],
                "format": "netcdf",
            }
            if pressure_level:
                request_params["pressure_level"] = [str(pressure_level)]
                dataset_name = "reanalysis-era5-pressure-levels"
            else:
                dataset_name = "reanalysis-era5-single-levels"

            # Download the data
            c.retrieve(
                dataset_name,
                request_params,
                output_file,
            )

            # Open the downloaded data
            ds = xr.open_dataset(output_file)
        return ds

    def variable_to_zarr_name(self, variable: str, pressure_level: int = None):
        """convert variable to zarr name"""
        # create zarr path for variable
        zarr_path = f"{self.base_path}/{variable}"
        if pressure_level:
            zarr_path += f"_pressure_level_{pressure_level}"
        zarr_path += ".zarr"
        return zarr_path

    def download_and_upload_chunk(
        self,
        variable: str,
        year: int,
        month: int,
        hours: List[int],
        pressure_level: int = None,
    ):
        """
        Downloads a chunk of ERA5 data for a specific variable and date range, and uploads it to a Zarr array.
        This downloads a 1-month chunk of data.

        Parameters
        ----------
        variable : str
            The variable to download.
        year : int
            The year to download.
        month : int
            The month to download.
        hours : List[int]
            A list of hours to download.
        pressure_level : int, optional
            Pressure levels to download, if applicable.
        """

        # Download the data
        ds = self.download_chunk(variable, year, month, hours, pressure_level)

        # Create the Zarr path
        zarr_path = self.variable_to_zarr_name(variable, pressure_level)

        # Specify the chunking options
        chunking = {"time": 1, "latitude": 721, "longitude": 1440}
        if "level" in ds.dims:
            chunking["level"] = 1

        # Re-chunk the dataset
        ds = ds.chunk(chunking)

        # Check if the Zarr dataset exists
        if self.fs.exists(zarr_path):
            mode = "a"
            append_dim = "time"
            create = False
        else:
            mode = "w"
            append_dim = None
            create = True

        # Upload the data to the Zarr dataset
        mapper = self.fs.get_mapper(zarr_path, create=create)
        ds.to_zarr(mapper, mode=mode, consolidated=True, append_dim=append_dim)

        # Update the metadata
        self.metadata["chunks"].append(
            {
                "variable": variable,
                "year": year,
                "month": month,
                "hours": hours,
                "pressure_level": pressure_level,
            }
        )
        self.save_metadata()

    def download(
        self,
        variables: List[Union[str, Tuple[str, int]]],
        date_range: Tuple[datetime.date, datetime.date],
        hours: List[int],
    ):
        """
        Start the process of mirroring the specified ERA5 variables for the given date range and hours.
        In other words, download multiple variables over specified date range and hours.
        Reformats the variables list, rounds the dates to months and creates tasks to downlaod the data using dask
        Ensures that the zarr arrays have correct time dimension

        Parameters
        ----------
        variables : List[Union[str, Tuple[str, List[int]]]]
            A list of variables to mirror, where each element can either be a string (single-level variable)
            or a tuple (variable with pressure level).
        date_range : Tuple[datetime.date, datetime.date]
            A tuple containing the start and end dates for the data to be mirrored. This will download and store every month in the range.
        hours : List[int]
            A list of hours for which to download the data.

        Returns
        -------
        zarr_paths : List[str]
            A list of Zarr paths for each of the variables.
        """

        start_date, end_date = date_range

        # Reformat the variables list so all elements are tuples
        reformated_variables = []
        for variable in variables:
            if isinstance(variable, str):
                reformated_variables.append(tuple([variable, None]))
            else:
                reformated_variables.append(variable)

        # Start Downloading
        with ProgressBar():
            # Round dates to months
            current_date = start_date.replace(day=1)
            end_date = end_date.replace(day=1)

            while current_date <= end_date:
                # Create a list of tasks to download the data
                tasks = []
                for variable, pressure_level in reformated_variables:
                    if not self.chunk_exists(
                        variable,
                        current_date.year,
                        current_date.month,
                        hours,
                        pressure_level,
                    ):
                        task = dask.delayed(self.download_and_upload_chunk)(
                            variable,
                            current_date.year,
                            current_date.month,
                            hours,
                            pressure_level,
                        )
                        tasks.append(task)
                    else:
                        print(
                            f"Chunk for {variable} {pressure_level} {current_date.year}-{current_date.month} already exists. Skipping."
                        )

                # Execute the tasks with Dask
                print(f"Downloading data for {current_date.year}-{current_date.month}")
                if tasks:
                    dask.compute(*tasks)

                # Update the metadata
                self.save_metadata()

                # Update the current date
                days_in_month = calendar.monthrange(
                    year=current_date.year, month=current_date.month
                )[1]
                current_date += datetime.timedelta(days=days_in_month)

        # Return the Zarr paths
        zarr_paths = []
        for variable, pressure_level in reformated_variables:
            zarr_path = self.variable_to_zarr_name(variable, pressure_level)
            zarr_paths.append(zarr_path)

        # Check that Zarr arrays have correct dt for time dimension
        for zarr_path in zarr_paths:
            ds = xr.open_zarr(zarr_path)
            time_stamps = ds.time.values
            dt = time_stamps[1:] - time_stamps[:-1]
            assert np.all(
                dt == dt[0]
            ), f"Zarr array {zarr_path} has incorrect dt for time dimension. An error may have occurred during download. Please delete the Zarr array and try again."

        return zarr_paths

###########################################################
# Trying to run this could equal to doing the following

# hydra.main allows the main to be configurable using a yaml configuration file
# the "conf" folder must be in the same directory as the python script.
# Inside the directory, a file name config_tas.yaml must exist
@hydra.main(version_base="1.2", config_path="conf", config_name="config_tas")
def main(cfg: DictConfig) -> None:
    # Make mirror data
    logging.getLogger().setLevel(logging.ERROR)  # Suppress logging from cdsapi
    # Initialize an instance of the class with base path for storing the zarr dataset
    mirror = ERA5Mirror(base_path=cfg.zarr_store_path)

    # split the years into train, validation, and test
    train_years = list(range(cfg.start_train_year, cfg.end_train_year + 1))
    test_years = cfg.test_years
    out_of_sample_years = cfg.out_of_sample_years
    all_years = train_years + test_years + out_of_sample_years

    # Set date range and hours that will be downloaded
    # Set the variables to download for 34 var dataset
    date_range = (
        datetime.date(min(all_years), 1, 1),
        datetime.date(max(all_years), 12, 31),
    )
    hours = [cfg.dt * i for i in range(0, 24 // cfg.dt)]

    # Start the mirror (download the data)
    zarr_paths = mirror.download(cfg.variables, date_range, hours)

    # Open the zarr files and construct the xarray from them
    zarr_arrays = [xr.open_zarr(path) for path in zarr_paths]
    era5_xarray = xr.concat(
        [z[list(z.data_vars.keys())[0]] for z in zarr_arrays], dim="channel"
    )
    era5_xarray = era5_xarray.transpose("time", "channel", "latitude", "longitude")
    era5_xarray.name = "fields"
    era5_xarray = era5_xarray.astype("float32")

    # Depending on the configuration file, global mean and std are computed and saved
    # Save mean and std
    if cfg.compute_mean_std:
        stats_path = os.path.join(cfg.hdf5_store_path, "stats")
        print(f"Saving global mean and std at {stats_path}")
        if not os.path.exists(stats_path):
            os.makedirs(stats_path)
        era5_mean = np.array(
            era5_xarray.mean(dim=("time", "latitude", "longitude")).values
        )
        np.save(
            os.path.join(stats_path, "global_means.npy"), era5_mean.reshape(1, -1, 1, 1)
        )
        era5_std = np.array(
            era5_xarray.std(dim=("time", "latitude", "longitude")).values
        )
        np.save(
            os.path.join(stats_path, "global_stds.npy"), era5_std.reshape(1, -1, 1, 1)
        )
        print(f"Finished saving global mean and std at {stats_path}")

    # iterate over all years and save the data to hdf5 files
    # Make hdf5 files
    for year in all_years:
        # HDF5 filename
        split = (
            "train"
            if year in train_years
            else "test"
            if year in test_years
            else "out_of_sample"
        )
        hdf5_path = os.path.join(cfg.hdf5_store_path, split)
        os.makedirs(hdf5_path, exist_ok=True)
        hdf5_path = os.path.join(hdf5_path, f"{year}.h5")

        # Save year using dask
        print(f"Saving {year} at {hdf5_path}")
        with dask.config.set(
            scheduler="threads",
            num_workers=8,
            threads_per_worker=2,
            **{"array.slicing.split_large_chunks": False},
        ):
            with ProgressBar():
                # Get data for the current year
                year_data = era5_xarray.sel(time=era5_xarray.time.dt.year == year)

                # Save data to a temporary local file
                year_data.to_netcdf(hdf5_path, engine="h5netcdf", compute=True)
        print(f"Finished Saving {year} at {hdf5_path}")


if __name__ == "__main__":
    main()
