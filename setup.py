#!/usr/bin/env
"""
Creation date: 2023-08-02
Creator : the_l
Python version : 3.10

Description:

Notes:
"""

from setuptools import setup, find_packages

setup(
    name='GetWeatherData',
    version='0.3',
    packages=find_packages(),
    install_requires=["xarray >= 2023.7.0","xclim >= 0.44.0", "siphon >= 0.9", "pystac", "fsspec", "clisops", "zarr"],
    python_requires=">=3.9.0",
    package_data={
        "":["*.json"]
    }
)