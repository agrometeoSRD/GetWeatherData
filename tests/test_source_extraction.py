#!/usr/bin/env
"""
Creation date: 2023-08-02
Creator : the_l
Python version : 3.10

Description:

Notes:
"""

# from source.observations.NRCAN import get_nrcan
from source import get_nrcan

def test_get_data():
    ds = get_nrcan.get_data()
    return ds

test_ds = test_get_data()
