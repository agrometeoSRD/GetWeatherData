"""
File: test_get_SM_data_v2.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description:
- Create a test class that inherits from unittest.TestCase

Created: 2024-02-23
"""

# Import statements
import unittest
import sys
from source.Observations.Stations.get_SM_data_v2 import main

# Constants

# Functions
# 2024-02-23 : current class is too advanced for me. Use something easier
# class TestArgparse(unittest.TestCase):
#     # Class can be used to initialize the parser from script
#     # Current example : testing years with test_years methods.
#     def setUp(self):
#         self.parser = main.parser
#
#     def test_years(self):
#         sys.argv = ['get_SM_data_v2.py', '--years', '2020', '2021']
#         args = self.parser.parse_args()
#         self.assertEqual(args.years, ['2020', '2021'])

# Main execution ---------------------------------------

if __name__ == '__main__':
    unittest.main()
