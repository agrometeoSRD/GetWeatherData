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
from unittest.mock import patch, mock_open
import sys
from source.Observations.Stations.get_SM_data import *

# Constants

# Functions
# 2024-02-23 : too complicated for me to understand mock loading. Will simply hard code the json
# def mock_json_load(open_mock):
#     config = {
#         "BRU_date_headers": ["Year", "Day", "Hour"],
#         "BRU_num_headers": ["Temp", "Humidity"]
#     }
#     open_mock.return_value.__enter__.return_value = config
#     return json.dumps(config)
# @patch("builtins.open", new_callable=mock_open, read_data=mock_json_load)
# @patch("json.load")

# 2024-02-23 : current class is too advanced for me. Use something easier
class TestArgparse(unittest.TestCase):
    # Class can be used to initialize the parser from script
    # Test is to check if the parser is working as expected
    def setUp(self):
        self.parser = main.parser_args([])

    def test_years(self):
        sys.argv = ['get_SM_data.py', '--years', '2020', '2021']
        args = self.parser.parse_args()
        self.assertEqual(args.years, ['2020', '2021'])

def test_download_and_process_data_known_output():
    # to do : test with known ouput, test with invalid inputs, test for partial data, test for data transformations
    # Given known inputs
    stations = ['Compton']
    years = ['2020']
    config = {
        "BRU_date_headers": ["Year","Day","Hour"],
        "BRU_num_headers": ["Num", "TMax", "TMin", "TMoy", "HR", "PR", "PR_start", "InSW", "Tair_5",
                         "TGr_5", "TGr_10", "TGr_20", "TGr_50", "Wind", "WindDir", "Pressure", "Mouillure_feuil", "Mouillure_feuil_thres",
                         "no_data", "no_data2"],
      "path_to_save": "C:\\Users\\sebastien.durocher\\PycharmProjects\\GetWeatherData\\source\\observations\\Stations"
    }
    # When calling the function
    result_df = download_and_process_data(stations,years,config)
    # Assert known outputs
    # => assert shape, specific values, column names, etc.
    assert not result_df.empty
    # assert specific value checks, e.g., result_df.loc[some_index, 'temp'] == expected_temp

# Main execution ---------------------------------------

if __name__ == '__main__':
    unittest.main()
