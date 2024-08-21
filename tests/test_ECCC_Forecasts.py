"""
File: test_ECCC_Forecasts.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description: This is a description of what the script does
Created: 2024-01-30

NOTE : THIS WAS A TEST FILE TO UNDERSTAND HOW TO WRITE TESTS IN PYTHON. NOT ACTUALLY WORTH USING
- It actually doesn't work because : we don't know how the dataframe of combine_past_and_current will look like. The output is non-deterministic.
- The below function expects hard-coded output of the function, which is not the case.
"""

# Import statements
import unittest
from source.Forecasts.ec_forecasts import combine_past_and_current_forecast
import pandas as pd
from pandas.testing import assert_frame_equal

# Constants

# Functions
# TODO : Learn about what is test coverage and how to use it
#TODO : Test the function combine_past_and_current_forecast
class TestCombinePastAndCurrentForecast(unittest.TestCase):
    def setUp(self):
        self.date_col = 'Date'
        self.forecast_variables = ["AIRTEMP [C]", "HR [%]", "RAIN [mm]", "GLOBALRAD [Wm2]"]
        self.past_df = pd.DataFrame({
            self.date_col: pd.date_range(start='1/1/2022', periods=3),
            self.forecast_variables[0]: [1, 2, 3],
            self.forecast_variables[1]: [4, 5, 6],
            self.forecast_variables[2]: [7, 8, 9],
            self.forecast_variables[3]: [10, 11, 12]
        })
        self.current_df = pd.DataFrame({
            self.date_col: pd.date_range(start='1/2/2022', periods=3),
            self.forecast_variables[0]: [13, 14, 15],
            self.forecast_variables[1]: [16, 17, 18],
            self.forecast_variables[2]: [19, 20, 21],
            self.forecast_variables[3]: [22, 23, 24]
        })

    def test_combine_past_and_current_forecast_happy_path(self):
        expected_df = pd.DataFrame({
            self.date_col: pd.date_range(start='1/1/2022', periods=4),
            self.forecast_variables[0]: [1, 13, 14, 15],
            self.forecast_variables[1]: [4, 16, 17, 18],
            self.forecast_variables[2]: [7, 0, 20, 21],
            self.forecast_variables[3]: [10, 22, 23, 24]
        })
        result_df = combine_past_and_current_forecast(self.past_df, self.current_df, self.date_col)
        assert_frame_equal(result_df, expected_df)

    def test_combine_past_and_current_forecast_when_past_df_is_empty(self):
        empty_df = pd.DataFrame()
        result_df = combine_past_and_current_forecast(empty_df, self.current_df, self.date_col)
        assert_frame_equal(result_df, self.current_df)

    def test_combine_past_and_current_forecast_when_current_df_is_empty(self):
        empty_df = pd.DataFrame()
        result_df = combine_past_and_current_forecast(self.past_df, empty_df, self.date_col)
        assert_frame_equal(result_df, self.past_df)

    def test_combine_past_and_current_forecast_when_both_dfs_are_empty(self):
        empty_df = pd.DataFrame()
        expected_df = pd.DataFrame(columns=[self.date_col] + self.forecast_variables)
        result_df = combine_past_and_current_forecast(empty_df, empty_df, self.date_col)
        assert_frame_equal(result_df, expected_df)

    def test_combine_past_and_current_forecast_when_dfs_have_different_columns(self):
        different_df = pd.DataFrame({
            self.date_col: pd.date_range(start='1/2/2022', periods=3),
            'Different_Column': [25, 26, 27]
        })
        result_df = combine_past_and_current_forecast(self.past_df, different_df, self.date_col)
        assert_frame_equal(result_df, different_df)

    def test_combine_past_and_current_forecast_when_no_overlapping_dates(self):
        pass

    def test_for_nulls(self):
        # edge case for when there is null value in the data
        pass

# Main execution ---------------------------------------

if __name__ == "__main__":
    unittest.main() # This will run all the tests when script is executed
