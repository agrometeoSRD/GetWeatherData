import unittest
from unittest.mock import patch, mock_open
import source.Observations.Daymet.Get_Daymet_Area as Get_Daymet_Area  # replace 'your_module' with the name of your Python script/module

class TestDownloadDaymet(unittest.TestCase):


    #1 Test define_parameters function
    # Verify it returns correct default values when called without arguments. Check that it correctly handles custom start and end years
    def test_define_parameters(self):
        expected_default = {
                'daymet_variables': ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp'],
                'years': [str(year) for year in range(2020, 2022 + 1)],
                'north': 55,
                'south': 44,
                'east': -64,
                'west': -80,
                's_stride': 1,
                't_stride': 1,
                'format': "netcdf"
            }
        variables = ['dayl', 'prcp', 'srad', 'swe', 'tmax', 'tmin', 'vp']
        start_year = 2020
        end_year = 2022
        dimensions = {'north': 55, 'south': 44, 'east': -64, 'west': -80}

        self.assertEqual(Get_Daymet_Area.define_parameters(variables,start_year,end_year,dimensions),expected_default)

#2 Test create_url function
    def test_create_url(self):
        params = Get_Daymet_Area.define_parameters()
        url = Get_Daymet_Area.create_url('prcp', 2020, params)
        self.assertIn('2020',url)
        self.assertIn('prcp',url)


    @patch('builtins.open',new_callable=mock_open)
    @patch('json.dump')
    def test_save_config(self,mock_json_dump,mock_file):
        Get_Daymet_Area.save_config('test.json', {'test':'data'})
        mock_json_dump.assert_called_once()
        mock_file.assert_called_once()


#3 Test save_config and read_config functions

#4 test download_daymet function