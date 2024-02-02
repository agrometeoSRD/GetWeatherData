import shutil
import tempfile
import xarray as xr
import urllib.request
import pandas as pd

#%% Trying out wth sarracenia
import sarracenia



#%% Code that doesn't work
def parse():
    # Pull out the different attributes in the request object
    url = request.json['url']
    attribute = request.json['attribute']
    queries = request.json['queries']

    # Copy the HRDPS file to a generated temporary file
    with tempfile.NamedTemporaryFile() as tmpFile:
        with urllib.request.urlopen(request.json['url']) as response:
            shutil.copyfileobj(response, tmpFile)
            tmpFile.flush()

        # Using Xarray and the cgfrib engine, open the HRDPS file and transform it to a 2D pandas dataframe
        with xr.open_dataset(tmpFile.name, engine='cfgrib',
                             backend_kwargs={'indexpath': ''}) as ds:
            hrdps = ds.to_dataframe()

        time = hrdps['time'].iloc[0]
        step = hrdps['step'].iloc[0]
        model_date = time.isoformat() + 'Z'
        forecast_date = (time + step).isoformat() + 'Z'

        # Query the dataset with a function that takes in the HRDPS data table, the lat & long, and the weather attribute specific to the file
        results = query_hrdps(hrdps, queries, attribute)

    # Return a response object
    return {
        'url': url,
        'forecastDate': forecast_date,
        'modelDate': model_date,
        'attribute': attribute,
        'results': results
    }


import requests

def download_data(api_url, headers=None, params=None):
    response = requests.get(api_url, headers=headers, params=params)

    if response.status_code == 200:
        return response.json()  # or response.text if the data is not in json format
    else:
        print(f"Failed to download data. Status code: {response.status_code}")
        return None

# Replace with your actual API URL, headers, and parameters
api_url = "https://example.com/api"
headers = {"Authorization": "Bearer YOUR_ACCESS_TOKEN"}
params = {"param1": "value1", "param2": "value2"}

data = download_data(api_url, headers, params)

