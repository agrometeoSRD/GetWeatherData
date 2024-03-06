import shutil
import tempfile
import xarray as xr
import urllib.request
import pandas as pd

#%% Trying out wth sarracenia
import sarracenia
import sarracenia.moth
import sarracenia.credentials
from sarracenia.config import default_config

import os
import time
import socket

cfg = default_config()
cfg.logLevel = 'debug'
cfg.broker = sarracenia.credentials.Credential('amqp://tfeed:password@localhost')
cfg.exchange = 'xpublic'
cfg.post_baseUrl = 'http://host'
cfg.post_baseDir = '/tmp'

posting_engine = sarracenia.moth.Moth.pubFactory( cfg.dictify() )

# create a file?
sample_fileName = '/tmp/sample.txt'
sample_file = open( sample_fileName , 'w')
sample_file.write(
"""
CACN00 CWAO 161800
PMN
160,2021,228,1800,1065,100,-6999,20.49,43.63,16.87,16.64,323.5,9.32,27.31,1740,317.8,19.22,1.609,230.7,230.7,230.7,230.7,0,0,0,16.38,15.59,305.
9,17.8,16.38,19.35,55.66,15.23,14.59,304,16.67,3.844,20.51,18.16,0,0,-6999,-6999,-6999,-6999,-6999,-6999,-6999,-6999,0,0,0,0,0,0,0,0,0,0,0,0,0,
13.41,13.85,27.07,3473
"""
)
sample_file.close()

# supply msg init the to file
# you can supply msg_init with your files, it will build a message appropriate for it.
m = sarracenia.Message.fromFileData(sample_fileName, cfg, os.stat(sample_fileName) )
# here is the resulting message.
print(m)

# feed the message to the posting engine.
posting_engine.putNewMessage(m)

# when done, should close... cleaner...
posting_engine.close()

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

