import shutil
import tempfile
import xarray as xr
import urllib.request
import pandas as pd

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