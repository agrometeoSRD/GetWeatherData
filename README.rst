Welcome to the GetWeatherData project! This project is designed to fetch, process, and analyze weather data from various sources. Below, you will find detailed instructions on how to use the different scripts within the project, especially focusing on ``get_SM_data.py``, ``noaa_forecast.py``, and ``ec_forecasts.py``.

.. contents::
   :local:
   :depth: 1

Project Structure
=================

.. code-block:: text

    GetWeatherData/
    ├── src/
    │   ├── getweatherdata/
    │   │   ├── observations/
    │   │   │   ├── Stations/
    │   │   │   │   └── get_SM_data.py
    │   │   │   ├── NRCAN/
    │   │   │   │   └── get_nrcan.py
    │   │   ├── reanalyses/
    │   │   │   ├── ERA5/
    │   │   │   │   └── era5_from_pavics.py
    │   │   │   ├── RDRS/
    │   │   │   │   └── rdrs_from_pavics.py
    │   ├── forecasts/
    │   │   ├── noaa_forecast.py
    │   │   ├── ec_forecasts.py
    │   ├── daymet/
    │   │   ├── Get_Daymet_Area.py
    │   │   ├── Get_Daymet_SinglePixel.py
    │   ├── utils/
    │   │   ├── combine_station_hrdps.py
    │   │   ├── patching_wu_with_bru.py
    │   │   ├── to_rimpro.py
    ├── README.md
    ├── requirements.txt
    └── setup.py

Installation
============

1. Clone the repository:

.. code-block:: sh

    git clone https://github.com/yourusername/GetWeatherData.git
    cd GetWeatherData

2. Install the required dependencies:

.. code-block:: sh

    pip install -r requirements.txt

What works well enough
=====
.. code-block:: sh

    ec_forecasts.py
    save_ec_nowcast.py
    get_SM_data.py
    noaa_forecast.py
    era5_from_pavics.py
    Get_Daymet_Area.py
    Get_Daymet_SinglePixel.py
    get_nrcan.py
    rdrs_from_pavics.py
    get_WU_data
    combine_station_hrdps.py (very specific use)
    patching_wu_with_bru.py (very specific use)
    to_rimpro.py (very speficic use)

Usage
=====

get_SM_data.py
--------------

This script is used to download and process weather station data from the specified sources.

**Command-line Arguments:**

* ``--stations``: List of station names (default: ``['Compton', 'Dunham']``)
* ``--years``: List of years (default: ``['2020', '2021']``)
* ``--save_path``: Path to save the output CSV (default: ``'./'``)
* ``--filename``: Filename for the output CSV (no extension) (default: ``'Compton_station'``)

**Example:**

.. code-block:: sh

    python src/getweatherdata/observations/Stations/get_SM_data.py --stations Compton Dunham --years 2020 2021 --save_path ./data --filename weather_data

noaa_forecast.py
----------------

This script fetches and processes weather forecast data from NOAA.

**Command-line Arguments:**

* ``--location``: Location for which to fetch the forecast (default: ``'New York'``)
* ``--days``: Number of days for the forecast (default: ``7``)
* ``--save_path``: Path to save the output CSV (default: ``'./'``)
* ``--filename``: Filename for the output CSV (no extension) (default: ``'noaa_forecast'``)

**Example:**

.. code-block:: sh

    python src/forecasts/noaa_forecast.py --location "New York" --days 7 --save_path ./data --filename noaa_forecast

ec_forecasts.py
---------------

This script fetches and processes weather forecast data from Environment Canada.

**Command-line Arguments:**

* ``--location``: Location for which to fetch the forecast (default: ``'Toronto'``)
* ``--days``: Number of days for the forecast (default: ``7``)
* ``--save_path``: Path to save the output CSV (default: ``'./'``)
* ``--filename``: Filename for the output CSV (no extension) (default: ``'ec_forecast'``)

**Example:**

.. code-block:: sh

    python src/forecasts/ec_forecasts.py --location "Toronto" --days 7 --save_path ./data --filename ec_forecast

Configuration
=============

The configuration for the scripts can be found in the ``config`` folder. Each script has its own configuration file, which can be modified to suit your needs.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
