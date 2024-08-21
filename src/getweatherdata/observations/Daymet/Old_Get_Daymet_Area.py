"""
File: Old_Get_Daymet_Area.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description: This is a description of what the script does
Created: 2024-04-11
"""
import urllib

STARTYEAR = 2010
ENDYEAR = 2020
# NO_NAME = "NULL"
# YEAR_LINE = "years:"
# VAR_LINE  = "variables:"
DAYMET_VARIABLES = ['tmax']  # Not sure if tmean could work
# DAYMET_VARIABLES = ['tmin', 'prcp']
# DAYMET_VARIABLES = ['tmean'] # N
DAYMET_YEARS = [str(year) for year in range(STARTYEAR, ENDYEAR + 1)]
north = 55
south = 44
east = -64
west = -80
s_stride = 1  # default is 1 to get all data
t_stride = 1  # default is 1 to get all data
format = "netcdf"

# %% Daily data
OutPath = "D:\\observations\\Daymet\\Daily\\"
Project_Name = ''  # if adding a project name, must add an _ at the beggining
for DAYMETVAR in DAYMET_VARIABLES:
    for YEAR in DAYMET_YEARS:
        print(f"Requesting daily data for {DAYMETVAR} ; {YEAR}")
        time_start = f"{YEAR}-01-01T12%3A00%3A00Z"  # Not sure what that last part is
        time_end = f"{YEAR}-12-31T12%3A00%3A00Z"

        # Update paths
        DAYMET_BASE_URL = f"https://thredds.daac.ornl.gov/thredds/ncss/ornldaac/2129/"
        DAYMET_VERSION_STR = f"daymet_v4_daily_na_{DAYMETVAR}_{YEAR}.nc?var=lat&var=lon&"
        DAYMET_VAR_AREA_STR = f"var={DAYMETVAR}&north={north}&west={west}&east={east}&south={south}" \
                              f"&disableProjSubset=on&horizStride={s_stride}&"
        DAYMET_TIME_STR = f"time_start={time_start}&time_end={time_end}&timeStride={t_stride}&"
        DAYMET_FULL_PTH = f"{DAYMET_BASE_URL}{DAYMET_VERSION_STR}{DAYMET_VAR_AREA_STR}{DAYMET_TIME_STR}accept={format}"
        OutName = f'DAYMET{Project_Name}_{DAYMETVAR}_{YEAR}.nc'
        print(f"Saving to :", OutPath + OutName)
        urllib.request.urlretrieve(DAYMET_FULL_PTH, OutPath + OutName)
