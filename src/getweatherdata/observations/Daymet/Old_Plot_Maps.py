#!/usr/bin/env
"""
Creation date: 2022-03-25
Creator : sebastien durocher 
Python version : ''

Description:
- Currently maps daymet netcdf
- Can add some point coordinates (currently ESSAQ data)

Updates:

Notes:

To-do:
"""

# imports
from netCDF4 import Dataset
from weather_params_standard import *
import matplotlib.pyplot as plt

from Plotting_Tools import *
from mapping import *
from weather_functions import *

class load_array():
    def __init__(self, file):
        self.file = file
        pass

    def check_file(self, band='0'):
        if 'hrdps' in self.file.lower():
            pass
        elif 'era' in self.file.lower():
            return self.compute_ERA(int(band))
        elif 'nrcan' in self.file.lower():
            return self.compute_NRCAN()
        elif 'daymet' in self.file.lower():
            return self.compute_DAYMET(band)

    def compute_ERA(self, band=1):  # era is a netcdf
        nc = Dataset(self.file, 'r', format="NETCDF3_CLASSIC")
        file_keys = list(nc.variables.keys())
        print(f"Key order for {self.file}:\n{file_keys}")
        if 'normal' in self.file.lower():
            data = nc.variables[file_keys[-1]][:]  # no bands for normals
        elif 'monthlymean' in self.file.lower():
            # IMPORTANT : NETCDF FILE DOES NOT KNOW THE YEARS,
            # IT JUST KNOWS HOW MANY DAYS HAVE PASSED SINCE THE FIRST YEAR
            # Currently have 19 years (0 : 2000 to 19:2018)
            print(f'Monthly mean : Acquiring the year corresponding to the band #{band}')
            data = nc.variables[file_keys[-2]][band, :, :]
        elif 'agseason' in self.file.lower():
            print(f'AgSeason : Acquiring the year corresponding to the band #{band}')
            data = nc.variables[file_keys[-1]][band, :, :]
        else:
            data = nc.variables[file_keys[-1]][:]

        lat = nc.variables['latitude'][:]
        lon = nc.variables['longitude'][:]

        return data, lat, lon

    def compute_NRCAN(self):  # NRCAN is geotiff
        # NRCAN = r"F:\Weather_Grid\NRCAN\Monthly\maxt\2016\test.tif"
        ds = gdal.Open(self.file)
        data = np.float32(ds.ReadAsArray())
        data[data == -9999] = np.nan  # Set all -999 to nans
        gt = ds.GetGeoTransform()
        # region Prepare data
        # Setup extent of map
        extent_map = Create_Extent(gt, ds)
        extent_map = extent_map.Boundary()
        return data, extent_map

    def compute_DAYMET(self, band='05'):  # Daymet is netcdf
        # File = "F:\\Weather_Grid\\Daymet\\Means\\DAYMET_tmin_Climate2015.nc"
        ds = xr.open_dataset(self.file)
        # When doing MonthlyNormals, rename each band into its appropriate month
        if 'normal' in self.file.lower():  # Band must be Band1, Band2, etc...
            ds = ds.rename_vars(
                {'Band1': '04', 'Band2': '05', 'Band3': '06', 'Band4': '07', 'Band5': '08', 'Band6': '09',
                 'Band7': '10'})
            sel_mth = band  # Use this for monthly normals or monthly climate
            print(f"Acquiring data for period of {sel_mth}")

        elif 'monthly' in self.file.lower():  # Band must be month
            ds = ds.to_array()[0]  # Cheating, but convert ds to array
            months = np.array(ds.time.dt.month)
            sel_mth = [idx for idx, val in enumerate(months) if int(band) == val][0]

        else:  # No Bands
            sel_mth = '05'  # Use this for AgSeason
            print(f"Acquiring data for period of {sel_mth}")
        data = np.array(ds[sel_mth])
        lat = ds[sel_mth].lat[:]
        lon = ds[sel_mth].lon[:]
        return data, lat, lon

def Create_array(file):
    ds = gdal.Open(file)
    data = np.float32(ds.ReadAsArray())

    data[data == -999] = np.nan  # Set all -999 to nans

    if 'ERA' in file:
        data[data == 0] = np.nan  # Only do this for homebrew masks (satellite, era)

    return ds, data

# region load data
# %%
# Load AF data
import pandas as pd

Workdir = r"C:\Users\sebastien durocher\OneDrive - IRDA\Nouveaux_Projets\Microbiome"
filename = "\data_820700_40sites_SRougerie.xlsx"
df = pd.read_excel(Workdir + filename)
coords = ["lat2","long2"]
coordinate_dict = dict(zip(coords, Standard_coordinate_format))
points =  df[coords].rename(columns=coordinate_dict).to_dict('list')
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[coords[1]], df[coords[0]]))
# gdf = gpd.read_file(Workdir + filename)
gdf = gdf.set_crs('epsg:4326')
# Plot with serie de sols
minx, miny, maxx, maxy = gdf.geometry.total_bounds
# extent = [minx-1,maxx+1,miny-0.5,maxy+0.5]
extent = [minx - 0.4, maxx + 0.4, miny - 0.3, maxy + 0.3]

# Load fadq munic coverage, munic boundaries, water boundaries
munic_gdf, water_gdf = load_basemaps()
min_all = 50
max_all = 150
var = 'pcp'  # pcp, maxt or mint
Type = 'Normal'  # Normal, AgSeason or Monthly
Year = 2014  # Only pertinent if using AgSeason or Monthly
Month = 5
names = extension(Type, var, Year, Month)

DAYMET_Path = 'E:\\Weather_Grid\\Daymet\\Means\\'
DAYMET_fname, DAYMET_band = names.DAYMET_ext()
DAYMET_file = f"{DAYMET_Path}{DAYMET_fname}"
# DAYMET_file = "F:\\Weather_Grid\\Daymet\\Means\\DAYMET_tmax_AgSeason2013.nc"
DAYMET = load_array(DAYMET_file)
DAYMET_data, DAYMET_lat, DAYMET_lon = DAYMET.check_file(band=DAYMET_band)
nc_plot(DAYMET_data,DAYMET_lat,DAYMET_lon,points,vmax=int(np.nanmax(DAYMET_data)),vmin=int(np.nanmin(DAYMET_data)),title='Normales des précipitations accumulées entre mai - octobre')
# nc_plot(gdf,DAYMET_data, DAYMET_lat, DAYMET_lon, points, vmin=min_all, vmax=max_all, dv=2, title='DAYMET', do_extent=0)

# # tif_plot(NRCAN_data,extent_map,vmin=int(np.nanmin(NRCAN_data)),vmax=int(np.nanmax(NRCAN_data)),title='NRCAN')
# tif_plot(DAYMET_data, extent_map, vmin=min_all, vmax=max_all, title='NRCAN Monthly', do_extent=0)

