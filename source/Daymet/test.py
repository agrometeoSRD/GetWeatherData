#!/usr/bin/env
'''
Creation date: 2020-05-13
Creator : sebastien durocher
Python version :

Description:

Updates:

Notes:


'''

import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np

class extension():
    def __init__(self, Type, var, year, month):
        self.Type = Type
        self.var = var
        self.year = year
        self.month = month
        list_types = ['Normal', 'AgSeason', 'Monthly']
        if int(self.month) < 10:
            self.month = f"0{self.month}"
        if Type not in list_types:
            print(f'WATCH OUT : TYPE NAME {Type} DOES NOT WORK ({list_types})')

    def DAYMET_ext(self):
        fname = ''
        Band = ''
        daymet_var = ''
        if self.var == 'pcp':
            daymet_var = 'prcp'
        elif self.var == 'maxt':
            daymet_var = 'tmax'
        elif self.var == 'mint':
            daymet_var = 'tmin'

        if self.Type == 'Normal':
            fname = f'DAYMET_{daymet_var}_NormalsMonthly.nc'
            Band = f"{self.month}"
        elif self.Type == 'AgSeason':
            fname = f'DAYMET_{daymet_var}_AgSeason{self.year}.nc'
            Band = 'Band1'  # Band doesn't matter for this
        elif self.Type == 'Monthly':
            fname = f"DAYMET_{daymet_var}_Monthly{self.year}.nc"
            Band = self.month
        else:  # This work for any other case that doesn't fit the above criteria
            fname = input('Specify the filename : ')
            Band = '05'
        return fname, Band

    def NRCAN_ext(self, res):
        fname = ''
        nrcan_var = self.var

        if self.Type == 'Normal':
            fname = f"NRCAN_{nrcan_var}_MonthlyNormal{int(self.month)}.tif"
        elif self.Type == 'AgSeason':
            fname = f"NRCAN_{nrcan_var}_AgSeason{self.year}.tif"
        elif self.Type == 'Monthly':
            fname = f"{nrcan_var}{self.year}_{int(self.month)}.tif"
        return fname

    def ERA_ext(self):
        fname = ''
        Band = ''
        if self.var == 'pcp':
            era_var = 'total_precipitation'
        else:
            era_var = "2m_temperature"

        if self.Type == 'Normal':
            # ERA5_total_precipitation_04MonthlyNormals.nc
            fname = f"ERA5_{era_var}_{self.month}MonthlyNormals.nc"
            # no bands for normal
        elif self.Type == 'Monthly':
            print(
                'ERA : Acquiring monthly data for a specific year. All years are within the same file. Assumes year range of 2000 to 2018')
            list_years = list(range(2000, 2019))
            # ERA5_2m_temperature_04MonthlyMean.nc
            fname = f"ERA5_{era_var}_{self.month}MonthlyMean.nc"
            # Band is number index of number of input years
            Band = str([idx for idx, val in enumerate(list_years) if val == self.year][0])
        elif self.Type == 'AgSeason':
            print(
                'ERA : Acquiring agseason data. All years are within the same file.  ssumes year range of 2000 to 2018')
            fname = f"ERA5_{era_var}_AgSeason.nc"
            list_years = list(range(2000, 2019))
            Band = str([idx for idx, val in enumerate(list_years) if val == self.year][0])

        return fname, Band


class Create_Extent:
    def __init__(self, gt, ds):
        self.gt = gt
        self.ds = ds
        self.xres = self.gt[1]
        self.yres = self.gt[5]

    def Boundary(self):
        xmin = self.gt[0] + self.xres * 0.5
        xmax = self.gt[0] + (self.xres * self.ds.RasterXSize) - self.xres * 0.5
        ymin = self.gt[3] + (self.yres * self.ds.RasterYSize) + self.yres * 0.5
        ymax = self.gt[3] - self.yres * 0.5
        extent = [xmin, xmax, ymin, ymax]
        return extent

    def central_pt(self):
        # create a grid of xy coordinates in the original projection
        xmin, xmax, ymin, ymax = self.Boundary()
        xy_source = np.mgrid[xmin:xmax + self.xres:self.xres, ymax + self.yres:ymin:self.yres]

        central_lon = xy_source[0, int(xy_source.shape[1] / 2), int(xy_source.shape[2] / 2)]
        central_lat = xy_source[1, int(xy_source.shape[1] / 2), int(xy_source.shape[2] / 2)]
        return central_lon, central_lat


def load_basemaps():
    shp_Munic_Path = "C:\\Users\\sebastien durocher\\OneDrive - IRDA\\Meteo\\GIS_Region\\QC_Administrative_Regions\\SHP\\"
    shp_Munic_Filename = "FADQ_munics_s_ForMap.shp"
    munic_gdf = gpd.read_file(shp_Munic_Path + shp_Munic_Filename)  # Not loaded, but could be used instead of fadq_df

    shp_Water_Path = r"C:\\Users\\sebastien durocher\\OneDrive - IRDA\\Meteo\\GIS_Region\\QC_Administrative_Regions\\Reseau_National_Hydrog\\"
    shp_Water_Filename = "Slice_hydro_s.shp"
    water_gdf = gpd.read_file(shp_Water_Path + shp_Water_Filename)

    return munic_gdf, water_gdf


def nc_plot(gdf, data, lat, lon,points, vmin=0, vmax=100, dv=10.0, unit='mm', do_extent=0, title='', savefig=1):
    '''

    testing
    data = ERA_data
    lat = ERA_lat
    lon =  ERA_lon
    vmax = 300
    vmin = 270
    dv = 10

    @param gdf: geodataframe
    @param data: numpy array
    @param lat:
    @param lon:
    @param points:
    @return:

    '''
    global extent

    if vmin == -999:
        vmin = int(np.nanquantile(data, 0.01))
    if vmax == -999:
        vmax = int(np.nanquantile(data, 0.99))

    fig, axe = plt.subplots(figsize=(8, 12), dpi=200, subplot_kw={'projection': ccrs.Mercator.GOOGLE})

    clevs = np.arange(vmin, vmax, dv)
    plt.contourf(lon, lat, data, clevs, transform=ccrs.Mercator.GOOGLE, cmap=plt.cm.Blues)

    munic_gdf, water_gdf = load_basemaps()

    water_gdf.plot(color='lightblue', alpha=1, ax=axe, transform=ccrs.Mercator.GOOGLE)  # Add water
    munic_gdf.plot(edgecolor='k', facecolor='none', ax=axe, alpha=0.6, linewidth=0.5,
                   transform=ccrs.Mercator.GOOGLE)  # add munics
    gdf.plot(color='g', ax=axe, transform=ccrs.epsg('3857'))

    axe.scatter(points['lon'],points['lat'],linewidth=3,zorder=2,color='pink')

    cb = plt.colorbar(ax=axe, orientation="vertical", pad=0.02, aspect=16, shrink=0.8)
    cb.set_label(unit, size=12, rotation=0, labelpad=15)

    if do_extent == 1:  # Zoom in AF fields
        axe.set_extent(extent, ccrs.Mercator.GOOGLE)
    else:  # use the netcdf field
        axe.set_extent([-80, -64, 44, 50], ccrs.Mercator.GOOGLE)

    AddLabels(axe, title=title)
    plt.get_current_fig_manager().window.state('zoomed')

    # Savefig
    if savefig == 1:
        # plt.savefig(".png")
        pass


def tif_plot(gdf,data, extent_map,points, do_extent=0, vmin=0, vmax=100, title=''):
    '''

    @param gdf: geodataframe
    @param data: tiff file
    @param extent_map:
    @param points: dictionnary with latitude and longitude. key names should be 'lat' and 'lon'

    @return:
    '''
    # Verify if inputs are propery format
    def check_point_variable():
        if isinstance(points,dict):
            raise TypeError(f'input variable "points" is not a dictionnary (currently : {type(points)})')
        if 'lat' not in points.keys() or 'lon' not in points.keys():
            raise KeyError(f'keys for variable "points" must be "lat" or "lon" (currently : {points.keys()}')

    check_point_variable()

    if vmin == -999:
        vmin = int(np.nanquantile(data, q=0.01))
    if vmax == -999:
        vmax = int(np.nanquantile(data, q=0.99))
    fig, axe = plt.subplots(figsize=(8, 12), dpi=200, subplot_kw={'projection': ccrs.Mercator.GOOGLE})
    tiff = axe.imshow(data, extent=extent_map, origin='upper', cmap='Blues', alpha=0.8, vmin=vmin, vmax=vmax,
                      transform=ccrs.Mercator.GOOGLE)  # for NRCAN

    munic_gdf, water_gdf = load_basemaps()

    water_gdf.plot(color='lightblue', alpha=1, ax=axe, transform=ccrs.Mercator.GOOGLE)  # Add water
    munic_gdf.plot(edgecolor='k', facecolor='none', ax=axe, alpha=0.6, linewidth=0.5,
                   transform=ccrs.Mercator.GOOGLE)  # add munics
    gdf.plot(color='g', ax=axe, transform=ccrs.Mercator.GOOGLE)

    # add data point location
    axe.scatter(points['lon'],points['lat'],linewidth=3,zorder=2,color='pink')


    if do_extent == 1:  # Zoom in AF fields
        axe.set_extent(extent, ccrs.Mercator.GOOGLE)
    else:  # use the netcdf field
        axe.set_extent([-80, -64, 44, 50], ccrs.Mercator.GOOGLE)
    cb = fig.colorbar(tiff)
    cb.set_label('mm', size=12, rotation=0, labelpad=15)

    AddLabels(axe, title=title)
    plt.get_current_fig_manager().window.state('zoomed')

