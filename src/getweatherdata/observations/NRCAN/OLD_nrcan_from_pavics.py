#!/usr/bin/env
"""
Creation date: 2023-04-19
Creator : sebastien.durocher 
Python version : 3.10

Description:
- Inspired from the PAVICS tutorial, we will extract NRCAN v2 data for a list of specific lat and lon indices

Updates:

Notes:
- 2023-08-02 DEPRECATED FILE BUT STILL KEEP BECAUSE IT COULD STILL CONTAIN SOME USEFUL CODE. See get_nrcan.py instead
"""
#todo deprecated file but still keep because it could still contain some useful code. See get_nrcan.py instead

# %% imports
from xclim import atmos
from dask import compute
from clisops.core import subset
import pandas as pd
import xarray as xr
from xclim import units
from siphon.catalog import TDSCatalog

# %% get url and metadata
url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/gridded_obs/catalog.html?dataset=datasets/gridded_obs/nrcan_v2.ncml"
# url = "https://pavics.ouranos.ca/twitcher/ows/proxy/thredds/catalog/datasets/simulations/bias_adjusted/cmip5/ouranos/cb-oura-1.0/catalog.xml"  # TEST_USE_PROD_DATA

# Create Catalog
cat = TDSCatalog(url)

# List of datasets
print(f"Number of datasets: {len(cat.datasets)}")

# Access mechanisms - here we are interested in OPENDAP, a data streaming protocol
cds = cat.datasets[0]
print(f"Access URLs: {tuple(cds.access_urls.keys())}")

# %% Open with xarray

ds = xr.open_dataset(cds.access_urls["OPENDAP"], chunks="auto")

# convert units to celcius
ds['tasmax'] = units.convert_units_to(ds['tasmax'], target='degC')
ds['tasmin'] = units.convert_units_to(ds['tasmin'], target='degC')

# Compute mean daily temperature
ds['tasmean'] = (ds['tasmin'] + ds['tasmax']) / 2
ds['tasmean'].attrs = ds['tasmax'].attrs

# %% Open the coordinate dataset (grilles de fertilisation)

print('Reading grilles coordinates')
path = "C:\\Users\\the_l\\OneDrive - IRDA\\Chargé de projets\\Petits projets\\Christine\\"
fname = "PR_DS_Degre_jours.xlsx"

lat_coor = 'SIT_GPS_COORD_POINT1_LATT'
lon_coor = 'SIT_GPS_COORD_POINT1_LONGT'
grilles_df = pd.read_excel(path + fname)

# %% Create the subset


startyear = "1979"
endyear = "2008"
ds_gridpoints = subset.subset_gridpoint(ds, lon=grilles_df[lon_coor].tolist(), lat=grilles_df[lat_coor].tolist())
ds_gridpoints_climate30 = subset.subset_time(ds_gridpoints,
                                             start_date=startyear,
                                             end_date=endyear)

# DJ in Agrometeo works by selecting specific months.
# The growing degree day in xclim works by temperature threshold. So we'll be forcing the code to work within only specific months
# Create a boolean mask for the months of November to March
mask = (ds_gridpoints_climate30['time.month'] >= 4) & (ds_gridpoints_climate30['time.month'] <= 10)
# Apply the mask to the dataset, setting values to NaN for months November to March
ds_gridpoints_climate30 = ds_gridpoints_climate30.where(mask,
                                                        other=0)  # Basically we turned every value within November and March into zeros

# %% Compute the indices (begginer form)

# Version 1 : Compute DJ by computing it for every year and then doing mean
DJ0 = compute(atmos.growing_degree_days(ds_gridpoints_climate30['tasmean'], thresh=0).mean(dim='time'))
DJ5 = compute(atmos.growing_degree_days(ds_gridpoints_climate30['tasmean'], thresh=5).mean(dim='time'))
# Version 2 : compute DJ by averaging temperature for the whole 30 years and then computing the DJ
# I Dont think that makes any sense, because it would cause issues with leap years

# %% Count degree day for the entire xarray dataset
import numpy as np
from clisops.core import subset

# Subsetting using a polygon
# Explore the polygon layer
ds_poly = subset.subset_shape(
    ds, shape="C:\\Temp\\RegionAgricolesQC.geojson"
)
# Add an extra subsetting layer
lon_bnds = [-80.5, -64]
lat_bnds = [44, 50]
ds_poly = subset.subset_bbox(ds_poly , lon_bnds=lon_bnds, lat_bnds=lat_bnds)

# Subsetting the time
startyear = "1979"
endyear = "2008"
ds_poly_climate30 = subset.subset_time(ds_poly,
                                             start_date=startyear,
                                             end_date=endyear)

# DJ in Agrometeo works by selecting specific months.
# The growing degree day in xclim works by temperature threshold. So we'll be forcing the code to work within only specific months
# Create a boolean mask for the months of November to March
mask = (ds_poly_climate30['time.month'] >= 4) & (ds_poly_climate30['time.month'] <= 10)
# Apply the mask to the dataset, setting values to NaN for months November to March
ds_poly_climate30 = ds_poly_climate30.where(mask,other=0)  # Basically we turned every value within November and March into zeros

DJ0_map = compute(atmos.growing_degree_days(ds_poly_climate30['tasmean'], thresh=0).mean(dim='time'))[0]
# DJ0_map = compute(atmos.growing_degree_days(ds_poly_climate30['tasmean'], thresh=5).mean(dim='time'))[0]

# Do a copy just in case things go south
DJ0_map_copy = DJ0_map.copy()

# Quick fix where we open a saved netcdf of DJ0_map
# import xarray as xr
# import numpy as np
# DJ0_map = xr.open_dataarray('C:\\temp\\dj_christine.nc')

# Categorise the values based on the following three sets of values
# Define the bins
bins = [np.NINF, 2418, 2810, np.PINF]
# categorize the values
DJ0_map_cat = DJ0_map.copy()
DJ0_map_cat = xr.where(DJ0_map_cat < bins[1], 1, DJ0_map_cat)  # values < 2418 are categorized as 1
DJ0_map_cat = xr.where((DJ0_map_cat >= bins[1]) & (DJ0_map_cat < bins[2]), 2,DJ0_map_cat)  # values >= 2418 and < 2810 are categorized as 2
DJ0_map_cat = xr.where(DJ0_map_cat >= bins[2], 3, DJ0_map_cat)  # values >= 2810 are categorized as 3

#%%( DOES NOT WORK ) Map the data
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns

def load_basemaps():
    shp_Munic_Path = "E:\\Projet_FADQ\\Donnees_FADQ\\shapefiles\\"
    shp_Munic_Filename = "FADQ_munics_s_ForMap.shp"
    munic_gdf = gpd.read_file(shp_Munic_Path + shp_Munic_Filename)  # Not loaded, but could be used instead of fadq_df

    shp_Water_Path = "D:\\EnviroData\\GIS_Region\\QC_Administrative_Regions\\Reseau_National_Hydrog\\"
    shp_Water_Filename = "Slice_hydro_s.shp"
    water_gdf = gpd.read_file(shp_Water_Path + shp_Water_Filename)

    return munic_gdf, water_gdf

def PrettySetup():
    sns.set(font_scale=4)
    sns.set_style("white")
    plt.rcParams["axes.grid"] = True
    plt.rcParams["font.family"] = "Times New Roman"
def AddLabels(in_ax, xlabel='', ylabel='',
              xticklabel = None,yticklabel=None,rotate=None,
              title='',ticksize=15,labelsize=17,legend=None,legend_label=None):

    PrettySetup()

    # Add stuff
    if 'FacetGrid' in str(type(in_ax)):
        for axes in in_ax.axes.flat:
            _ = axes.set_xticklabels(axes.get_xticklabels(), fontsize=ticksize,rotation=rotate)
            _ = axes.set_yticklabels(axes.get_yticklabels(), fontsize=ticksize,rotation=rotate)
            axes.set_xlabel(xlabel, fontsize=labelsize)
            axes.set_ylabel(ylabel, fontsize=labelsize)
            axes.set_title(title, fontsize=labelsize)
    else:
        in_ax.set_xlabel(xlabel, fontsize=labelsize)
        in_ax.set_ylabel(ylabel, fontsize=labelsize)
        in_ax.set_title(title, fontsize=labelsize)
        in_ax.xaxis.set_tick_params(labelsize=ticksize,rotation=rotate)
        in_ax.yaxis.set_tick_params(labelsize=ticksize,rotation=rotate)
    if xticklabel != None:
        in_ax.set_xticklabels(xticklabel)
    if yticklabel != None:
        in_ax.set_yticklabels(yticklabel)
    if legend == True:
        handles, label = in_ax.get_legend_handles_labels()
        if legend_label != None: # If we want to modify legend, it has to have both label and color
            if isinstance(legend_label,dict):
                if 'handles' in legend_label.keys():
                    handles = legend_label['handles']
                if 'label' in legend_label.keys():
                    label = legend_label['label']
                in_ax.legend(handles=handles, labels=label, prop={'size': ticksize})
                # patch_list = list()
                # for el in legend_label.keys():
                #     patch_list.append(mpatches.Patch(color=legend_label[el], label=el))
                # in_ax.legend(handles=patch_list, prop={'size': ticksize})
            else:
                print('Unacceptable legend format. Make sure it is a dict with name + color')
        else:
            in_ax.legend(handles=handles, labels=label, prop={'size': ticksize})

    plt.get_current_fig_manager().window.state('zoomed')

import matplotlib.colors as mcolors
import geopandas as gpd
def tif_plot(axe, fig, data, extent, proj =  ccrs.Mercator.GOOGLE, points=None, do_extent=False,
             vmin=-999, vmax=-999,cbticks = 10,show_colorbar=True,colorbar='continous',ticksize=15,
             title='', units=None,labelsize=20):
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

    if vmin == -999:
        vmin = int(np.nanquantile(data, q=0.01))
    if vmax == -999:
        vmax = int(np.nanquantile(data, q=0.99))

    cmap = mcolors.ListedColormap(['red', 'yellow', 'green'])
    cmap.set_bad(color='white')

    tiff = axe.imshow(data, extent=extent, origin='upper', cmap=cmap, alpha=1, vmin=vmin, vmax=vmax,
                      transform=proj, interpolation='spline16')


    munic_gdf, water_gdf = load_basemaps()

    water_gdf.plot(color='lightblue', alpha=1, ax=axe, transform=proj)  # Add water
    munic_gdf.plot(edgecolor='k', facecolor='none', ax=axe, alpha=0.8, linewidth=0.5,
                   transform=proj)  # add munics

    # add data point location
    if points is not None:
        check_point_variable()
        axe.scatter(points['lon'],points['lat'],s=12,zorder=2,color='black')

    if do_extent:  # Zoom in AF fields
        axe.set_extent(extent, proj)
    else:  # use the netcdf field
        axe.set_extent([-80, -64, 44, 50], proj)

    if show_colorbar:
        cb = plt.colorbar(tiff,ax=axe,fraction=0.046, pad=0.04)
        cb.set_label(units, size=12, rotation=0)
        cb.ax.tick_params(labelsize=ticksize)

    AddLabels(axe, title=title,labelsize=labelsize)
    # plt.get_current_fig_manager().window.state('zoomed')
    return tiff

def do_2d_interpolation(data):
    # Taken from : https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
    # Replace NaN values with linear interpolation
    from scipy import interpolate
    x = np.arange(0, data.shape[1])
    y = np.arange(0, data.shape[0])
    # mask invalid values
    array = np.ma.masked_invalid(data)
    xx, yy = np.meshgrid(x, y)
    # get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                               (xx, yy),
                               method='cubic')

    return GD1

data = DJ0_map_cat.to_numpy()

# Data must be a 3d numpy array
# ----------
plot_aspect = 1.5
plot_height = 10.0
plot_width = int(plot_height * plot_aspect)
# ----------
# vmin = myround(int(data.min()))
# vmax = myround(int(data.max()))
show_colorbar = False
colorbar = 'discrete'
vmin = 1  # DOY (244 : 09-01 ; 258 : 09-15)
vmax = 3  # DOY (305 11-01 ; 313 : 11-11)
vstep = 1
step_range = np.arange(vmin, vmax + 1, vstep)
cb_ticks_renamed = pd.to_datetime(step_range, format='%j').strftime('%m-%d')
ticksize = 25
cb_dict = {'show_colorbar': show_colorbar, 'colorbar': colorbar,
           'vmin': vmin, 'vmax': vmax, 'cbticks': len(step_range) - 1, 'ticksize': ticksize}
# ----------
# Define extent and projection
extent = [-79.54, -64.0416, 45.04166794, 49.95833206]
proj = ccrs.PlateCarree()
# ----------
fig, axe = plt.subplots(figsize=(plot_width, plot_height),
                        facecolor='w',
                        subplot_kw={'projection': proj})
plt.subplots_adjust(left=0.05, right=1.00, top=0.90, bottom=0.06, hspace=0.30)
# ----------
# Perform plot
title = 'Titre test'
tiff = tif_plot(axe, fig, data, extent, proj=proj, do_extent=True,
                        **cb_dict,
                        title=title, labelsize=30)


#%% Trying out the chatgpt plot because the last one doesnt fit
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj_type = ccrs.PlateCarree()

my_green = (117/255, 170/255, 65/255)
my_yellow = (250/255, 220/255, 83/255)
my_red = (213/255, 120/255, 90/255)

cmap = mcolors.ListedColormap([my_green, my_yellow, my_red])
norm = mcolors.BoundaryNorm([0.5,1.5,2.5,3.5], cmap.N)

# Assuming 'ds' is your xarray Dataset
ds_interp = DJ0_map_cat.interp(lat=np.linspace(DJ0_map_cat.lat.min(), DJ0_map_cat.lat.max(), 800),
                      lon=np.linspace(DJ0_map_cat.lon.min(), DJ0_map_cat.lon.max(), 800))

_, water_gdf = load_basemaps()

# %% Plot Here
fig = plt.figure(figsize=[10, 7])
ax = plt.axes(projection=proj_type)

# set extent
lon_min, lon_max = ds_interp.lon.min(), ds_interp.lon.max()
lat_min, lat_max = ds_interp.lat.min(), ds_interp.lat.max()
ax.set_extent([lon_min, lon_max, lat_min, lat_max])

# Plot the data
img = ax.pcolormesh(ds_interp.lon, ds_interp.lat, ds_interp.values, cmap=cmap, norm=norm, transform=proj_type)

# Add a colorbar
cbar = plt.colorbar(img, ax=ax, orientation='vertical', pad=0.03, aspect=40, shrink=0.7)
cbar.set_ticks([1, 2, 3])
cbar.ax.set_yticklabels(['<2450', '2450-2850', '>2850'],fontsize=30)  # set the labels of the colorbar

# Create a new axis for the colorbar label and position it
cax = fig.add_axes([ax.get_position().x1+0.01, ax.get_position().y0, 0.02, ax.get_position().height])
cax.axis('off')  # hide the new axis
cax.text(0, 1.5, 'DJ (0)', rotation=0,fontsize=30)  # add a horizontal label

# Draw lakes, rivers and land
# ax.add_feature(cfeature.LAKES.with_scale('10m'), alpha=1, color='lightblue')
# ax.add_feature(cfeature.RIVERS.with_scale('10m'), alpha=1, color='lightblue')
# ax.add_feature(cfeature.OCEAN.with_scale('10m'), alpha=1, color='lightblue')
# ax.add_feature(cfeature.LAND.with_scale('10m'), alpha=1, color='lightgrey')
water_gdf.plot(color='lightblue', alpha=1, ax=ax, transform=proj_type)  # Add water

# Define coordinates for cities
cities = {
    'Montreal': (-73.5673, 45.5017),
    'Saint-Hyacinthe': (-72.9554, 45.6307),
    'Drummondville': (-72.4851, 45.8838),
    'Québec': (-71.2082, 46.8139),
    'Normandin': (-72.5278, 48.8342),
    'Val-d\'Or': (-77.7828, 48.0975),
    'La Pocatière': (-70.0389, 47.3642),
    'Tadoussac': (-69.7075, 48.1375),
    'Gaspé': (-64.4786, 48.8378),
    'Trois-Rivières' : (-72.5411, 46.3508),
    'Joliette' : (-73.4350, 46.0214),
    'St-Jérôme' : (-74.0169, 45.7769),
    'Gatineau' : (-75.6981, 45.4773),
    'Ville Marie' : (-79.4331, 47.3331)
}

# Plot each city
import matplotlib.patheffects as pe
for city, (lon, lat) in cities.items():
    ax.scatter(lon, lat, transform=proj_type, color='black',s=10)
    ax.text(lon + 0.1, lat - 0.1, city, transform=proj_type, color='black',
            fontsize=15,path_effects=[pe.withStroke(linewidth=4, foreground="white")])

plt.show()

#%% Other attempts #%% Do interactive maps of xarray dataset using pavics system
# # import logging
# # import warnings
# # from pathlib import Path
#
# # import holoviews as hv
# import hvplot # MUST IMPORT THIS
# # import hvplot.pandas
# import hvplot.xarray # MUST IMPORT THIS
# # import numpy as np
# # import pandas as pd
# # import panel as pn
# # import xarray as xr
# # from bokeh.models.tools import HoverTool
# # from clisops.core import subset
# # from holoviews import streams
# # from xclim import ensembles as xens
# # from xclim.core import units
#
# map1 = DJ0_map_cat.hvplot.quadmesh(
#     cmap="Spectral_r", geo=True, tiles="EsriImagery", framewise=False, frame_width=400
# )
# hvplot.show(map1)
#
# # Define the categories
# bins = [DJ0_map.min(), 2418, 2810, DJ0_map.max()]
# # Bin the data
# DJ0_map_binned = xr.apply_ufunc(np.digitize, DJ0_map, bins)
# # Define the colormap as a dictionary
# cmap = {1: 'blue', 2: 'green', 3: 'red'}
# # Create the interactive map
# plot = DJ0_map_binned.hvplot.image(x='lon', y='lat', cmap=cmap, hover=True)
# hvplot.show(plot)
#
# # Modify the colormap
# from bokeh.models import FixedTicker
# cat_map = DJ0_map.hvplot.quadmesh(geo=True).opts(
#     cmap=["green", "yellow","red"],
#     colorbar_opts={"ticker": FixedTicker(ticks=[int(DJ0_map.min()), 2418, 2810, int(DJ0_map.max())])},
#     color_levels=[int(DJ0_map.min()), 2418, 2810, int(DJ0_map.max())],
#     clim=(int(DJ0_map.min()), int(DJ0_map.max())),
# )
# hvplot.show(cat_map)



#%% Try with plotly
# import plotly.graph_objects as go
# import plotly.express as px
#
# # Define boundaries
# dfp = DJ0_map.to_dataframe().reset_index().dropna()
# lon_min, lon_max = dfp['lon'].min(), dfp['lon'].max()
# lat_min, lat_max = dfp['lat'].min(), dfp['lat'].max()
#
# # Define color bins manually
# bins = [-np.inf, 2418, 2810, np.inf]
# labels = ['< 2418', '2418 - 2810', '> 2810']
#
# # Create a new column 'category' based on the bins and labels defined
# dfp['category'] = pd.cut(dfp['growing_degree_days'], bins=bins, labels=labels)
#
# # Map labels to colors
# color_discrete_map = {'< 2418': 'blue', '2418 - 2810': 'green', '>= 2810': 'red'}
#
# fig = go.Figure()
# # fig = px.scatter_geo(dfp, lat='lat', lon='lon', color=dfp['growing_degree_days'],
# #                      color_continuous_scale=px.colors.sequential.Plasma,
# #                      projection='natural earth')
# # Add scatter geo data
# fig.add_trace(
#     go.Scattergeo(
#         lon=dfp['lon'],
#         lat=dfp['lat'],
#         text=dfp['category'],
#         mode='markers',
#         marker=dict(
#             size=8,
#             opacity=0.8,
#             reversescale=True,
#             autocolorscale=False,
#             symbol='square',
#             line=dict(
#                 width=1,
#                 color='rgba(102, 102, 102)'
#             ),
#             colorscale=[[0, "blue"], [0.5, "green"], [1, "red"]],
#             cmin=dfp['category'].cat.codes.min(),
#             color=dfp['category'].cat.codes,
#             cmax=dfp['category'].cat.codes.max(),
#             colorbar_title="Categories"
#         )
#     )
# )
#
# # Set geo boundaries
# fig.update_geos(
#     resolution=50,
#     showcoastlines=True, coastlinecolor="RebeccaPurple",
#     showland=True, landcolor="LightGrey",
#     showocean=True, oceancolor="Azure",
#     showlakes=True, lakecolor="Azure",
#     showrivers=True, rivercolor="Azure",
#     lonaxis_range=[lon_min, lon_max],
#     lataxis_range=[lat_min, lat_max],
# )
#
# fig.show()
#

# %% Merge to dataset and send to Mick
DJ0_df = DJ0[0].to_dataframe().rename(columns={'growing_degree_days': 'DJ(0)'})
DJ5_df = DJ5[0].to_dataframe().rename(columns={'growing_degree_days': 'DJ(5)'})
grilles_df['DJ0_num'] = DJ0_df['DJ(0)']
grilles_df['DJ5_num'] = DJ5_df['DJ(5)']

# %% Maps
import geopandas as gpd
import folium
import branca.colormap as cm

# Convert the pandas dataframe to a geodataframe
gdf = gpd.GeoDataFrame(
    grilles_df, geometry=gpd.points_from_xy(grilles_df[lon_coor], grilles_df[lat_coor])
)

# Calculate the extent
min_lon, max_lon = gdf.geometry.x.min(), gdf.geometry.x.max()
min_lat, max_lat = gdf.geometry.y.min(), gdf.geometry.y.max()


# Create a base map
def create_map(lat, lon, zoom):
    m = folium.Map(location=[lat, lon], zoom_start=zoom)
    return m


# Add points to the map
def add_points(m, gdf, column, colormap):
    for idx, row in gdf.iterrows():
        popup_text = f"#champ: {row['SIT_CD']} <br> {column}: {row[column]}"
        popup = folium.Popup(popup_text, max_width=250)

        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color=colormap(row[column]),
            fill=True,
            fill_opacity=0.7,
            popup=popup
        ).add_to(m)


# Create maps for DJ0_num and DJ5_num
map_dj0 = create_map((min_lat + max_lat) / 2, (min_lon + max_lon) / 2, 4)
colormap_dj0 = cm.LinearColormap(['yellow', 'red'], vmin=gdf['DJ0_num'].min(), vmax=gdf['DJ0_num'].max())
colormap_dj0.caption = 'DJ0_num'
colormap_dj0.add_to(map_dj0)
add_points(map_dj0, gdf, 'DJ0_num', colormap_dj0)

map_dj5 = create_map((min_lat + max_lat) / 2, (min_lon + max_lon) / 2, 4)
colormap_dj5 = cm.LinearColormap(['yellow', 'red'], vmin=gdf['DJ5_num'].min(), vmax=gdf['DJ5_num'].max())
colormap_dj5.caption = 'DJ5_num'
colormap_dj5.add_to(map_dj5)
add_points(map_dj5, gdf, 'DJ5_num', colormap_dj5)

# Display the maps
map_dj0.save('map_dj0.html')
map_dj5.save('map_dj5.html')

# %% Compute the indices (Advance form, incomplete)
from xclim import atmos

calcs = []
# Growing degree day - base 0
calcs.append(
    dict(
        func=atmos.growing_degree_days,
        invars=dict(tas="tasmean"),
        args=dict(thresh=0, freq="Y"),
    )
)

# Growing degree day - base 5
calcs.append(
    dict(
        func=atmos.growing_degree_days,
        invars=dict(tas="tasmean"),
        args=dict(thresh=5, freq="Y"),
    )
)


def indicator_calc(func, outfile, args, attrs):
    dsout = xr.Dataset(attrs=attrs)
    out = func(**args)
    dsout[out.name] = out
    return dsout.to_netcdf(outfile, mode="w", compute=False)


for c in calcs:
    jobs = []
    # add the subsetted dataset to the list of input arguments
    args = c["args"]
    for v, vv in c["invars"].items():
        args[v] = ds_gridpoints_climate30[vv]
    # call our custom sub-function with the arguments and data
    # append the returned dask.delayed object to our list
    # outfile = 'C:\\temp\\tmp.nc'
    # jobs.append(indicator_calc(c["func"], outfile, args, ds_gridpoints_climate30.attrs))
    stophere