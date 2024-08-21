"""
File: historical_rainfall_amounts.py
Author: sebastien.durocher
Email: sebastien.rougerie-durocher@irda.qc.ca
Github: https://github.com/MorningGlory747
Description:
- Get historical rainfall amounts
- Use station data from SM
- Select a defined area
- Use the data to create a histogram
Created: 2024-04-02
"""

# Import statements
from utils import utils
from utils import load_areas
import getweatherdata.source.Observations.Stations.get_SM_data as get_SM_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Constants

# Functions
def compute_average_summer_rainfall(df):
    # Filter DataFrame to only include dates between June 1st and August 31st
    df = df[df['Date'].dt.month.isin([6, 7, 8])]

    # Group DataFrame by year and station, and compute the sum of the precipitation for each group
    df_sum = df.groupby([df['Date'].dt.month, df['Date'].dt.year, 'name'])['PR'].sum()
    # rename index
    df_sum.index = df_sum.index.rename(['month','year'],level=[0,1])
    df_sum = df_sum.reset_index()
    # groupby in order to just get one value per month/year (the average of the whole area)
    df_avg = df_sum.groupby(['year','month'])['PR'].mean().reset_index()
    return df_avg

# Main execution ---------------------------------------
# Define the area
config = utils.load_config('gis_config.json')
area = load_areas.load_basemaps(config)
# Select the area of interest (within Monteregie)
area_sel = area[area['MUS_NM_REG'] == 'Montérégie']

# Define the period of interest
year_start = 2000
year_end = 2023
year_range = list(range(year_start, year_end+1))

# Get the corresponding stations within the area
stations_within_area = get_SM_data.get_stations_within_area(area_sel)
stations_within_area.drop_duplicates(subset=['NOM'], inplace=True)

# download the stations
dfs = get_SM_data.download_and_process_data(stations_within_area['NOM'], years=year_range)

# Compute the average summer rainfall for each year
df_avg = compute_average_summer_rainfall(dfs)

# Plot the histogram --------------------------------------------------------------------
month_dict = {6: 'Juin', 7: 'Juillet', 8: 'Août'}

# Replace numeric month values with their names
df_avg['month'] = df_avg['month'].replace(month_dict)

# Create a bar plot
plt.figure(figsize=(12, 6))
sns.barplot(data=df_avg, x='year', y='PR', hue='month')

# Set plot title and labels
plt.title('Accumulation moyenne mensuelle de précipitations en été pour les stations météorologiques situées en Montérégie')
plt.xlabel('Années')
plt.ylabel('Précipitations (mm)')
plt.legend(title='Mois', title_fontsize='13', loc='upper left')

# Show the plot
plt.show()

# Interactive histogram plot -----------------------------
import plotly.express as px
# Using plotly.express to create the bar chart
fig = px.bar(df_avg, x='year', y='PR', color='month', barmode='group', title='Accumulation moyenne mensuelle de précipitations en été pour les stations météorologiques situées en Montérégie')

# Customizing the layout
fig.update_layout(
    xaxis_title='Années',
    yaxis_title='Précipitations (mm)',
    legend_title='Mois'
)

# Show the plot
fig.show()

# To save the plot as an interactive HTML file
fig.write_html('interactive_histogram.html')

# plot the data points on a map ------------------------------
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Assuming df is your DataFrame and it has 'latitude' and 'longitude' columns
stations_within_area['geometry'] = stations_within_area.apply(lambda row: Point(row['Lon'], row['Lat']), axis=1)
stations_gdf = gpd.GeoDataFrame(stations_within_area, geometry='geometry')

# Now plot the GeoDataFrame
fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the size of the map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))  # Get a simple world map
world.plot(ax=ax, color='white', edgecolor='black')  # Plot the world map as the base

# Plot the point data on top of the world map
# You can adjust the size and color of the points with the markersize and color parameters
stations_gdf.plot(ax=ax, markersize=5, color='red')

# Optionally, set the extent of the map to be just slightly larger than your point data
x_min, y_min, x_max, y_max = stations_gdf.total_bounds
dx, dy = x_max - x_min, y_max - y_min  # Calculate the extent difference
ax.set_xlim(x_min - 10 * dx, x_max + 10 * dx)
ax.set_ylim(y_min - 10 * dy, y_max + 10 * dy)

plt.show()

# Map with folium ----------------------------------------
import folium

# Assuming df is your DataFrame and it has 'latitude' and 'longitude' columns
locations = stations_within_area[['Lat', 'Lon']]

# Create a map centered around the first location
m = folium.Map(location=[locations.iloc[0]['Lat'], locations.iloc[0]['Lon']], zoom_start=13)

# Add a marker for each location
for index, row in locations.iterrows():
    folium.Marker([row['Lat'], row['Lon']]).add_to(m)

# Save the map as an HTML file
m.save('map.html')



