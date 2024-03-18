import subprocess
import os

# Change directory to the Python script directory
os.chdir(r"C:\scripts\GetWeatherData-master")

# Get the total number of lines in the data file
with open(r"C:\scripts\GetWeatherData-master\data\coordinates\VStations.dat", 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Calculate the split index
split_index = (len(lines) // 2) + 1

# Extract the first half of the data file
first_half_data = ''.join(lines[:split_index])

# Save the first half of the data to a temporary file
temp_file_path = r"C:\scripts\GetWeatherData-master\data\coordinates\temp_data.dat"
with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
    temp_file.write(first_half_data)

# Run the first Python script
subprocess.run(["python", "-m", "source.Forecasts.ec_forecasts", "--dat-file", temp_file_path])

# Run the second Python script
subprocess.run(["python", "-m", "source.Forecasts.save_ec_nowcast"])

# Remove the temporary file
os.remove(temp_file_path)
