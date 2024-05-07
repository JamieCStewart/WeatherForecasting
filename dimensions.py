# main.py
#
# Author: Jamie Stewart 
# Date: February - March 2024
# Description: A Python script for a masters project titled "Learning the 
# environment deeply". In particular, I am building a neural network to 
# downscale low resolution forecasts

import data
import csv 
import numpy as np 
from model import create_model, compile_model
from evaluate_model import evaluate_model
import matplotlib.pyplot as plt 
import os 
import netCDF4 as nc

km1_file_path = "Data/1km/Rainfall/rainfall_hadukgrid_uk_1km_day_20220101-20220131.nc"
km5_file_path = "Data/5km/Rainfall/rainfall_hadukgrid_uk_5km_day_20220101-20220131.nc"
km12_file_path = "Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20220101-20220131.nc"
km25_file_path = "Data/25km/Rainfall/rainfall_hadukgrid_uk_25km_day_20220101-20220131.nc"
km60_file_path = "Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20220101-20220131.nc"

# Open the NetCDF files
km1_dataset = nc.Dataset(km1_file_path)
km5_dataset = nc.Dataset(km5_file_path)
km12_dataset = nc.Dataset(km12_file_path)
km25_dataset = nc.Dataset(km25_file_path)
km60_dataset = nc.Dataset(km60_file_path)

# Access the shape of each file
km1_shape = km1_dataset.variables['rainfall'][:].shape
km5_shape = km5_dataset.variables['rainfall'][:].shape
km12_shape = km12_dataset.variables['rainfall'][:].shape
km25_shape = km25_dataset.variables['rainfall'][:].shape
km60_shape = km60_dataset.variables['rainfall'][:].shape

print("Shape of 1km file:", km1_shape)
print("Shape of 5km file:", km5_shape)
print("Shape of 12km file:", km12_shape)
print("Shape of 25km file:", km25_shape)
print("Shape of 60km file:", km60_shape)

# Define file paths for each resolution
resolutions = {
    "1km": "Data/1km/Rainfall/rainfall_hadukgrid_uk_1km_day_20220101-20220131.nc",
    "5km": "Data/5km/Rainfall/rainfall_hadukgrid_uk_5km_day_20220101-20220131.nc",
    "12km": "Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20220101-20220131.nc",
    "25km": "Data/25km/Rainfall/rainfall_hadukgrid_uk_25km_day_20220101-20220131.nc",
    "60km": "Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20220101-20220131.nc"
}

# Loop through each resolution
for resolution, file_path in resolutions.items():
    # Open the NetCDF file
    dataset = nc.Dataset(file_path)
    
    # Access rainfall data for one day (e.g., 2022-01-01)
    rainfall_data = dataset.variables['rainfall'][0, :, :]  # Assuming you want data for the first day
    
    # Count NaN and non-NaN values
    nan_count = sum(sum(rainfall_data.mask))
    non_nan_count = rainfall_data.size - nan_count
    
    # Print the results
    print(f"Resolution: {resolution}")
    print(f"NaN values: {nan_count}")
    print(f"Non-NaN values: {non_nan_count}")
    
    # Close the NetCDF file
    dataset.close()

# Close the NetCDF files
km1_dataset.close()
km5_dataset.close()
km12_dataset.close()
km25_dataset.close()
km60_dataset.close()

    


