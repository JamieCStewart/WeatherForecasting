import xarray as xr
import matplotlib.pyplot as plt
import numpy as np 
#import torch
#import torch.nn as nn
#import torch.optim as optim
#from torch.utils.data import DataLoader, TensorDataset

# Load the dataset
ds = xr.open_dataset('Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20210101-20210131.nc')

# Create a figure with two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Select a specific time index, e.g., the first time step
data_slice = ds.isel(time=0)

# Get longitude and latitude values
lon = data_slice['longitude'].values
lat = data_slice['latitude'].values

# Create a map plot for the first subplot
img1 = axs[0].imshow(data_slice['rainfall'], cmap='Blues', vmin=0, vmax=data_slice['rainfall'].max(),
                     extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                     origin='lower')
axs[0].set_title(f'Rainfall on {data_slice["time"].values}')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
fig.colorbar(img1, ax=axs[0], label='Rainfall (mm)')

# 
data_slice = ds.isel(time=0)

# Create a map plot for the second subplot
img2 = axs[1].imshow(data_slice['rainfall'], cmap='Blues', vmin=0, vmax=data_slice['rainfall'].max(),
                     extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                     origin='lower')
axs[1].set_title(f'Rainfall on {data_slice["time"].values}')
axs[1].set_xlabel('Longitude')
axs[1].set_ylabel('Latitude')
fig.colorbar(img2, ax=axs[1], label='Rainfall (mm)')

# Adjust layout to prevent clipping of titles and labels
plt.tight_layout()

# Show the plot
plt.show()

