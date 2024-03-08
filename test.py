import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

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
axs[0].set_title(f'Original Rainfall on {data_slice["time"].values}')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
fig.colorbar(img1, ax=axs[0], label='Rainfall (mm)')

# Select a different time index, e.g., the second time step
data_slice_all_ones = ds.isel(time=1).copy()

# Set NaN values to be the same in both graphs
nan_mask = np.isnan(data_slice['rainfall'])
data_slice_all_ones['rainfall'].values[nan_mask] = np.nan

# Replace existing values with 1 in the right graph
data_slice_all_ones['rainfall'].values[~nan_mask] = 1

# Create a map plot for the second subplot with NaN values and existing values replaced with 1
img2 = axs[1].imshow(data_slice_all_ones['rainfall'], cmap='Blues', vmin=0, vmax=1,
                      extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                      origin='lower')
axs[1].set_title(f'Rainfall Modified on {data_slice_all_ones["time"].values}')
axs[1].set_xlabel('Longitude')
axs[1].set_ylabel('Latitude')
fig.colorbar(img2, ax=axs[1], label='Rainfall (mm)')

# Adjust layout to prevent clipping of titles and labels
plt.tight_layout()

# Show the plot
plt.show()

import numpy as np

# Flatten the rainfall data from both images
flat_rainfall_original = data_slice['rainfall'].values.flatten()
flat_rainfall_modified = data_slice_all_ones['rainfall'].values.flatten()

# Exclude NaN values from both arrays
valid_indices = ~np.isnan(flat_rainfall_original) & ~np.isnan(flat_rainfall_modified)

# Calculate Mean Squared Error (MSE) for valid indices
mse = np.mean((flat_rainfall_original[valid_indices] - flat_rainfall_modified[valid_indices])**2)

# Optionally, normalize MSE
max_possible_pixel_value = 1.0  # Assuming your pixel values are normalized between 0 and 1
normalized_mse = mse / (max_possible_pixel_value ** 2)

print(f"Mean Squared Error: {mse}")
print(f"Normalized Mean Squared Error: {normalized_mse}")



