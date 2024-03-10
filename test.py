import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
ds1 = xr.open_dataset('Data/1km/Rainfall/rainfall_hadukgrid_uk_1km_day_20210101-20210131.nc')

# Select the first time index
data_slice_original = ds1.isel(time=0)
day_label = 'Day 1'
resolution_label = '1km'

# Generate random numbers between 0 and 10
random_multiplier = np.random.uniform(0, 10, size=data_slice_original['rainfall'].shape)

# Multiply each value by the random number
data_slice_randomized = data_slice_original['rainfall'] * random_multiplier

# Get longitude and latitude values
lon = data_slice_original['longitude'].values
lat = data_slice_original['latitude'].values

# Create a 1x2 grid of subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the original data
img_original = axs[0].imshow(data_slice_original['rainfall'], cmap='Blues', vmin=0,
                              vmax=data_slice_original['rainfall'].max(),
                              extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                              origin='lower')
axs[0].set_title(f'Original Rainfall on {day_label} ({resolution_label} res)')
cbar_original = plt.colorbar(img_original, ax=axs[0], label='Rainfall (mm)')

# Plot the randomized data
img_randomized = axs[1].imshow(data_slice_randomized, cmap='Blues', vmin=0,
                               vmax=data_slice_randomized.max(),
                               extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                               origin='lower')
axs[1].set_title(f'Randomized Rainfall on {day_label} ({resolution_label} res)')
cbar_randomized = plt.colorbar(img_randomized, ax=axs[1], label='Rainfall (mm)')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save output
plt.savefig('rainfall_plots_original_and_randomized.png')

# Show the plot
plt.show()

# Close the dataset
ds1.close()


