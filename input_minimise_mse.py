import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
ds = xr.open_dataset('Data/1km/Rainfall/rainfall_hadukgrid_uk_1km_day_20210801-20210831.nc')

# Select a specific time index
data_slice_original = ds.isel(time=3).copy()

# Get longitude and latitude values
lon = data_slice_original['longitude'].values
lat = data_slice_original['latitude'].values

# Flatten the original rainfall data
flat_rainfall_original = data_slice_original['rainfall'].values.flatten()

# Initialize variables to track the best value and the minimum MSE
best_value = 0
min_mse = float('inf')

# Arrays to store inputted values and corresponding MSE values
input_values = []
mse_values = []

# Create a figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Plot the original rainfall map on the left subplot
img1 = axs[0].imshow(data_slice_original['rainfall'], cmap='Blues', vmin=0, vmax=data_slice_original['rainfall'].max(),
                     extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                     origin='lower')
axs[0].set_title(f'Original Rainfall on {data_slice_original["time"].values}')
axs[0].set_xlabel('Longitude')
axs[0].set_ylabel('Latitude')
fig.colorbar(img1, ax=axs[0], label='Rainfall (mm)')

# Try different values for the second graph
for value in np.arange(0, 4, 0.01):  # Adjust the range and step size as needed
    # Create a copy of the original data
    data_slice_modified = data_slice_original.copy()

    # Set NaN values to be the same in both graphs
    nan_mask = np.isnan(data_slice_original['rainfall'])
    data_slice_modified['rainfall'].values[nan_mask] = np.nan

    # Replace existing values with the current test value in the right graph
    data_slice_modified['rainfall'].values[~nan_mask] = value

    # Flatten the modified rainfall data
    flat_rainfall_modified = data_slice_modified['rainfall'].values.flatten()

    # Exclude NaN values from both arrays
    valid_indices = ~np.isnan(flat_rainfall_original) & ~np.isnan(flat_rainfall_modified)

    # Calculate Mean Squared Error (MSE) for valid indices
    mse = np.mean((flat_rainfall_original[valid_indices] - flat_rainfall_modified[valid_indices])**2)

    # Update the best value and minimum MSE if a better value is found
    if mse < min_mse:
        best_value = value
        min_mse = mse

    # Store inputted value and MSE in arrays
    input_values.append(value)
    mse_values.append(mse)

# Plot inputted value vs. MSE in the center subplot
axs[1].plot(input_values, mse_values, marker='o')
axs[1].set_title('Inputted Value vs. Mean Squared Error')
axs[1].set_xlabel('Inputted Value')
axs[1].set_ylabel('Mean Squared Error')

# Plot the modified rainfall map with the best value on the right subplot
data_slice_best_value = data_slice_original.copy()
nan_mask = np.isnan(data_slice_original['rainfall'])
data_slice_best_value['rainfall'].values[nan_mask] = np.nan
data_slice_best_value['rainfall'].values[~nan_mask] = best_value

img2 = axs[2].imshow(data_slice_best_value['rainfall'], cmap='Blues', vmin=0, vmax=2,
                      extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                      origin='lower')
axs[2].set_title(f'Rainfall Modified with Best Value: {best_value}')
axs[2].set_xlabel('Longitude')
axs[2].set_ylabel('Latitude')
fig.colorbar(img2, ax=axs[2], label='Rainfall (mm)')

# Adjust layout to prevent clipping of titles and labels
plt.tight_layout()

# Show the plot
plt.show()


