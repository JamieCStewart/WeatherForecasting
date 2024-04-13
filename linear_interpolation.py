import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from scipy.ndimage import distance_transform_edt, binary_fill_holes

#### Testing the model 

new_12 = 'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20010401-20010430.nc'
new_60 = 'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20010401-20010430.nc' 

# Define the time index for selecting data
time_for_image = 27

# Load and preprocess the new_60 data
with xr.open_dataset(new_60) as ds_new_60:
    data_slice_60 = ds_new_60.isel(time=time_for_image)
    rainfall_data_60_original = data_slice_60['rainfall'].values

    ds_new_60 = ds_new_60.interpolate_na(dim='projection_x_coordinate', method='nearest')
    ds_new_60 = ds_new_60.interpolate_na(dim='projection_y_coordinate', method='nearest')
    data_slice_60 = ds_new_60.isel(time=time_for_image)
    rainfall_data_60 = data_slice_60['rainfall'].values

    x_grid_60 = data_slice_60['projection_x_coordinate'].values
    y_grid_60 = data_slice_60['projection_y_coordinate'].values

    # Load the grid coordinates from new_12 for interpolation
    with xr.open_dataset(new_12) as ds_new_12:
        data_slice_12 = ds_new_12.isel(time=time_for_image)

        x_grid_12 = data_slice_12['projection_x_coordinate'].values
        y_grid_12 = data_slice_12['projection_y_coordinate'].values
        rainfall_data_12 = data_slice_12['rainfall'].values

        x_mesh_60, y_mesh_60 = np.meshgrid(x_grid_60, y_grid_60)
        x_mesh_12, y_mesh_12 = np.meshgrid(x_grid_12, y_grid_12)

        ### STEP 1 - Input ghost values
        # Nearest neighbour not working
        ## Suggest input average rainfall or 0
        average_rainfall = data_slice_60['rainfall'].mean()
        data_slice_60['rainfall'] = data_slice_60['rainfall'].fillna(average_rainfall)
        rainfall_data_60 = data_slice_60['rainfall'].values
        #nan_count = np.sum(np.isnan(rainfall_data_60))
        #print(f"Number of NaN values in ghost values: {nan_count}")

        ### STEP 2 - Interpolation 
        # Perform linear interpolation from 60km to 12km grid
        interp_rainfall_linear = griddata((x_mesh_60.flatten(), y_mesh_60.flatten()), 
                              rainfall_data_60.flatten(), 
                              (x_mesh_12, y_mesh_12), 
                              method='linear')

        ### STEP 3 - Apply a mask
        interp_rainfall_linear = interp_rainfall_linear.copy()
        interp_rainfall_linear[np.isnan(rainfall_data_12)] = np.nan

plt.figure(figsize=(15, 5))

# Calculate the max
vmax = max(np.nanmax(rainfall_data_60_original), np.nanmax(interp_rainfall_linear), np.nanmax(rainfall_data_12))

# Input
plt.subplot(1, 3, 1)
plt.imshow(rainfall_data_60_original, cmap='Blues', origin='lower', aspect='auto', vmin=0, vmax=vmax)
plt.colorbar(label='Rainfall (mm)')
plt.title('60km forecast')

# Plot linearly interpolated rainfall at 12km resolution
plt.subplot(1, 3, 2)
plt.imshow(interp_rainfall_linear, cmap='Blues', origin='lower', aspect='auto', vmin=0, vmax=vmax)
plt.colorbar(label='Rainfall (mm)')
plt.title('Linear Interpolation')

# Overlay true rainfall values at 12km resolution
plt.subplot(1, 3, 3)
plt.imshow(rainfall_data_12, cmap='Blues', origin='lower', aspect='auto', vmin=0, vmax=vmax)
plt.colorbar(label='Rainfall (mm)')
plt.title('12km forecast')

plt.savefig('linear_interpolation_example.png')

plt.tight_layout()
plt.show()









