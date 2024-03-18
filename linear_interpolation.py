import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator

#### Testing the model 

new_12 = 'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20201101-20201130.nc'
new_60 = 'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20201101-20201130.nc' 

# Define the time index for selecting data
time_for_image = 0

# Load and preprocess the new_60 data
with xr.open_dataset(new_60) as ds_new_60:
    data_slice_60 = ds_new_60.isel(time=time_for_image)
    x_grid_60 = data_slice_60['projection_x_coordinate'].values
    y_grid_60 = data_slice_60['projection_y_coordinate'].values
    rainfall_data_60 = data_slice_60['rainfall'].values

    # Load the grid coordinates from new_12 for interpolation
    with xr.open_dataset(new_12) as ds_new_12:
        data_slice_12 = ds_new_12.isel(time=time_for_image)
        x_grid_12 = ds_new_12['projection_x_coordinate'].values
        y_grid_12 = ds_new_12['projection_y_coordinate'].values
        rainfall_data_12 = data_slice_12['rainfall'].values

        x_mesh_60, y_mesh_60 = np.meshgrid(x_grid_60, y_grid_60)
        x_mesh_12, y_mesh_12 = np.meshgrid(x_grid_12, y_grid_12)

        # Perform linear interpolation from 60km to 12km grid
        interp_rainfall_12 = griddata((x_mesh_60.flatten(), y_mesh_60.flatten()), 
                              rainfall_data_60.flatten(), 
                              (x_mesh_12, y_mesh_12), 
                              method='linear')

plt.figure(figsize=(10, 6))

# Plot interpolated rainfall at 12km resolution
plt.subplot(1, 2, 1)
plt.imshow(interp_rainfall_12, cmap='Blues', origin='lower', aspect='auto')
plt.colorbar(label='Rainfall (mm)')
plt.title('Interpolated Rainfall at 12km Resolution')

# Overlay true rainfall values at 12km resolution
plt.subplot(1, 2, 2)
plt.imshow(rainfall_data_12, cmap='Blues', origin='lower', aspect='auto')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Interpolated Rainfall vs True Values at 12km Resolution')

plt.show()

nan_count = np.sum(np.isnan(interp_rainfall_12))
print(f"Number of NaN values in predicted 12km: {nan_count}")

nan_count = np.sum(np.isnan(rainfall_data_12))
print(f"Number of NaN values in true value: {nan_count}")






