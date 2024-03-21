import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

### Problem with masking!!!!!
### Mask before or after interpolation?? 3

### Temporary solution: Assign all NaNs to 0 for both linear interpolation and the NN outputs, then pixelwise metrics seem valid

def inverse_distance_weighting(x, y, z, xi, yi, power=1, default_value=0):
    """
    Perform inverse distance weighting interpolation.
    
    Parameters:
        x (array-like): Array of x coordinates of the sampled points.
        y (array-like): Array of y coordinates of the sampled points.
        z (array-like): Array of values at the sampled points.
        xi (array-like): Array of x coordinates of the grid points to be interpolated.
        yi (array-like): Array of y coordinates of the grid points to be interpolated.
        power (float, optional): Power parameter for distance weighting. Default is 2.
        default_value (float, optional): Default value to assign to NaNs in the output array. Default is 0.
        
    Returns:
        zi (ndarray): Interpolated values at the grid points.
    """
    tree = cKDTree(list(zip(x, y)))
    distances, indices = tree.query(np.c_[xi.flatten(), yi.flatten()])
    
    zi = np.zeros_like(xi.flatten()) + default_value
    for i in range(len(zi)):
        if not np.isnan(distances[i]):  # Exclude NaN distances
            if distances[i] == 0:  # If the grid point coincides with a sampled point
                zi[i] = z[indices[i]]
            else:
                weights = 1 / distances[i]**power
                zi[i] = np.sum(weights * z[indices[i]]) / np.sum(weights)
    
    return zi.reshape(xi.shape)



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
        x_grid_12 = data_slice_12['projection_x_coordinate'].values
        y_grid_12 = data_slice_12['projection_y_coordinate'].values
        rainfall_data_12 = data_slice_12['rainfall'].values

        x_mesh_60, y_mesh_60 = np.meshgrid(x_grid_60, y_grid_60)
        x_mesh_12, y_mesh_12 = np.meshgrid(x_grid_12, y_grid_12)

        # Perform linear interpolation from 60km to 12km grid
        interp_rainfall_linear = griddata((x_mesh_60.flatten(), y_mesh_60.flatten()), 
                              rainfall_data_60.flatten(), 
                              (x_mesh_12, y_mesh_12), 
                              method='linear')
        
        # Perform nearest neighbor interpolation from 60km to 12km grid
        interp_rainfall_nearest = griddata((x_mesh_60.flatten(), y_mesh_60.flatten()), 
                              rainfall_data_60.flatten(), 
                              (x_mesh_12, y_mesh_12), 
                              method='nearest')

        print(x_grid_12.shape)
        print(y_grid_12.shape)
        print(rainfall_data_12.shape)

        # Perform inverse distance weighting interpolation from 60km to 12km grid
        interp_rainfall_idw = inverse_distance_weighting(x_mesh_60.flatten(), y_mesh_60.flatten(), rainfall_data_60.flatten(), x_mesh_12, y_mesh_12)

plt.figure(figsize=(15, 5))

# Input
plt.subplot(1, 5, 1)
plt.imshow(rainfall_data_60, cmap='Blues', origin='lower', aspect='auto')
plt.colorbar(label='Rainfall (mm)')
plt.title('Input 60km forecast')

# Plot linearly interpolated rainfall at 12km resolution
plt.subplot(1, 5, 2)
plt.imshow(interp_rainfall_linear, cmap='Blues', origin='lower', aspect='auto')
plt.colorbar(label='Rainfall (mm)')
plt.title('Linear Interpolation')

# Plot nearest neighbor interpolated rainfall at 12km resolution
plt.subplot(1, 5, 3)
plt.imshow(interp_rainfall_nearest, cmap='Blues', origin='lower', aspect='auto')
plt.colorbar(label='Rainfall (mm)')
plt.title('Nearest Neighbor Interpolation')

# Plot nearest neighbor interpolated rainfall at 12km resolution
plt.subplot(1, 5, 4)
plt.imshow(interp_rainfall_idw, cmap='Blues', origin='lower', aspect='auto')
plt.colorbar(label='Rainfall (mm)')
plt.title('Inverse distance weighting')

# Overlay true rainfall values at 12km resolution
plt.subplot(1, 5, 5)
plt.imshow(rainfall_data_12, cmap='Blues', origin='lower', aspect='auto')
plt.colorbar(label='Rainfall (mm)')
plt.title('Output 12km forecast')

plt.tight_layout()
plt.show()


nan_count = np.sum(np.isnan(interp_rainfall_linear))
print(f"Number of NaN values in linear predicted 12km: {nan_count}")

nan_count = np.sum(np.isnan(interp_rainfall_nearest))
print(f"Number of NaN values in nearest neighbour predicted 12km: {nan_count}")

nan_count = np.sum(np.isnan(interp_rainfall_idw))
print(f"Number of NaN values in idw predicted 12km: {nan_count}")

nan_count = np.sum(np.isnan(rainfall_data_12))
print(f"Number of NaN values in true value: {nan_count}")






