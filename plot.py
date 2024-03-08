import xarray as xr
import matplotlib.pyplot as plt

# Load the datasets
ds1 = xr.open_dataset('Data/1km/Rainfall/rainfall_hadukgrid_uk_1km_day_20210101-20210131.nc')
ds2 = xr.open_dataset('Data/5km/Rainfall/rainfall_hadukgrid_uk_5km_day_20210101-20210131.nc')
ds3 = xr.open_dataset('Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20210101-20210131.nc')

# Create a 3x3 grid of subplots
fig, axs = plt.subplots(3, 3, figsize=(10, 8))

# Loop through all subplots
for i, ax in enumerate(axs.flatten()):
    # Select a specific time index based on the subplot index and dataset
    if i < 3:
        data_slice = ds1.isel(time=i)
        day_label = f'Day {i+1}'
        resolution_label = '1km'
    elif i < 6:
        data_slice = ds2.isel(time=i-3)
        day_label = f'Day {i-2}'
        resolution_label = '5km'
    else:
        data_slice = ds3.isel(time=i-6)
        day_label = f'Day {i-5}'
        resolution_label = '60km'

    # Get longitude and latitude values
    lon = data_slice['longitude'].values
    lat = data_slice['latitude'].values

    # Create a map plot using xarray.plot and imshow with flipped axes
    img = ax.imshow(data_slice['rainfall'], cmap='Blues', vmin=0, vmax=data_slice['rainfall'].max(),
                    extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                    origin='lower')  # Setting origin to 'lower' flips the y-axis

    # Add a colorbar
    cbar = plt.colorbar(img, ax=ax, label='Rainfall (mm)')

    # Set plot title with resolution information
    ax.set_title(f'Rainfall on {day_label} ({resolution_label} res)')

# Adjust layout to prevent overlap
plt.tight_layout()

# Save output
plt.savefig('rainfall_plots.png')

# Show the plot
plt.show()



# Close the datasets
ds1.close()
ds2.close()
ds3.close()
