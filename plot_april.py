import xarray as xr
import matplotlib.pyplot as plt

# Load the dataset
ds1 = xr.open_dataset(
                      f'Data/12km/Rainfall/rainfall_hadukgrid'
                      f'_uk_12km_day_20010401-20010430.nc'
                      )

# Extract data for April 27th, 2001
data_slice = ds1.isel(time=26)

# Get longitude and latitude values
lon = data_slice['longitude'].values
lat = data_slice['latitude'].values

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot rainfall using imshow
img = ax.imshow(data_slice['rainfall'], cmap='Blues', vmin=0, 
                vmax=data_slice['rainfall'].max(),
                extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                origin='lower')  # Setting origin to 'lower' flips the y-axis

# Add colorbar
cbar = plt.colorbar(img, ax=ax, label='Rainfall (mm)')

# Set plot title
ax.set_title('Rainfall on April 27th, 2001 (12km resolution)')

# Show the plot
plt.show()

# Close the dataset
ds1.close()

