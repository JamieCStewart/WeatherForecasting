import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Masking, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import data 
from scipy.interpolate import NearestNDInterpolator, griddata

X_paths, y_paths = data.generate_rainfall_paths(2000, 2019, 60, 12)

X, y = data.load_data(X_paths, y_paths)

X_train = X 
y_train = y 

X_train, X_test, y_train, y_test = data.test_train_split(X, y, 0.2, random_state=10)

# Create masks for NaN values in X_train and y_train for the entire dataset
nan_mask_X_train = np.isnan(X_train)
nan_mask_y_train = np.isnan(y_train)

# Fill NaN values with zeros or any other appropriate value for each dataset
X_train_masked = np.where(nan_mask_X_train, 0, X_train)
y_train_masked = np.where(nan_mask_y_train, 0, y_train)

# Flatten the input data
X_train_flat = X_train_masked.reshape((X_train_masked.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

# Define a simple neural network model with Masking layer
model = Sequential()
model.add(Masking(mask_value=0, input_shape=(X_train_flat.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(np.prod(y_train.shape[1:]), activation='linear'))  # Adjust the output dimensions

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model with the masked data
model.fit(X_train_flat, y_train_masked.reshape((y_train_masked.shape[0], -1)), epochs=5, batch_size=32, validation_split=0.1)

# Evaluate the model on the test set
test_loss = model.evaluate(X_test_flat, y_test.reshape((y_test.shape[0], -1)))
print(f'Test Loss: {test_loss}')

# Make predictions
predictions = model.predict(X_test_flat)

# Reshape predictions to match the original shape of y_test
predictions_reshaped = predictions.reshape((predictions.shape[0], y_test.shape[1], y_test.shape[2]))


#### Testing the model 

new_12 = 'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20201201-20201231.nc'
new_60 = 'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20201201-20201231.nc' 

time_for_image = 0

# Load and preprocess the new_60 data
with xr.open_dataset(new_60) as ds_new_60:
    data_slice = ds_new_60.isel(time=time_for_image)
    X_new_60 = data_slice['rainfall'].values
    nan_mask_X_new_60 = np.isnan(X_new_60)
    X_new_60_masked = np.where(nan_mask_X_new_60, 0, X_new_60)
    X_new_60_flat = X_new_60_masked.reshape((1, -1))  # Reshape to match the input shape of the model
    print(X_new_60_flat.shape) 

# Make predictions on the new_60 data
predictions_new_60 = model.predict(X_new_60_flat)

# Reshape predictions to match the original shape of y_test
predictions_new_60_reshaped = predictions_new_60.reshape((predictions_new_60.shape[0], y_test.shape[1], y_test.shape[2]))

# Load and preprocess the corresponding new_12 data for comparison
with xr.open_dataset(new_12) as ds_new_12:
    data_slice = ds_new_12.isel(time=time_for_image)
    y_new_12 = data_slice['rainfall'].values

# Assuming predictions_new_60_reshaped, X_new_60, and y_new_12 are your data arrays
plt.figure(figsize=(18, 6))

X_new_60_flat = X_new_60_flat.reshape((X_new_60_flat.shape[0], X_test.shape[1], X_test.shape[2]))
# Find the maximum value across all three data arrays
max_value = np.max([np.nanmax(X_new_60_flat), np.nanmax(predictions_new_60_reshaped), np.nanmax(y_new_12)])



X_new_60_flat = X_new_60_flat.reshape((X_new_60_flat.shape[0], X_test.shape[1], X_test.shape[2]))

# Apply the NaN mask to predictions to match the pattern in y_train
predictions_new_60_reshaped[0][nan_mask_y_train[0]] = np.nan
print(nan_mask_y_train[0].shape)
print(predictions_new_60_reshaped[0].shape)


#### TEST Linear Interpolation  
# Extract projection coordinates
x_grid = ds_new_12['projection_x_coordinate'].values
y_grid = ds_new_12['projection_y_coordinate'].values

linear_interp = ds_new_60[['rainfall']].interp(
            projection_x_coordinate=x_grid,
            projection_y_coordinate=y_grid,
            method="linear",
        )

print("Fingers crossed this works")
linear_interp_data = linear_interp['rainfall']




# Assuming predictions_new_60_reshaped, X_new_60, and y_new_12 are your data arrays
plt.figure(figsize=(18, 6))

# Plot original 60km data
plt.subplot(1, 4, 1)
cmap = plt.cm.Blues  # Choose your colormap
cmap.set_bad(color=cmap(0.0))
plt.imshow(X_new_60_flat[0], cmap=cmap, origin='lower', aspect='auto', vmin=0, vmax=max_value)  # Set vmin and vmax
plt.title('Original 60km')
plt.colorbar()

# Plot predicted data
plt.subplot(1, 4, 2)
plt.imshow(predictions_new_60_reshaped[0], cmap=cmap, origin='lower', aspect='auto', vmin=0, vmax=max_value)  # Set vmin and vmax
plt.title('Predicted 12km')
plt.colorbar()

# Plot true data
plt.subplot(1, 4, 3)
plt.imshow(y_new_12, cmap=cmap, origin='lower', aspect='auto', vmin=0, vmax=max_value)  # Set vmin and vmax
plt.title('True 12km')
plt.colorbar()

# Plot linear interpolation predictions
plt.subplot(1, 4, 4)
plt.imshow(linear_interp_data[0], cmap=cmap, origin='lower', aspect='auto', vmin=0, vmax=max_value)  # Set vmin and vmax
plt.title('Linear Interpolation 12km')
plt.colorbar()

plt.tight_layout()
plt.show()

print(X_new_60_flat[0].shape)
print(predictions_new_60_reshaped[0].shape)
print(linear_interp_data[0].shape)
print(y_new_12.shape)

nan_count = np.sum(np.isnan(predictions_new_60_reshaped))
print(f"Number of NaN values in predicted 12km: {nan_count}")

nan_count = np.sum(np.isnan(y_new_12))
print(f"Number of NaN values in true value: {nan_count}")