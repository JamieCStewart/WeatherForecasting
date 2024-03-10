import xarray as xr
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Masking
from tensorflow.keras.optimizers import Adam


# Assuming you have a list of dataset paths
X_dataset_paths = [
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20210101-20210131.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20210201-20210228.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20210301-20210331.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20210401-20210430.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20210501-20210531.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20210601-20210630.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20210701-20210731.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20210801-20210831.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20210901-20210930.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20211001-20211031.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20211101-20211130.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20211201-20211231.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20220101-20220131.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20220201-20220228.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20220301-20220331.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20220401-20220430.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20220501-20220531.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20220601-20220630.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20220701-20220731.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20220801-20220831.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20220901-20220930.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20221001-20221031.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20221101-20221130.nc',
    'Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20221201-20221231.nc'
]

y_dataset_paths = [
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20210101-20210131.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20210201-20210228.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20210301-20210331.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20210401-20210430.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20210501-20210531.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20210601-20210630.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20210701-20210731.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20210801-20210831.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20210901-20210930.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20211001-20211031.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20211101-20211130.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20211201-20211231.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20220101-20220131.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20220201-20220228.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20220301-20220331.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20220401-20220430.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20220501-20220531.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20220601-20220630.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20220701-20220731.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20220801-20220831.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20220901-20220930.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20221001-20221031.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20221101-20221130.nc',
    'Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_20221201-20221231.nc'
]

# Initialize an empty list to store all rainfall_numpy instances
rainfall_instances = []

# Loop through each data file
for file in X_dataset_paths:
    with xr.open_dataset(file) as ds:
        # Extract the rainfall data for all time steps
        rainfall_data = ds['rainfall'].values
        print(rainfall_data.shape)

        # Append the data for all time steps to the list
        rainfall_instances.append(rainfall_data)

# Concatenate the list of rainfall instances along the first axis
X_train = np.concatenate(rainfall_instances, axis=0)
print(X_train.shape)


# Initialize an empty list to store all rainfall_numpy instances
rainfall_instances = []

# Loop through each data file
for file in y_dataset_paths:
    with xr.open_dataset(file) as ds:
        # Extract the rainfall data for all time steps
        rainfall_data = ds['rainfall'].values
        print(rainfall_data.shape)

        # Append the data for all time steps to the list
        rainfall_instances.append(rainfall_data)

# Concatenate the list of rainfall instances along the first axis
y_train = np.concatenate(rainfall_instances, axis=0)
print(y_train.shape)


### MODEL TIME

### WE NEED MASKING

# Assuming you have your X_train and y_train data loaded

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normalize the data (optional but often beneficial)
X_train_normalized = X_train / 255.0  # Adjust as needed
X_test_normalized = X_test / 255.0    # Adjust as needed

# Create masks for NaN values in X_train and y_train for the entire dataset
nan_mask_X_train = np.isnan(X_train_normalized)
nan_mask_y_train = np.isnan(y_train)

# Fill NaN values with zeros or any other appropriate value for each dataset
X_train_masked = np.where(nan_mask_X_train, 0, X_train_normalized)
y_train_masked = np.where(nan_mask_y_train, 0, y_train)

# Flatten the input data
X_train_flat = X_train_masked.reshape((X_train_masked.shape[0], -1))
X_test_flat = X_test_normalized.reshape((X_test_normalized.shape[0], -1))

# Define a simple neural network model with Masking layer
model = Sequential()
model.add(Masking(mask_value=0, input_shape=(X_train_flat.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(np.prod(y_train.shape[1:]), activation='linear'))  # Adjust the output dimensions

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model with the masked data
model.fit(X_train_flat, y_train_masked.reshape((y_train_masked.shape[0], -1)), epochs=100, batch_size=32, validation_split=0.1)

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

# Load and preprocess the new_60 data
with xr.open_dataset(new_60) as ds_new_60:
    X_new_60 = ds_new_60['rainfall'].values
    nan_mask_X_new_60 = np.isnan(X_new_60)
    X_new_60_masked = np.where(nan_mask_X_new_60, 0, X_new_60)
    X_new_60_flat = X_new_60_masked.reshape((1, -1))  # Reshape to match the input shape of the model

# Make predictions on the new_60 data
predictions_new_60 = model.predict(X_new_60_flat)

# Reshape predictions to match the original shape of y_test
predictions_new_60_reshaped = predictions_new_60.reshape((predictions_new_60.shape[0], y_test.shape[1], y_test.shape[2]))

# Load and preprocess the corresponding new_12 data for comparison
with xr.open_dataset(new_12) as ds_new_12:
    y_new_12 = ds_new_12['rainfall'].values

# Plot the predictions against the true new_12 data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(predictions_new_60_reshaped[0], cmap='viridis', origin='lower', aspect='auto')
plt.title('Predictions (new_60)')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(y_new_12, cmap='viridis', origin='lower', aspect='auto')
plt.title('True new_12')
plt.colorbar()

plt.tight_layout()
plt.show()
