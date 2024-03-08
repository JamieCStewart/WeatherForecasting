import xarray as xr
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

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

# Load datasets
datasets = [xr.open_dataset(path) for path in X_dataset_paths]

# Concatenate rainfall values along a new feature axis (assuming 'rainfall' is the variable of interest)
X_train = xr.concat([ds['rainfall'] for ds in datasets], dim='feature_axis')

print(X_train)

# Optionally, convert the xarray dataset to a numpy array if needed
X_train_array = X_train.values

# Load target datasets
y_datasets = [xr.open_dataset(path) for path in y_dataset_paths]

# Concatenate target values along a new feature axis (assuming 'rainfall' is the target variable)
y_train = xr.concat([ds['rainfall'] for ds in y_datasets], dim='feature_axis')

print(y_train)

# Optionally, convert the xarray dataset to a numpy array if needed
y_train_array = y_train.values

# Close the target datasets
for ds in y_datasets:
    ds.close()

# Now, y_train contains the concatenated target values from all datasets

# Close the datasets
for ds in datasets:
    ds.close()

# Now, X_train contains the concatenated rainfall values from all datasets

# Assuming X_train_array and y_train_array are NumPy arrays
X_train_array = X_train.values
y_train_array = y_train.values

# Split the data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_array, y_train_array, test_size=0.2, random_state=42
)

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train_split).float()
y_train_tensor = torch.from_numpy(y_train_split).float()

X_val_tensor = torch.from_numpy(X_val_split).float()
y_val_tensor = torch.from_numpy(y_val_split).float()

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model
input_size = X_train_tensor.shape[1]
hidden_size = 64
output_size = y_train_tensor.shape[1]  # Assuming y_train is a 2D array
model = SimpleNN(input_size, hidden_size, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 1000
batch_size = 32

# Create DataLoader for training and validation sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            val_loss += criterion(outputs, batch_y).item()

    val_loss /= len(val_loader)
    
    if epoch % 50 == 0:
        print(f'Epoch {epoch}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss}')

# Now, the model is trained, and you can use it for predictions
# For predictions on new data, convert the data to a PyTorch tensor and use the model's forward method
# Example:
# new_data_tensor = torch.from_numpy(new_data).float()
# predictions = model(new_data_tensor)

