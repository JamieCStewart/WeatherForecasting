import xarray as xr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load the dataset
ds = xr.open_dataset('Data/rainfall_hadukgrid_uk_5km_day_20210101-20210131.nc')



