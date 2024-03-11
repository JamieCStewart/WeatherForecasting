import calendar 
import xarray as xr 
import numpy as np 

def generate_rainfall_paths(start_year, end_year, low_res, high_res):
    months = [f"{i:02d}" for i in range(1, 13)]  # Generate month strings with leading zeros
    
    X_dataset_paths = []
    y_dataset_paths = []

    for year in range(start_year, end_year + 1):
        for month in months:
            last_day = calendar.monthrange(year, int(month))[1]
            X_file_path = f"Data/{low_res}km/Rainfall/rainfall_hadukgrid_uk_{low_res}km_day_{year}{month}01-{year}{month}{last_day}.nc"
            y_file_path = f"Data/{high_res}km/Rainfall/rainfall_hadukgrid_uk_{high_res}km_day_{year}{month}01-{year}{month}{last_day}.nc"
            X_dataset_paths.append(X_file_path)
            y_dataset_paths.append(y_file_path)

    return X_dataset_paths, y_dataset_paths 


def load_data(X_dataset_paths, y_dataset_paths):
    rainfall_instances = []

    for file in X_dataset_paths:
        with xr.open_dataset(file) as ds:
            rainfall_data = ds['rainfall'].values
            rainfall_instances.append(rainfall_data)

    X = np.concatenate(rainfall_instances, axis=0)

    rainfall_instances = []


    for file in y_dataset_paths:
        with xr.open_dataset(file) as ds:
            rainfall_data = ds['rainfall'].values
            rainfall_instances.append(rainfall_data)

    y = np.concatenate(rainfall_instances, axis=0)

    return X, y 