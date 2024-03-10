import xarray as xr
import numpy as np

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


with xr.open_dataset('Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_20210101-20210131.nc') as ds:
    # 390 values, 23 horizontal by 17 vertical 
    data_slice = ds.isel(time=0)
    lon = data_slice['longitude'].values
    lat = data_slice['latitude'].values
    print(data_slice['rainfall'])


with xr.open_dataset('Data/5km/Rainfall/rainfall_hadukgrid_uk_5km_day_20210101-20210131.nc') as ds:
    # 52200 values, 290 horizontal by 180 vertical 
    data_slice = ds.isel(time=0)
    lon = data_slice['longitude'].values
    lat = data_slice['latitude'].values
    print(data_slice['rainfall'])

# FIGURE OUT WHY THESE ARENT INTEGERS!!! 
##   60km grid points not a subset of 5km??????
### Linear interpolation hard to get to work if not lined up 

print(52200/390)
print(290 / 23)
print(180 / 17)
