import calendar 
import xarray as xr
import numpy as np 

def evaluate_model(model, low_res, high_res, year, month):
        month = str(month).zfill(2)
        last_day = calendar.monthrange(year, int(month))[1]

        new_12 = f"Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_{year}{month}01-{year}{month}{last_day}.nc"
        new_60 = f"Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_{year}{month}01-{year}{month}{last_day}.nc"

        loss_values = []

        for i in range(last_day): 
                time_for_image = i

                # Load and preprocess the new_60 data
                with xr.open_dataset(new_60) as ds_new_60:
                        data_slice = ds_new_60.isel(time=time_for_image)
                        X_new_60 = data_slice['rainfall'].values
                        nan_mask_X_new_60 = np.isnan(X_new_60)
                        X_new_60_masked = np.where(nan_mask_X_new_60, 0, X_new_60)
                        X_new_60_flat = X_new_60_masked.reshape((1, -1))  # Reshape to match the input shape of the model

                # Make predictions on the new_60 data
                predictions_new_60 = model.predict(X_new_60_flat)

                # Load and preprocess the corresponding new_12 data for comparison
                with xr.open_dataset(new_12) as ds_new_12:
                        data_slice = ds_new_12.isel(time=time_for_image)
                        y_new_12 = data_slice['rainfall'].values

                # Reshape predictions to match the original shape of y_test
                predictions_new_60_reshaped = predictions_new_60.reshape((predictions_new_60.shape[0], y_new_12.shape[0], y_new_12.shape[1]))
                predictions_new_60_reshaped = predictions_new_60.reshape(y_new_12.shape[0], y_new_12.shape[1])

                # Apply the NaN mask to predictions to match the pattern in y_train
                nan_mask_y = np.isnan(y_new_12)
                predictions_new_60_reshaped[nan_mask_y] = np.nan

                # Evaluate the model 
                daily_loss = np.nanmean((predictions_new_60_reshaped - y_new_12)**2)
                loss_values.append(daily_loss)

        loss = np.mean(loss_values)

        return loss