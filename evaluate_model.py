import calendar 
import xarray as xr
import numpy as np 

def evaluate_model(model, low_res, high_res, year, month):
        month = str(month).zfill(2)
        last_day = calendar.monthrange(year, int(month))[1]

        new_high_res = (
                        f"Data/{high_res}km/Rainfall/rainfall_hadukgrid_uk"
                        f"_{high_res}km_day_{year}{month}01-"
                        f"{year}{month}{last_day}.nc"
                        )
        new_low_res = (f"Data/{low_res}km/Rainfall/rainfall_hadukgrid_uk"
                       f"_{low_res}km_day_{year}{month}01-"
                       f"{year}{month}{last_day}.nc")

        loss_values = []

        for i in range(last_day): 
                time_for_image = i

                # Load and preprocess data
                with xr.open_dataset(new_low_res) as ds_new_low_res:
                        data_slice = ds_new_low_res.isel(
                                        time=time_for_image)

                        X_new_low_res = data_slice['rainfall'].values

                        nan_mask_X_new_low_res = np.isnan(X_new_low_res)

                        X_new_low_res_masked = np.where(nan_mask_X_new_low_res, 
                                                0, X_new_low_res)

                        X_new_low_res_flat = X_new_low_res_masked.reshape(
                                                (1, -1))  

                # Make predictions on the new_60 data
                predictions_new_ = model.predict(X_new_low_res_flat)

                # Load and preprocess the corresponding data for comparison
                with xr.open_dataset(new_high_res) as ds_new_high_res:
                        data_slice = ds_new_high_res.isel(time=time_for_image)
                        y_new_high_res = data_slice['rainfall'].values

                # Reshape predictions to match the original shape of y_test

                predictions_new_reshaped = predictions_new_.reshape(
                                                y_new_high_res.shape[0], 
                                                y_new_high_res.shape[1])

                # Apply the NaN mask to predictions to 
                # match the pattern in y_train
                nan_mask_y = np.isnan(y_new_high_res)
                predictions_new_reshaped[nan_mask_y] = np.nan

                # Evaluate the model 
                daily_loss = np.nanmean((predictions_new_reshaped 
                                                - y_new_high_res)**2)
                                                
                loss_values.append(daily_loss)

        loss = np.mean(loss_values)

        return loss