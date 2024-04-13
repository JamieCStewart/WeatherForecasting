import calendar 

def evaluate_model(model1, low_res, high_res, year, month):
        last_day = calendar.monthrange(year, int(month))[1]
        X_file_path = f"Data/{low_res}km/Rainfall/rainfall_hadukgrid_uk_{low_res}km_day_{year}{month}01-{year}{month}{last_day}.nc"
                
        new_12 = f"Data/12km/Rainfall/rainfall_hadukgrid_uk_12km_day_{year}{month}01-{year}{month}{last_day}.nc"
        new_60 = f"Data/60km/Rainfall/rainfall_hadukgrid_uk_60km_day_{year}{month}01-{year}{month}{last_day}.nc"

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

        X_new_60_flat = X_new_60_flat.reshape((X_new_60_flat.shape[0], X_test.shape[1], X_test.shape[2]))

        # Apply the NaN mask to predictions to match the pattern in y_train
        predictions_new_60_reshaped[0][nan_mask_y_train[0]] = np.nan

        # Evaluate the model 
        loss = np.nanmean((predictions_new_60_reshaped[0] - y_new_12)**2)

        return loss