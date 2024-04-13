# main.py
#
# Author: Jamie Stewart 
# Date: February - March 2024
# Description: A Python script for a masters project titled "Learning the 
# environment deeply". In particular, I am building a neural network to 
# downscale low resolution forecasts

import data
import numpy as np 
from model import create_model, compile_model
from evaluate_model import evaluate_model

def main():
    X_paths, y_paths = data.generate_rainfall_paths(2000, 2019, 60, 12)

    X, y = data.load_data(X_paths, y_paths)

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

    # Define parameters
    input_shape = X_train_flat.shape[1:]
    output_shape = y_train.shape[1:]

    # Create and compile Model 1
    model1 = create_model(input_shape, output_shape, model_type=1)
    model1 = compile_model(model1)

    # Train Model 1
    history1 = model1.fit(X_train_flat, y_train_masked.reshape((y_train_masked.shape[0], -1)), epochs=5, batch_size=32, validation_split=0.1)



    ### Reshape for CNN 
    X_train_reshaped = X_train.reshape((X_train.shape[0], 23, 17, 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], 23, 17, 1))

    # Define parameters
    input_shape = X_train_reshaped.shape[1:]

    # Create and compile Model 2
    model2 = create_model(input_shape, output_shape, model_type=2)
    model2 = compile_model(model2)

    # Train Model 2
    history2 = model2.fit(X_train_reshaped, y_train_masked.reshape((y_train_masked.shape[0], -1)), epochs=5, batch_size=32, validation_split=0.1)

    # Evaluate both models on the same day
    ###  Create function 
    loss = evaluate_model(model1, 60, 12, 2020, 1)
    print(f'Test Loss: {loss}')
    

    

    





if __name__ == "__main__":
    main()



