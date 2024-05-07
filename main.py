# main.py
#
# Author: Jamie Stewart 
# Date: February - March 2024
# Description: A Python script for a masters project titled "Learning the 
# environment deeply". In particular, I am building a neural network to 
# downscale low resolution forecasts

import data
import csv 
import numpy as np 
from model import create_model, compile_model
from evaluate_model import evaluate_model
import matplotlib.pyplot as plt 
from tensorflow.keras.models import load_model 
import os 

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

    # Define file names for models
    model1_file = 'model1.h5'
    model2_file = 'model2.h5'
    model3_file = 'model3.h5'
    model4_file = 'model4.h5'

    # Check if the model files exist
    if os.path.isfile(model1_file):
        # Load Model 1 if the file exists
        model1 = load_model(model1_file)
    else:
        # Create and compile Model 1 if the file doesn't exist
        model1 = create_model(input_shape, output_shape, model_type=1)
        model1 = compile_model(model1)

    # Repeat the same process for other models
    if os.path.isfile(model2_file):
        model2 = load_model(model2_file)
    else:
        model2 = create_model(input_shape, output_shape, model_type=2)
        model2 = compile_model(model2)

    if os.path.isfile(model3_file):
        model3 = load_model(model3_file)
    else:
        model3 = create_model(input_shape, output_shape, model_type=3)
        model3 = compile_model(model3)

    if os.path.isfile(model4_file):
        model4 = load_model(model4_file)
    else:
        model4 = create_model(input_shape, output_shape, model_type=4)
        model4 = compile_model(model4)


    ### Reshape for CNN 
    #X_train_reshaped = X_train.reshape((X_train.shape[0], 23, 17, 1))
    #X_test_reshaped = X_test.reshape((X_test.shape[0], 23, 17, 1))

    # Define parameters
    #input_shape = X_train_reshaped.shape[1:]

    # Create and compile Model 4
    #model4 = create_model(input_shape, output_shape, model_type=4)
    #model4 = compile_model(model4)

    # Train Model 4
    #history4 = model4.fit(X_train_reshaped, y_train_masked.reshape((y_train_masked.shape[0], -1)), epochs=5, batch_size=32, validation_split=0.1)




    # Create a list of epochs (x-axis values)
    epochs = [5 * i for i in range(1, 101)]

    # Define the file name
    file_name = "losses_data.csv"

    # Write the initial header row to the CSV file
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epochs', 'Model 1 Loss', 'Model 2 Loss', 'Model 3 Loss','Model 4 Loss'])

    # Iterate over the training epochs
    for j in range(1, 21):
        print("EPOCH NUMBER: " + str(j))
        # Train each model for a fixed number of epochs
        model1.fit(X_train_flat, y_train_masked.reshape((y_train_masked.shape[0], -1)), epochs=5, batch_size=32, validation_split=0.1)
        model2.fit(X_train_flat, y_train_masked.reshape((y_train_masked.shape[0], -1)), epochs=5, batch_size=32, validation_split=0.1)
        model3.fit(X_train_flat, y_train_masked.reshape((y_train_masked.shape[0], -1)), epochs=5, batch_size=32, validation_split=0.1)
        model4.fit(X_train_flat, y_train_masked.reshape((y_train_masked.shape[0], -1)), epochs=5, batch_size=32, validation_split=0.1)


        model1.save('model1.h5')
        model2.save('model2.h5')
        model3.save('model3.h5')
        model4.save('model4.h5')

        # Initialize variables to accumulate losses over all months
        total_loss_model1 = 0
        total_loss_model2 = 0
        total_loss_model3 = 0
        total_loss_model4 = 0

        # Iterate over all months
        for i in range(1, 13):
            # Evaluate each model for the current month
            loss_model1 = evaluate_model(model1, 60, 12, 2020, i)
            loss_model2 = evaluate_model(model2, 60, 12, 2020, i)
            loss_model3 = evaluate_model(model3, 60, 12, 2020, i)
            loss_model4 = evaluate_model(model4, 60, 12, 2020, i)

            # Accumulate losses for each model
            total_loss_model1 += loss_model1
            total_loss_model2 += loss_model2
            total_loss_model3 += loss_model3
            total_loss_model4 += loss_model4

        # Calculate annual average losses for each model
        annual_loss_model1 = total_loss_model1 / 12
        annual_loss_model2 = total_loss_model2 / 12
        annual_loss_model3 = total_loss_model3 / 12
        annual_loss_model4 = total_loss_model4 / 12


        # Append the losses for this epoch to the CSV file
        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([j, annual_loss_model1, annual_loss_model2, 
            annual_loss_model3, annual_loss_model4])


        

    

    





if __name__ == "__main__":
    main()



