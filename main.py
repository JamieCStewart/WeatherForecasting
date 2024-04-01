# main.py
#
# Author: Jamie Stewart 
# Date: February - March 2024
# Description: A Python script for a masters project titled "Learning the 
# environment deeply". In particular, I am building a neural network to 
# downscale low resolution forecasts

import data
from model import create_model, compile_model

def main():
    X_paths, y_paths = data.generate_rainfall_paths(2010, 2019, 60, 12)

    X, y = data.load_data(X_paths, y_paths)

    X_train, X_test, y_train, y_test = data.test_train_split(X, y, 0.2, random_state=10)

    # Define parameters
    input_shape = X_train_flat.shape[1:]
    output_shape = y_train.shape[1:]

    # Create and compile Model 1
    model1 = create_model(input_shape, output_shape, model_type=1)
    model1 = compile_model(model1)

    # Train Model 1
    history1 = model1.fit(X_train_flat, y_train_masked.reshape((y_train_masked.shape[0], -1)), epochs=5, batch_size=32, validation_split=0.1)

    # Evaluate Model 1
    test_loss1 = model1.evaluate(X_test_flat, y_test.reshape((y_test.shape[0], -1)))
    print(f'Test Loss for Model 1: {test_loss1}')

    # Make predictions for Model 1
    predictions1 = model1.predict(X_test_flat)
    predictions1_reshaped = predictions1.reshape((predictions1.shape[0], y_test.shape[1], y_test.shape[2]))

    # Create and compile Model 2
    model2 = create_model(input_shape, output_shape, model_type=2)
    model2 = compile_model(model2)

    # Train Model 2
    history2 = model2.fit(X_train_flat, y_train_masked.reshape((y_train_masked.shape[0], -1)), epochs=5, batch_size=32, validation_split=0.1)

    # Evaluate Model 2
    test_loss2 = model2.evaluate(X_test_flat, y_test.reshape((y_test.shape[0], -1)))
    print(f'Test Loss for Model 2: {test_loss2}')

    # Make predictions for Model 2
    predictions2 = model2.predict(X_test_flat)
    predictions2_reshaped = predictions2.reshape((predictions2.shape[0], y_test.shape[1], y_test.shape[2]))




if __name__ == "__main__":
    main()



