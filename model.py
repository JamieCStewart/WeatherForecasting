# model.py
import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Masking, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

def create_model(input_shape, output_shape, model_type=1):
    if model_type == 1:
        model = create_model1(input_shape, output_shape)
    elif model_type == 2:
        model = create_model2(input_shape, output_shape)
    elif model_type == 3:
        model = create_model3(input_shape, output_shape)
    elif model_type == 4:
        model = create_model4(input_shape, output_shape)
    elif model_type == 5:
        model = create_model5(input_shape, output_shape)
    else:
        raise ValueError("Invalid model_type."
                         "Supported values are 1,2,3,4 or 5.")

def create_model1(input_shape, output_shape):
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(np.prod(output_shape), activation='linear'))
    return model

def create_model2(input_shape, output_shape):
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(np.prod(output_shape), activation='linear'))
    return model

def create_model3(input_shape, output_shape):
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(np.prod(output_shape), activation='linear'))
    return model

def create_model4(input_shape, output_shape):
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(np.prod(output_shape), activation='linear'))
    return model

def create_model5(input_shape, output_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
                    input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), 
                    activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(np.prod(output_shape), activation='linear'))
    return model

def compile_model(model, optimizer='adam', 
    learning_rate=0.001, loss='mean_squared_error'):

    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer.")
    model.compile(optimizer=optimizer, loss=loss)
    return model
