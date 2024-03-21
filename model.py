# model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Masking
from tensorflow.keras.optimizers import Adam

def create_model(input_shape, output_shape, model_type=1):
    if model_type == 1:
        model = create_model1(input_shape, output_shape)
    elif model_type == 2:
        model = create_model2(input_shape, output_shape)
    else:
        raise ValueError("Invalid model_type. Supported values are 1 and 2.")
    return model

def create_model1(input_shape, output_shape):
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(np.prod(output_shape), activation='linear'))
    return model

def create_model2(input_shape, output_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(np.prod(output_shape), activation='linear'))
    return model

def compile_model(model, optimizer='adam', learning_rate=0.001, loss='mean_squared_error'):
    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer.")
    model.compile(optimizer=optimizer, loss=loss)
    return model

# You can define other functions for training, evaluation, and prediction if needed
