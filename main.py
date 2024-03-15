# main.py
#
# Author: Jamie Stewart 
# Date: February - March 2024
# Description: A Python script for a masters project titled "Learning the 
# environment deeply". In particular, I am building a neural network to 
# downscale low resolution forecasts

import data

def main():
    X_paths, y_paths = data.generate_rainfall_paths(2010, 2019, 5, 1)

    X, y = data.load_data(X_paths, y_paths)

    X_train, X_test, y_train, y_test = data.test_train_split(X, y, 0.2, random_state=10)



if __name__ == "__main__":
    main()



