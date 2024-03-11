# main.py
#
# Author: Jamie Stewart 
# Date: February - March 2024
# Description: A Python script for a masters project titled "Learning the 
# environment deeply". In particular, I am building a neural network to 
# downscale low resolution forecasts

import data

def main():
    X_paths, y_paths = data.generate_rainfall_paths(2021, 2022, 60, 12)

    X, y = data.load_data(X_paths, y_paths)



if __name__ == "__main__":
    main()



