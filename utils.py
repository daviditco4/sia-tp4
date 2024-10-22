import json

import pandas as pd
from sklearn.preprocessing import StandardScaler


# Load configuration from JSON file
def load_config(json_file):
    with open(json_file, 'r') as file:
        config = json.load(file)
    return config


# Load the CSV file containing unstandardized data
def load_csv_data(csv_file):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(csv_file)

    # Remove the first column (by index)
    data = data.drop(data.columns[0], axis=1)

    return data.values


# Standardize the input data (mean=0, variance=1)
def standardize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)
