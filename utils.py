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
    data = pd.read_csv(csv_file)
    return data[data.columns[0]], data.drop(data.columns[0], axis=1).to_numpy()


# Standardize the input data (mean=0, variance=1)
def standardize_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data)
