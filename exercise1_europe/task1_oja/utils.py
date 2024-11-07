import pandas as pd
import numpy as np

def read_input_normalize(input_file_path):
    # Read the CSV file, skipping the first rows
    df = pd.read_csv(input_file_path, header=None, skiprows=1)
    
    # Normalize each column except the first (which is assumed to be country names)
    for column in df.columns:
        if column != 0:
            # Subtract the mean from each value and divide by the standard deviation to normalize
            aux = df[column] - df[column].mean()
            df[column] = aux / df[column].std()
    
    # Convert the DataFrame to a numpy array for further processing
    aux = df.to_numpy()
    
    # Remove the first column (country names) from the numpy array
    aux = np.delete(aux, 0, axis=1)
    
    # Read the country names again from the CSV file
    names = pd.read_csv(input_file_path)
    names = names['Country']
    
    # Return the normalized data, the number of features (columns), and the country names
    return aux, len(aux[0]), names