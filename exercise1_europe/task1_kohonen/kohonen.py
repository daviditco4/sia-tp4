import numpy as np
import pandas as pd
import json
from scipy.spatial.distance import euclidean


# Load configuration from JSON file
def load_config(json_file):
    with open(json_file, 'r') as file:
        config = json.load(file)
    return config


# Kohonen Self-Organizing Map class
class KohonenSOM:
    def __init__(self, width, height, input_dim, radius, learning_rate, distance_method='euclidean', iterations=1000):
        self.width = width
        self.height = height
        self.radius = radius
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.input_dim = input_dim

        # Initialize weights randomly
        self.weights = np.random.rand(width, height, input_dim)

        # Choose the distance method (Euclidean by default)
        if distance_method == 'euclidean':
            self.distance_method = euclidean
        else:
            raise ValueError(f"Distance method '{distance_method}' not supported.")

    def train(self, data):
        for it in range(self.iterations):
            # Reduce learning rate and radius over time
            learning_rate = self.learning_rate * (1 - it / self.iterations)
            radius = self.radius * (1 - it / self.iterations)

            # Randomly select an input vector
            random_idx = np.random.randint(0, data.shape[0])
            input_vector = data[random_idx]

            # Find the best matching unit (BMU)
            bmu, bmu_idx = self.find_bmu(input_vector)

            # Update the BMU and its neighbors
            self.update_weights(input_vector, bmu_idx, learning_rate, radius)

    def find_bmu(self, input_vector):
        """Find the Best Matching Unit (BMU) for a given input vector."""
        min_dist = float('inf')
        bmu_idx = None
        for i in range(self.width):
            for j in range(self.height):
                dist = self.distance_method(input_vector, self.weights[i, j])
                if dist < min_dist:
                    min_dist = dist
                    bmu_idx = (i, j)
        return self.weights[bmu_idx], bmu_idx

    def update_weights(self, input_vector, bmu_idx, learning_rate, radius):
        """Update the weights of the neurons around the BMU."""
        for i in range(self.width):
            for j in range(self.height):
                dist_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_idx))
                if dist_to_bmu <= radius:
                    weight_update = learning_rate * (input_vector - self.weights[i, j])
                    self.weights[i, j] += weight_update

    def get_weights(self):
        return self.weights


# Example usage with a CSV file containing unstandardized data
def load_csv_data(csv_file):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(csv_file)

    # Remove the first column (by index)
    data = data.drop(data.columns[0], axis=1)

    return data.values


if __name__ == "__main__":
    # Load configuration from JSON file
    config = load_config('exercise1_europe/task1_kohonen/configs/prototype.json')

    # Load unstandardized data from CSV file
    data = load_csv_data('exercise1_europe/task1_kohonen/data/europe.csv')

    # Initialize Kohonen SOM with configuration from JSON
    som = KohonenSOM(width=config['width'],
                     height=config['height'],
                     input_dim=data.shape[1],
                     radius=config['radius'],
                     learning_rate=config['learning_rate'],
                     distance_method=config['distance_method'],
                     iterations=config['iterations'])

    # Train the SOM with the loaded data
    som.train(data)

    # Print final weights
    print("Final SOM Weights:")
    print(som.get_weights())
