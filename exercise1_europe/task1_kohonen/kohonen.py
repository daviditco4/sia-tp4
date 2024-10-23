import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import euclidean
import seaborn as sns

# Add the path to the folder containing utils.py
sys.path.append(os.path.abspath("."))

# Now you can import load_config, load_csv_data, standardize_data
from utils import load_config, load_csv_data, standardize_data


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

        # Initialize grid to count data points assigned to each neuron (for the heatmap)
        self.assignment_counts = np.zeros((width, height))

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

            # Increment the assignment count for the BMU
            self.assignment_counts[bmu_idx] += 1

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

    def plot_heatmap(self):
        """Plot a heatmap of the number of assignments per neuron with integer formatting."""
        plt.figure(figsize=(10, 7))

        # Create the heatmap and format annotations as integers
        sns.heatmap(self.assignment_counts,
                    annot=True,
                    cmap='coolwarm',
                    cbar=True,
                    fmt='g',  # 'g' means general format (no scientific notation)
                    annot_kws={"size": 12, "weight": "bold", "color": "black"})  # Customize annotation appearance

        plt.title('Heatmap of Neuron Assignments')
        plt.xlabel('Neuron X')
        plt.ylabel('Neuron Y')
        plt.show()


if __name__ == "__main__":
    # Load configuration from JSON file
    config = load_config('exercise1_europe/task1_kohonen/configs/prototype.json')

    # Load unstandardized data from CSV file
    data = load_csv_data('exercise1_europe/data/europe.csv')

    # Standardize the data
    standardized_data = standardize_data(data)

    # Initialize Kohonen SOM with configuration from JSON
    som = KohonenSOM(width=config['width'],
                     height=config['height'],
                     input_dim=standardized_data.shape[1],
                     radius=config['radius'],
                     learning_rate=config['learning_rate'],
                     distance_method=config['distance_method'],
                     iterations=config['iterations'])

    # Train the SOM with the standardized data
    som.train(standardized_data)

    # Plot heatmap of assignments per neuron
    som.plot_heatmap()
