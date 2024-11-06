import math
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
    def __init__(self, width, height, input_dim, radius, learning_rate, distance_method='euclidean', iterations=1000,
                 decay_rate=0.0):
        self.width = width
        self.height = height
        self.radius = radius
        self.initial_radius = radius  # Save the initial radius for shrinking
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.input_dim = input_dim
        self.decay_rate = decay_rate

        # Initialize weights randomly
        self.weights = np.random.rand(width, height, input_dim)

        # Matrix to store lists of assigned data point indices for each neuron
        self.final_assignments = [[[] for _ in range(height)] for _ in range(width)]

        # Dictionary to store the most recent assignment of each data point
        self.latest_assignments = {}

        # Choose the distance method (Euclidean by default)
        if distance_method == 'euclidean':
            self.distance_method = euclidean
        else:
            raise ValueError(f"Distance method '{distance_method}' not supported.")

    def train(self, data):
        num_samples = data.shape[0]

        for it in range(self.iterations):
            # Reduce learning rate over time
            learning_rate = self.learning_rate * (1 - it / self.iterations)

            # Optionally shrink the radius towards 1 if `decay_rate` is positive
            radius = 1 + (self.initial_radius - 1) * math.exp(-self.decay_rate * it)

            # Randomly select an input vector
            random_idx = np.random.randint(0, num_samples)
            input_vector = data[random_idx]

            # Find the best matching unit (BMU)
            _, bmu_idx = self.find_bmu(input_vector)

            # Update the BMU and its neighbors
            self.update_weights(input_vector, bmu_idx, learning_rate, radius)

            # Track the latest assignment of each data point
            self.latest_assignments[random_idx] = bmu_idx  # Map the data point index to the BMU

        # Update `final_assignments` with only the last known assignment for each data point
        for data_idx, neuron_pos in self.latest_assignments.items():
            if data_idx not in self.final_assignments[neuron_pos[0]][neuron_pos[1]]:
                self.final_assignments[neuron_pos[0]][neuron_pos[1]].append(data_idx)

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

    def calculate_u_matrix(self):
        """Calculate the Unified Distance Matrix (U-Matrix)."""
        u_matrix = np.zeros((self.width, self.height))

        for i in range(self.width):
            for j in range(self.height):
                neighbors = []

                # Get neighbors (within grid bounds)
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if (di != 0 or dj != 0) and (0 <= i + di < self.width) and (0 <= j + dj < self.height):
                            neighbors.append(self.weights[i + di, j + dj])

                # Calculate the average distance to all neighbors
                if neighbors:
                    u_matrix[i, j] = np.mean(
                        [self.distance_method(self.weights[i, j], neighbor) for neighbor in neighbors])

        return u_matrix

    def plot_u_matrix(self):
        """Plot the Unified Distance Matrix (U-Matrix)."""
        u_matrix = self.calculate_u_matrix()

        plt.figure(figsize=(10, 7))
        sns.heatmap(u_matrix, cmap='plasma', cbar=True)
        plt.title('Unified Distance Matrix (U-Matrix)')
        plt.xlabel('Neuron X')
        plt.ylabel('Neuron Y')
        plt.savefig("kohonen_u_matrix2.png", dpi=300, bbox_inches='tight')
        plt.close()

    def plot_heatmap(self, labels):
        """Plot a heatmap with concatenated labels for final assignments per neuron."""
        label_matrix = np.empty((self.width, self.height), dtype=object)
        amount_matrix = np.zeros((self.width, self.height))

        # Fill the label matrix with concatenated labels of data points assigned to each neuron
        for i in range(self.width):
            for j in range(self.height):
                data_indices = self.final_assignments[i][j]
                if data_indices:  # If there are data points assigned
                    label_matrix[i, j] = '\n'.join([labels[idx] for idx in data_indices])
                    amount_matrix[i, j] = len(data_indices)
                else:
                    label_matrix[i, j] = ''  # No assignment for this neuron

        plt.figure(figsize=(10, 7))
        sns.heatmap(amount_matrix, annot=label_matrix, fmt='', cmap='coolwarm', cbar=False)
        plt.title('Heatmap of Final Neuron Assignments with Labels')
        plt.xlabel('Neuron X')
        plt.ylabel('Neuron Y')
        plt.savefig("kohonen_heatmap2.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Load configuration from JSON file
    config = load_config('exercise1_europe/task1_kohonen/configs/prototype.json')

    # Load unstandardized data from CSV file
    labels, data = load_csv_data('exercise1_europe/task1_kohonen/data/europe.csv')

    # Standardize the data
    standardized_data = standardize_data(data)

    # Initialize Kohonen SOM with configuration from JSON
    som = KohonenSOM(width=config['width'],
                     height=config['height'],
                     input_dim=standardized_data.shape[1],
                     radius=config['radius'],
                     learning_rate=config['learning_rate'],
                     distance_method=config['distance_method'],
                     iterations=config['iterations'],
                     decay_rate=config['decay_rate'])

    # Train the SOM with the standardized data
    som.train(standardized_data)

    # Plot heatmap of final assignments with labels
    som.plot_heatmap(labels)

    # Plot the U-Matrix
    som.plot_u_matrix()
