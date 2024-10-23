import os
import sys

import numpy as np

from MultilayerPerceptron import MultilayerPerceptron

# Add the path to the folder containing utils.py
sys.path.append(os.path.abspath("."))

# Now you can import load_config, load_csv_data, standardize_data
from utils import load_config, load_csv_data, standardize_data


class SimpleOjaPerceptron(MultilayerPerceptron):
    def __init__(self, input_dim, learning_rate=0.001, momentum=0.0, weight_updates_by_epoch=True):
        super().__init__([input_dim, 1], None, learning_rate, momentum, weight_updates_by_epoch)

    # Identity activation function
    def sigmoid(self, x):
        return x

    # Derivative of the identity (calculated on weighted sums, not activations)
    def sigmoid_derivative(self, x):
        return 1.0

    def initialize_weights(self):
        self.weights = [np.random.rand(self.layer_sizes[0], self.layer_sizes[1])]
        self.prev_weight_updates = [np.zeros_like(w) for w in self.weights]

    def calculate_weight_gradients(self, activations, _, x_sample):
        return -activations[-1] * (np.array([x_sample[0]]).T - activations[-1] * self.weights[-1])

    def compute_error(self, _, __):
        return 1.0


if __name__ == "__main__":
    # Load configuration from JSON file
    config = load_config('exercise1_europe/task2_oja/configs/prototype.json')

    # Load unstandardized data from CSV file
    data = load_csv_data('exercise1_europe/data/europe.csv')

    # Standardize the data
    standardized_data = standardize_data(data)

    # Initialize simple Oja perceptron with configuration from JSON
    mlp = SimpleOjaPerceptron(standardized_data.shape[1],
                              learning_rate=config['learning_rate'],
                              momentum=config['momentum'],
                              weight_updates_by_epoch=config['weight_updates_by_epoch'])

    # Train the MLP
    trained_weights, err, epochs, _, _ = mlp.train(standardized_data, np.zeros((1, standardized_data.shape[0])),
                                                   config['epochs'], 0.0)

    print("Trained weights:", trained_weights)
    print("Minimum error:", err)
    print("Epoch reached:", epochs)
