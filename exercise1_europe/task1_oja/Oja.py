import numpy as np
from numpy import ndarray

# Initialize the weights randomly between 0 and 1 for the input dimensions.
def _initialize_weights(dim: int) -> ndarray:
    return np.random.uniform(0, 1, dim)

# Define the Oja class, which implements Oja's learning rule.
class Oja:

    # Initialize the class with learning rate (eta_0) and input data.
    def __init__(self, eta_0: float, data: ndarray):
        self._eta_0: float = eta_0  # Initial learning rate.
        self._data: ndarray = data  # Input data for training.
        # Initialize the weights with random values for each feature in the data.
        self.weights: ndarray = _initialize_weights(len(data[0]))
        self._epoch: int = 0  # Track the number of training iterations (epochs).
        self._eta: float = eta_0  # Current learning rate, initialized as eta_0.

    # Update the learning rate, which decreases over time based on the number of epochs.
    def _update_eta(self) -> None:
        self._epoch += 1  # Increment the epoch count.
        self._eta = self._eta_0 / self._epoch  # Reduce the learning rate over time.

    # Compute the change in weights based on Oja's rule.
    def _delta_weights(self, x: ndarray, output: float) -> ndarray:
        # Oja's learning rule: delta_w = eta * output * (x - output * weights)
        return (self._eta * output * (x - output * self.weights)).astype(float)

    # Train the Oja network for a given number of iterations (limit).
    # Optionally, pass a callback (on_epoch) to track weights after each epoch.
    def train(self, limit: int = 100, on_epoch: callable = None) -> ndarray:
        for i in range(limit):
            # Call the on_epoch callback (if provided) to track progress after each epoch.
            if on_epoch is not None:
                on_epoch(i, self.weights.copy())

            # Update the learning rate after each epoch.
            self._update_eta()

            # Loop through each data point in the dataset.
            for u in self._data:
                # Calculate the output by taking the dot product of the input and the weights.
                output = np.dot(u, self.weights)

                # Calculate the weight update using Oja's rule.
                delta_w = self._delta_weights(u, output)

                # Update the weights by adding the change calculated from Oja's rule.
                self.weights = self.weights + delta_w

        # Return the final weights after training is complete.
        return self.weights.copy()