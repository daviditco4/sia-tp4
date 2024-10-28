import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from utils import read_input_normalize  # Import your function from utils.py


def main():
    # Check if a configuration file has been provided as a command-line argument.
    if len(sys.argv) < 2:
        print("Configuration file is missing")  # Notify if no config file is provided.
        exit(1)

    # Load the configuration file provided as the first command-line argument.
    with open(f"{sys.argv[1]}", "r") as file:
        config = json.load(file)  # Load the config settings.

    # Read and normalize the input data from the CSV file specified in the configuration.
    data, _, names = read_input_normalize(config["input"])

    # Retrieve the PC1 eigenvector from the configuration.
    pc1_vector = np.array([0.125, -0.5, 0.407, -0.483, 0.188, -0.476, 0.272])

    # Project data onto PC1
    pc1_projection = np.dot(data, pc1_vector)

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(pc1_projection, np.zeros_like(pc1_projection), alpha=0.6)  # Project onto PC1 axis

    # Add country names as annotations (optional)
    for i, name in enumerate(names):
        plt.annotate(name, (pc1_projection[i], 0), rotation=45, ha="right", xytext=(0, 5), textcoords="offset points")

    plt.xlabel("Projection on First Principal Component (PC1)")
    plt.title("Scatter Plot of Data Projected onto PC1")
    plt.show()


# Run the main function if this script is executed from the terminal.
if __name__ == "__main__":
    main()