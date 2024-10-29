import os
import sys
import json
from Oja import Oja  # Import the Oja class from the oja.py script.
from utils import read_input_normalize  # Import the function to read and normalize data from utils.py.


def main():
    # Check if a configuration file has been provided as a command-line argument.
    if len(sys.argv) < 1:
        # If no configuration file is provided, print an error message and exit.
        print("Falta el archivo de configuración")  # "Configuration file is missing"
        exit(1)

    # Open the configuration file passed as the first command-line argument.
    with open(f"{sys.argv[1]}", "r") as file:
        config = json.load(file)  # Load the configuration in JSON format.

        # Read and normalize the input data from the CSV file specified in the configuration.
        data, dimension, names = read_input_normalize(config["input"])
        n = config["eta"]  # Learning rate (eta) value from the config.
        limit = config["limit"]  # Number of training iterations from the config.

        # Initialize an Oja model with the provided learning rate and input data.
        oja = Oja(eta_0=n, data=data)

        # Train the Oja model using the specified number of iterations (limit).
        eigenvector = oja.train(limit=limit)

        # Print the first principal component eigenvector (PC1) with values rounded to 3 decimal places.
        eigenvector_rounded = eigenvector.round(3).tolist()
        print(f"El autovector de PC1 es: {eigenvector_rounded}")  # "The eigenvector for PC1 is:"

         # Save the result to a file in the 'outputs' folder
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)  # Create the 'outputs' folder if it doesn't exist
        output_file_path = os.path.join(output_dir, "PC1_eigenvector.txt")

        with open(output_file_path, "w") as output_file:
            output_file.write(f"El autovector de PC1 es: {eigenvector_rounded}\n")
    
        print(f"PC1 eigenvector saved to {output_file_path}")


# Run the main function if this script is executed from the terminal.
if __name__ == "__main__":
    main()