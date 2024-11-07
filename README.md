# SIA TP4
Implementation of unsupervised learning algorithms (Kohonen, Oja &amp; Hopfield) built with Python

# Task 1.1
This task contains an implementation of a Kohonen Self-Organizing Map (SOM) in Python, a type of unsupervised learning algorithm primarily used for clustering and visualizing high-dimensional data.

## System requirements
* Python 3.7+


## Configuration
Customize the SOM parameters in prototype.json:
* width and height: Dimensions of the SOM grid.
* radius: The initial radius of the neighborhood function.
* learning_rate: The learning rate of the SOM.
* distance_method: The method used to calculate the distance between neurons.
* iterations: The number of iterations to train the SOM.
* decay_rate: The rate at which the radius decay.
* init_from_data: Whether to initialize weights based on the input data.

## How to use
 Clone or download this repository in the folder you desire
* In a new terminal, navigate to the `task1_kohonen` repository using `cd`
* When you are ready, enter a command as follows:
```sh
python3 kohonen.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

# Task 1.2 - First Principal Component using Oja’s Rule

This task involves calculating the first principal component (PC1) for a dataset of European countries using Oja’s rule. The goal is to understand the main variance in the data by obtaining the eigenvector (autovector) that represents PC1, and then visualizing the projections of each country onto this component.

## Structure of Task 1.2

1. Main Script: exercise1_europe/task1_oja/main.py
   * This script executes the Oja algorithm and calculates the eigenvector for PC1.
   * The calculated eigenvector is used to project each country in the dataset onto the PC1 axis.

2. Configuration File: configs/oja_config.json
   * Contains the settings required for running the main script, including:
   * The path to the dataset (europe.csv).
   * The learning rate (eta) for the algorithm.
   * The number of training iterations (limit).

3. Plotting Script: exercise1_europe/task1_oja/barplot.py
   * This script generates a bar plot of the country projections on PC1.
   * The projections are calculated using the eigenvector from Oja’s rule, and the plot visualizes each country’s position along the principal component.

# Running the Main Script

To run the main script and calculate the eigenvector for PC1, use the following command in the terminal:
```bash
python3 main.py configs/oja_config.json
```
This will output the eigenvector (PC1) calculated using Oja’s rule.

# Generating the Bar Plot

After running the main script, you can visualize the projections of each country onto PC1 by running the barplot.py script:
```bash
python3 barplot.py configs/plot_config.json
```
This script will create a bar plot with the projections of each country on PC1, allowing you to visually interpret how each country aligns with the main dimension of variance in the dataset.

# Task 2 - Reconstruction of 5x5 'pixelated' letters with Hopfield

This task involves using 4 patterns so that the Hopfield Model can learn from it and then you try to reconstruct altered patterns by different levels of noise, and then calculate the accuracy percentage and see and analize for spurious states.

# Running the Script

To See the "pixelated" letters run LetrasHopfield.py
To See the top and bottom 20 ranking of ortogonality run Ortogonalidad.py
To See the Energy function graphs run newHopfield.py
To See the Reconstruction of the Letters with noise and the accuracy porcentages run Hopfield.py

For both Hopfield.py and newHopfield.py to test both best and worst case of group of letters you can change:
	test_letters to letras for best case, letras_bottom for worst case
	test_patterns to patterns for best case, pattern_bottom for worst case
