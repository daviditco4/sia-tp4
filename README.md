# sia-tp4
Implementation of unsupervised learning algorithms (Kohonen, Oja &amp; Hopfield) built with Python



# Task 1.2 - First Principal Component using Oja’s Rule

This task involves calculating the first principal component (PC1) for a dataset of European countries using Oja’s rule. The goal is to understand the main variance in the data by obtaining the eigenvector (autovector) that represents PC1, and then visualizing the projections of each country onto this component.

## Structure of Task 1.2

	1.	Main Script: exercise1_europe/task1_oja/main.py
	•	This script executes the Oja algorithm and calculates the eigenvector for PC1.
	•	The calculated eigenvector is used to project each country in the dataset onto the PC1 axis.

	2.	Configuration File: configs/oja_config.json
	•	Contains the settings required for running the main script, including:
	•	The path to the dataset (europe.csv).
	•	The learning rate (eta) for the algorithm.
	•	The number of training iterations (limit).

	3.	Plotting Script: exercise1_europe/task1_oja/barplot.py
	•	This script generates a bar plot of the country projections on PC1.
	•	The projections are calculated using the eigenvector from Oja’s rule, and the plot visualizes each country’s position along the principal component.

# Running the Main Script

To run the main script and calculate the eigenvector for PC1, use the following command in the terminal:
--
python3 main.py configs/oja_config.json
--
This will output the eigenvector (PC1) calculated using Oja’s rule.

# Generating the Bar Plot

After running the main script, you can visualize the projections of each country onto PC1 by running the barplot.py script:
--
python3 barplot.py configs/plot_config.json
--
This script will create a bar plot with the projections of each country on PC1, allowing you to visually interpret how each country aligns with the main dimension of variance in the dataset.