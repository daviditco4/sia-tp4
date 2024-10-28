import matplotlib.pyplot as plt
import numpy as np
from utils import read_input_normalize  # Assuming `read_input_normalize` is in utils.py

# Load and normalize data
data, _, names = read_input_normalize("data/europe.csv")  # Adjust path as needed

# PC1 eigenvector from Oja's rule
pc1_vector = np.array([0.125, -0.5, 0.407, -0.483, 0.188, -0.476, 0.272])

# Project data onto PC1
pc1_projection = np.dot(data, pc1_vector)

# Sort projections and names for better readability in the bar plot
sorted_indices = np.argsort(pc1_projection)
sorted_projections = pc1_projection[sorted_indices]
sorted_names = [names[i] for i in sorted_indices]

# Create bar plot
plt.figure(figsize=(12, 8))
plt.barh(sorted_names, sorted_projections, color="skyblue")
plt.xlabel("Projection on First Principal Component (PC1)")
plt.title("Bar Plot of Data Projected onto PC1")
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.show()