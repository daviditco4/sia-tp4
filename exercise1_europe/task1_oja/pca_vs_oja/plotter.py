import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import read_input_normalize

# Assuming `data` is your normalized dataset, and `pc1_vector` is the PC1 eigenvector from Oja's rule
# Load your normalized data and PC1 eigenvector, which you likely have from earlier steps.
data, _, names = read_input_normalize("data/europe.csv")  # Modify path as needed

#PC1 eigenvector from oja rules
pc1_vector = np.array([0.125, -0.5, 0.407, -0.483, 0.188, -0.476, 0.272])

# Project data onto PC1
pc1_projection = np.dot(data, pc1_vector)

# Create scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(pc1_projection, np.zeros_like(pc1_projection), alpha=0.6)  # Project onto PC1 axis

# Add country names as annotations (optional)
for i, name in enumerate(names):
    plt.annotate(name, (pc1_projection[i], 0), rotation=45, ha="right")

plt.xlabel("Projection on First Principal Component (PC1)")
plt.title("Scatter Plot of Data Projected onto PC1")
plt.show()