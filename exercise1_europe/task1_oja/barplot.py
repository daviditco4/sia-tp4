import matplotlib.pyplot as plt
import numpy as np
from utils import read_input_normalize 

# Load and normalize data
data, _, names = read_input_normalize("data/europe.csv")  

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
bars = plt.barh(sorted_names, sorted_projections, color="skyblue")
plt.xlabel("PC1 Projection Value")
plt.title("Projection of European Countries onto PC1 Using Oja’s Rule")
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add projection values to each bar
for bar, projection in zip(bars, sorted_projections):
    plt.text(
        bar.get_width(),          # X position of the text (width of the bar)
        bar.get_y() + bar.get_height() / 2,  # Y position (centered on the bar)
        f"{projection:.2f}",      # Projection value with 2 decimal places
        va='center',              # Center vertically
        ha='left' if projection >= 0 else 'right',  # Adjust text alignment based on positive/negative
        color="black"
    )


plt.show()