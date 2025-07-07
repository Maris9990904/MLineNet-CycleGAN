'''结构'''
# import matplotlib.pyplot as plt
# import numpy as np

# # Data from your table
# methods = ["MLineNet", "CLAHE+BF+MLineNet", "CycleGAN+MLineNet", "Ours"]
# metrics = ["LDP", "TC", "LS", "Q"]

# # Corresponding values
# data = [
#     [84.91, 88.54, 77.91, 83.79],
#     [93.17, 79.52, 87.05, 86.58],
#     [83.63, 91.82, 80.42, 85.29],
#     [90.89, 87.78, 89.10, 89.26]
# ]

# # Normalize the values to range [0, 1] for radar chart
# max_value = 100  # Assuming the values are percentages
# normalized_data = [[v / max_value for v in row] for row in data]

# # Compute angle of each axis
# num_vars = len(metrics)
# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# angles += angles[:1]  # Complete the loop

# # Function to plot radar chart for each method
# def plot_radar_chart(ax, values, label, color):
#     values += values[:1]  # Complete the loop
#     ax.plot(angles, values, color=color, linewidth=2, label=label)

# # Initialize the radar chart without the outer frame
# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
# ax.spines['polar'].set_visible(False)

# # Set the limits for the y-axis
# ax.set_ylim([0, 1])

# # Draw one axe per variable and add labels
# ax.set_theta_offset(np.pi / 2)
# ax.set_theta_direction(-1)
# plt.xticks(angles[:-1], metrics, fontsize=12, fontweight='bold')

# # Customize the y-ticks and ensure five circles
# ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
# ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
# ax.set_rscale('linear')  # Ensure the scaling is linear

# # Plot each method's radar chart
# colors = ['blue', 'green', 'yellow', 'red']
# for i, method in enumerate(methods):
#     values = normalized_data[i]
#     plot_radar_chart(ax, values, method, colors[i])

# # Add a legend with color information
# plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10, frameon=False)

# # Display the radar chart
# plt.title("Radar Chart of Different Methods", fontsize=14, fontweight='bold')
# plt.show()

'''敦煌'''
# import matplotlib.pyplot as plt
# import numpy as np

# # Data from your table
# methods = ["Reference", "Canny", "Sobel", "DiffusionEdge", "Teed", "Ours"]
# metrics = ["LDP", "TC", "LS", "Q"]

# # Corresponding values
# data = [
#     [86.84, 89.79, 86.30, 87.64],
#     [86.57, 70.43, 76.70, 77.90],
#     [84.28, 78.00, 72.07, 78.12],
#     [81.36, 92.43, 85.27, 86.35],
#     [88.83, 88.48, 87.84, 88.38],
#     [90.89, 87.78, 89.10, 89.26]
# ]

# # Normalize the values to range [0, 1] for radar chart
# max_value = 100  # Assuming the values are percentages
# normalized_data = [[v / max_value for v in row] for row in data]

# # Compute angle of each axis
# num_vars = len(metrics)
# angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
# angles += angles[:1]  # Complete the loop

# # Function to plot radar chart for each method
# def plot_radar_chart(ax, values, label, color):
#     values += values[:1]  # Complete the loop
#     ax.plot(angles, values, color=color, linewidth=2, label=label)

# # Initialize the radar chart without the outer frame
# fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
# ax.spines['polar'].set_visible(False)

# # Set the limits for the y-axis
# ax.set_ylim([0, 1])

# # Draw one axe per variable and add labels
# ax.set_theta_offset(np.pi / 2)
# ax.set_theta_direction(-1)
# plt.xticks(angles[:-1], metrics, fontsize=12, fontweight='bold')

# # Customize the y-ticks and ensure five circles
# ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
# ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
# ax.set_rscale('linear')  # Ensure the scaling is linear
# # ['blue', 'green', 'yellow', 'red']
# # Plot each method's radar chart
# colors = ['gray', 'magenta', 'blue', 'green', 'yellow', 'red']
# for i, method in enumerate(methods):
#     values = normalized_data[i]
#     plot_radar_chart(ax, values, method, colors[i])

# # Add a legend with color information at lower left
# plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10, frameon=False)

# # Display the radar chart
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Data from your table
methods = [ "Canny", "Sobel", "DiffusionEdge", "Teed", "Ours"]
metrics = ["LDP", "TC", "LS", "Q"]

# Corresponding values
data = [
    [88.49, 73.07, 79.96, 80.51],
    [85.29, 77.76, 76.57, 79.87],
    [86.37, 95.48, 85.63, 89.16],
    [90.69, 89.03, 88.41, 89.38],
    [92.56, 88.09, 90.85, 90.50]
]

# Normalize the values to range [0, 1] for radar chart
max_value = 100  # Assuming the values are percentages
normalized_data = [[v / max_value for v in row] for row in data]

# Compute angle of each axis
num_vars = len(metrics)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]  # Complete the loop

# Function to plot radar chart for each method
def plot_radar_chart(ax, values, label, color):
    values += values[:1]  # Complete the loop
    ax.plot(angles, values, color=color, linewidth=2, label=label)

# Initialize the radar chart without the outer frame
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.spines['polar'].set_visible(False)

# Set the limits for the y-axis
ax.set_ylim([0, 1])

# Draw one axe per variable and add labels
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], metrics, fontsize=12, fontweight='bold')

# Customize the y-ticks and ensure five circles
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)
ax.set_rscale('linear')  # Ensure the scaling is linear
# ['blue', 'green', 'yellow', 'red']
# Plot each method's radar chart
colors = [ 'magenta', 'blue', 'green', 'yellow', 'red']
for i, method in enumerate(methods):
    values = normalized_data[i]
    plot_radar_chart(ax, values, method, colors[i])

# Add a legend with color information at lower left
plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10, frameon=False)

# Display the radar chart
plt.show()

