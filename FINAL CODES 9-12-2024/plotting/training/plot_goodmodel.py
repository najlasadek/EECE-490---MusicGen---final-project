import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Paths to the three folders
folder_paths = [
    '256_2_full_data_seq128_dropout_0.2',
    '256_2_full_data_seq64_dropout_0.2',
    '256_2_full_data_seq32_dropout_0.2'
]

# Labels for the plots corresponding to each folder
labels = ['seq_len128', 'seq_len64', 'seq_len32']

# Colors for each plot (optional)
colors = ['#800020', '#8A2BE2', '#008080']

# Debugging: Check folder contents
for folder_path in folder_paths:
    if os.path.exists(folder_path):
        print(f"Folder '{folder_path}' exists. Contents:")
        print(os.listdir(folder_path))
    else:
        print(f"Folder '{folder_path}' does not exist.")

# Plotting
plt.figure(figsize=(14, 8))

for folder_path, label, color in zip(folder_paths, labels, colors):
    epochs = []
    losses = []

    if os.path.exists(folder_path):
        # Loop through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.pth'):
                # Extract epoch and loss from filename using regex
                match = re.search(r'epoch(\d+)_loss([\d.]+)', filename)
                if match:
                    epoch = int(match.group(1))
                    loss = float(match.group(2).strip('.'))
                    epochs.append(epoch)
                    losses.append(loss)

        # Sort epochs and losses by epoch number
        if epochs and losses:
            sorted_data = sorted(zip(epochs, losses))
            epochs, losses = zip(*sorted_data)

            # Plot each folder's data
            plt.plot(epochs, losses, marker='o', linestyle='-', markersize=4, label=label, color=color)

# Define uniform ticks for x-axis at fixed intervals
if epochs:  # Avoid max() errors if data is missing
    x_ticks = np.arange(0, max(epochs) + 10, 10)  # Adjust interval as needed
else:
    x_ticks = []

# Define uniform ticks for y-axis at fixed intervals
y_step = 0.01  # Set the step size for y-axis
y_ticks = np.arange(0, 0.2 + y_step, y_step)  # Adjusted y range for realistic data

# Customize the plot
plt.title(
    'Loss vs Epochs \n Full Data,Model 2x256, Batch Size 64, LR 0.001, Dropout 0.2,\n Varying Sequence Length',
    fontsize=16
)

plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xticks(ticks=x_ticks, fontsize=12, rotation=45)  # Uniform x-axis labels
plt.yticks(ticks=y_ticks, fontsize=12)  # Uniform y-axis labels
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()  # Turn on minor ticks for finer granularity
plt.legend(fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title and subtitle
plt.show()
