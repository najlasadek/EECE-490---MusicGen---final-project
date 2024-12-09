import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Paths to the three folders
folder_paths = [
    'saved_weights',
    'training_again_half_data_seq64_dropout_0.2',
    'training_again_quarter_data_seq64_dropout_0.2',
]

# Labels for the plots corresponding to each folder
labels = ['full_data', 'half_data', 'quarter_data']

# Colors for each plot (optional)
colors = ['#FF5733', '#33FF57', '#3357FF']  # Orange, Green, Blue

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

# Define uniform ticks for y-axis with finer limits
y_step = 0.005  # Adjusted for finer granularity
y_ticks = np.arange(0, 0.15 + y_step, y_step)  # Adjusted y range to end at 0.15

# Customize the plot
plt.title(
    'Loss vs Epochs \n Model 4x512, Batch Size 64, LR 0.001, Dropout 0.2, Sequence Length = 64\n Varying Data Size',
    fontsize=16
)

plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xticks(ticks=x_ticks, fontsize=12, rotation=45)  # Uniform x-axis labels
plt.yticks(ticks=y_ticks, fontsize=12)  # Uniform y-axis labels
plt.ylim(0, 0.15)  # Set y-axis limit to zoom in
plt.grid(which='both', linestyle='--', linewidth=0.5)
plt.minorticks_on()  # Turn on minor ticks for finer granularity
plt.legend(fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit the title and subtitle
plt.show()
