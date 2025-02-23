import numpy as np
import matplotlib.pyplot as plt

# Load the npz file
file_path = "/Users/dibya/MyThesisDatasets/CHAOS_Dataset_for_Dibya/all_patient/13.npz"  # Update path if needed
data = np.load(file_path)

# Extract MRI image data and segmentation masks
image_data = data["data"]  # Shape (H, W, Slices)
mask_liver = data["mask_Liver"]
mask_rk = data["mask_RK"]
mask_lk = data["mask_LK"]
mask_spleen = data["mask_Spleen"]

# Get number of slices
num_slices = image_data.shape[2]

# Set up figure size based on number of slices
cols = 6  # Number of columns
rows = int(np.ceil(num_slices / cols))  # Calculate rows dynamically

fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
fig.suptitle("Organ Mask Overlay on MRI Slices", fontsize=16)

for i, ax in enumerate(axes.flat):
    if i < num_slices:
        ax.imshow(image_data[:, :, i], cmap="gray")  # Base MRI image

        # Overlay all organs with different colors
        ax.imshow(mask_liver[:, :, i], cmap="Reds", alpha=0.5)  # Liver in Red
        ax.imshow(mask_rk[:, :, i], cmap="Blues", alpha=0.5)  # Right Kidney in Blue
        ax.imshow(mask_lk[:, :, i], cmap="Greens", alpha=0.5)  # Left Kidney in Green
        ax.imshow(mask_spleen[:, :, i], cmap="Purples", alpha=0.5)  # Spleen in Purple

        ax.set_title(f"Slice {i}")
        ax.axis("off")
    else:
        ax.axis("off")  # Hide unused subplots

plt.tight_layout()
plt.show()
