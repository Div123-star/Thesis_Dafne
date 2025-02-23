import numpy as np
import matplotlib.pyplot as plt
data = np.load("/Users/dibya/MyThesisDatasets/CHAOS_Dataset_for_Dibya/all_patient/13.npz")  # Replace with an actual file
print(data.files)  # Check stored arrays
print(np.unique(data["mask_Liver"]))  # Verify liver mask values
print(np.unique(data["mask_RK"]))  # Verify right kidney values
# Print available keys
print("Stored keys:", data.files)

# Extract MRI image data
image_data = data["data"]  # Shape (H, W, Slices)
num_slices = image_data.shape[2]

# Extract segmentation masks
mask_liver = data["mask_Liver"]
mask_rk = data["mask_RK"]
mask_lk = data["mask_LK"]
mask_spleen = data["mask_Spleen"]

# Extract resolution
resolution = data["resolution"]

# Print details
print(f"✅ Patient 13 Data Information")
print(f"- Image shape: {image_data.shape} (Height, Width, Slices)")
print(f"- Number of slices: {num_slices}")
print(f"- Resolution: {resolution}")
print(f"- Liver mask shape: {mask_liver.shape}, Unique values: {np.unique(mask_liver)}")
print(f"- Right Kidney mask shape: {mask_rk.shape}, Unique values: {np.unique(mask_rk)}")
print(f"- Left Kidney mask shape: {mask_lk.shape}, Unique values: {np.unique(mask_lk)}")
print(f"- Spleen mask shape: {mask_spleen.shape}, Unique values: {np.unique(mask_spleen)}")




# Select a slice (e.g., slice 15)
slice_idx = 20

# Display the MRI image with the liver mask overlaid
plt.figure(figsize=(6, 6))
plt.imshow(image_data[:, :, slice_idx], cmap="gray")
# plt.imshow(mask_liver[:, :, slice_idx], cmap="Reds", alpha=0.5)
# plt.title(f"Liver Mask Overlay - Slice {slice_idx}")
plt.axis("off")
plt.show()
