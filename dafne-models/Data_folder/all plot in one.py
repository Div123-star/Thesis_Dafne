import numpy as np
from dafne_dl import DynamicDLModel
import matplotlib.pyplot as plt

# Load the model
with open('//Users/dibya/160_amos.model', 'rb') as f:
    m = DynamicDLModel.Load(f)

# Load the data
with open('/Users/dibya/MyThesisDatasets/CHAOS_Dataset_for_Dibya/all_patient/test_chaos/32.npz', 'rb') as f:
    d = np.load(f)
    image = d['data']
    resolution = d['resolution']
    mask_Liver = d['mask_Liver']

out_mask_liver = np.zeros_like(image)
base_mask_liver = np.zeros_like(image)

# Find the first slice where the liver appears
start_slice = 0
for slice in range(mask_Liver.shape[2]):
    if np.sum(mask_Liver[:, :, slice]) > 1000:
        start_slice = slice
        break

num_slices = 10  # Number of slices to process

fig, axes = plt.subplots(num_slices, 3, figsize=(10, num_slices * 3))  # Create grid

for i, slice in enumerate(range(start_slice, start_slice + num_slices)):
    # Apply model to get output mask
    output = m.apply({'image': image[:, :, slice], 'resolution': resolution[:2]})

    try:
        out = output['Liver']
    except KeyError:
        out = output['liver']

    # Store results for Dice calculation
    out_mask_liver[:, :, slice] = out
    base_mask_liver[:, :, slice] = mask_Liver[:, :, slice]

    # Plot results in the figure
    axes[i, 0].imshow(image[:, :, slice], cmap='gray')
    axes[i, 0].set_title(f"Slice {slice} - Original")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(out, cmap='jet')
    axes[i, 1].set_title("Predicted Mask")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(mask_Liver[:, :, slice], cmap='jet')
    axes[i, 2].set_title("Ground Truth Mask")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()

# Compute Dice Score
numerator = np.sum(out_mask_liver * base_mask_liver)
denominator = np.sum(out_mask_liver) + np.sum(base_mask_liver)

dice = 2 * numerator / denominator
print(f"Dice Score: {dice}")
