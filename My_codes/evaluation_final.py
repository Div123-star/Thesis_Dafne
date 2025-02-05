import numpy as np
from dafne_dl import DynamicDLModel

# Load the model
with open('/Users/dibya/dafne/MyThesisDatasets/final_model/chaos_base.model', 'rb') as f:
    m = DynamicDLModel.Load(f)

# Load the data
with open('/Users/dibya/dafne/MyThesisDatasets/amos22/MRI_data/test_npz/amos_0600.npz', 'rb') as f:
    d = np.load(f)
    image = d['data']
    resolution = d['resolution']
    mask_Spleen = d['mask_spleen']  # Ground truth liver mask

# Initialize masks
out_mask_spleen = np.zeros_like(image)
base_mask_spleen = np.zeros_like(image)

# Process slices
for slice_idx in range(10, 15):
    print(f"Processing slice {slice_idx}...")
    output = m.apply({'image': image[:, :, slice_idx], 'resolution': resolution[:2]})

    # Debugging output keys
    print(f"Output keys: {output.keys()}")  # Debugging step

    # Access the correct output key for liver mask
    if 'mask_Spleen' in output:
        out_mask_spleen[:, :, slice_idx] = output['mask_Spleen']
    elif 'mask_Spleen' in output:  # Handle lowercase key
        out_mask_spleen[:, :, slice_idx] = output['mask_Spleen']
    else:
        print(f"'mask_Spleen' key not found in model output for slice {slice_idx}.")
        continue

    base_mask_spleen[:, :, slice_idx] = mask_Liver[:, :, slice_idx]

# Calculate Dice score
intersection = np.sum(out_mask_spleen * base_mask_spleen)
union = np.sum(out_mask_spleen) + np.sum(base_mask_spleen)
dice = (2 * intersection) / (union + 1e-6)  # Add epsilon to avoid division by zero

print(f"Dice Score for slices 10-15: {dice:.4f}")
