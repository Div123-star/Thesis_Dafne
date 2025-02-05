import numpy as np
from dafne_dl import DynamicDLModel

# Load the model
with open('/Users/dibya/dafne/MyThesisDatasets/final_model/chaos_transfer.model', 'rb') as f:
    m = DynamicDLModel.Load(f)

# Load the data
with open('/Users/dibya/dafne/MyThesisDatasets/amos22/MRI_data/test_npz/amos_0600.npz', 'rb') as f:
    d = dict(np.load(f))
    image = d['data']
    resolution = d['resolution']

# Map model output keys to AMOS ground truth keys
ORGAN_MAP = {
    'Liver': 'mask_liver',
    'Spleen': 'mask_spleen',
    'RK': 'mask_right_kidney',
    'LK': 'mask_left_kidney',
}

# Store Dice scores
dice_scores = {}

# Process each organ
for model_key, gt_key in ORGAN_MAP.items():
    if gt_key not in d:
        print(f"Ground truth mask for {model_key} not found.")
        continue

    print(f"Processing {model_key}...")

    # Ground truth mask
    gt_mask = d[gt_key]

    # Initialize prediction and ground truth masks
    out_mask = np.zeros_like(image)
    base_mask = np.zeros_like(image)

    # Process slices 10 to 15
    for slice in range(10, 15):
        print(f"Processing slice {slice} for {model_key}...")
        output = m.apply({'image': image[:, :, slice], 'resolution': resolution[:2]})

        # Ensure the correct key exists in the model output
        if model_key in output:
            out_mask[:, :, slice] = output[model_key] > 0.5  # Threshold the predicted mask
        else:
            print(f"Prediction for {model_key} not found in slice {slice}.")
            continue

        base_mask[:, :, slice] = gt_mask[:, :, slice]

    # Calculate Dice score
    intersection = np.sum(out_mask * base_mask)
    union = np.sum(out_mask) + np.sum(base_mask)
    dice = (2 * intersection) / (union + 1e-6)  # Add epsilon to avoid division by zero

    dice_scores[model_key] = dice

# Print Dice scores
print("\nDice Scores for Slices 10 to 15:")
for organ, score in dice_scores.items():
    print(f"{organ}: {score:.4f}")
