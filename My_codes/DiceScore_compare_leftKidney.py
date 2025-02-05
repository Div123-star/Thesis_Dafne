import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dafne_dl import DynamicDLModel

# Define paths to models
model_paths = {
    "Chaos Transfer": "/Users/dibya/MyThesisDatasets/final_model/chaos_transfer.model",
    "Chaos Base": "/Users/dibya/MyThesisDatasets/final_model/chaos_base.model",
    "Amos Base": "/Users/dibya/MyThesisDatasets/final_model/amos_base.model"
}

# Define test datasets
test_cases = [
    "amos_0554.npz", "amos_0557.npz", "amos_0558.npz",
    "amos_0578.npz", "amos_0588.npz", "amos_0590.npz",
    "amos_0592.npz", "amos_0594.npz", "amos_0596.npz"
]
test_data_path = "/Users/dibya/MyThesisDatasets/amos22/MRI_data/test_npz/"

# Prepare DataFrame to store results
results = []

# Iterate through each test dataset
for test_case in test_cases:
    with np.load(test_data_path + test_case, allow_pickle=True) as d:
        image = d['data']
        resolution = d['resolution']
        mask_LK = d['mask_left_kidney'].copy()  # Ground truth for Left Kidney

    # Define output masks for Left Kidney
    out_masks = {model_name: np.zeros_like(image) for model_name in model_paths.keys()}
    base_mask_LK = np.zeros_like(image)

    # Find the first slice with significant left kidney presence
    start_slice = next((slice for slice in range(image.shape[2]) if np.sum(mask_LK[:, :, slice]) > 500), 0)

    for slice in range(start_slice, start_slice + 5):
        for model_name, model_path in model_paths.items():
            # Load model
            with open(model_path, 'rb') as f:
                m = DynamicDLModel.Load(f)

            # Apply model
            output = m.apply({'image': image[:, :, slice], 'resolution': resolution[:2]})

            # Extract segmentation output for Left Kidney
            try:
                out_masks[model_name][:, :, slice] = output["LK"]  # CHAOS model format
            except KeyError:
                try:
                    out_masks[model_name][:, :, slice] = output["left_kidney"]  # AMOS model format
                except KeyError:
                    print(f"Warning: Left Kidney not found in {model_name}")

            # Assign ground truth mask
            base_mask_LK[:, :, slice] = mask_LK[:, :, slice]

    # Compute Dice Score and IoU for each model
    for model_name, out_mask in out_masks.items():
        numerator = np.sum(out_mask * base_mask_LK)
        denominator = np.sum(out_mask) + np.sum(base_mask_LK)

        dice = 2 * numerator / denominator if denominator != 0 else 0  # Avoid division by zero
        iou = numerator / (denominator - numerator) if (denominator - numerator) != 0 else 0  # Avoid division by zero

        results.append([test_case, model_name, "Left Kidney", dice, iou])

# Convert results to DataFrame
df = pd.DataFrame(results, columns=["Test Dataset", "Model", "Organ", "Dice Score", "IoU"])

# Save to Excel file
output_excel_path = "/Users/dibya/MyThesisDatasets/final_model/left_kidney_comparison.xlsx"
df.to_excel(output_excel_path, index=False)

print(f"Results saved to {output_excel_path}")
