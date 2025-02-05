import numpy as np
from dafne_dl import DynamicDLModel
import matplotlib.pyplot as plt


# -------------------- Metric Functions --------------------

def dice_score(pred, target, epsilon=1e-6):
    pred = pred > 0  # Binarize predictions
    target = target > 0  # Binarize target
    intersection = (pred & target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + epsilon) / (union + epsilon)


def iou_score(pred, target, epsilon=1e-6):
    pred = pred > 0  # Binarize predictions
    target = target > 0  # Binarize target
    intersection = (pred & target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + epsilon) / (union + epsilon)


# -------------------- Load Model --------------------

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = DynamicDLModel.Load(f)
    return model


# -------------------- Validate and Compare Function --------------------

def validate_and_compare_models(models, file_path, slice_index):
    """
    Validate and compare multiple models on the specified slice of the dataset.
    """
    # Open and process dataset
    with open(file_path, 'rb') as f:
        dataset = np.load(f)

    # Access the data and masks
        images = dataset['data'].astype(np.float32)  # Image data
        organ_masks = [key for key in dataset.files if key.startswith('mask_')]
        print("Organ masks available:", organ_masks)

        resolution = dataset['resolution'][:2]  # Resolution

    # Select the specified slice
    image_slice = images[:, :, slice_index]

    results = {}

    for organ in organ_masks:
        mask_slice = dataset[organ][:, :, slice_index].astype(np.int32)
        organ_results = {}

        for model_name, model in models.items():
            output = model.apply({'image': image_slice, 'resolution': resolution})
            predicted_mask = np.zeros_like(image_slice, dtype=np.int32)
            for val, mask in enumerate(output.values()):
                predicted_mask += mask * (val + 1)

            dice = dice_score(predicted_mask, mask_slice)
            iou = iou_score(predicted_mask, mask_slice)

            organ_results[model_name] = {
                'Dice Score': dice,
                'IoU Score': iou
            }

        results[organ] = organ_results

    # Print and visualize results
    for organ, organ_results in results.items():
        print(f"\n--- Results for {organ} ---")
        for model_name, metrics in organ_results.items():
            print(f"Model: {model_name}")
            print(f"  Dice Score: {metrics['Dice Score']:.4f}")
            print(f"  IoU Score: {metrics['IoU Score']:.4f}")

    # Visualization for the first organ mask
    first_organ = organ_masks[0]
    mask_slice = dataset[first_organ][:, :, slice_index].astype(np.int32)

    plt.figure(figsize=(20, 5))
    plt.subplot(1, len(models) + 2, 1)
    plt.title("Original Image")
    plt.imshow(image_slice, cmap='gray')

    plt.subplot(1, len(models) + 2, 2)
    plt.title("Ground Truth Mask")
    plt.imshow(mask_slice, cmap='jet', alpha=0.6)

    for idx, (model_name, model) in enumerate(models.items(), start=3):
        output = model.apply({'image': image_slice, 'resolution': resolution})
        predicted_mask = np.zeros_like(image_slice, dtype=np.int32)
        for val, mask in enumerate(output.values()):
            predicted_mask += mask * (val + 1)

        plt.subplot(1, len(models) + 2, idx)
        plt.title(f"{model_name} Prediction")
        plt.imshow(predicted_mask, cmap='jet', alpha=0.6)

    plt.show()


# -------------------- Main Function --------------------

def main():
    # Paths for the models and dataset
    model_paths = {
        'CHAOS Base': "/Users/dibya/dafne/MyThesisDatasets/final_model/chaos_base.model",
        'AMOS Base': "//Users/dibya/dafne/MyThesisDatasets/final_model/amos_base.model",
        'CHAOS Transfer': "/Users/dibya/dafne/MyThesisDatasets/final_model/chaos_transfer.model"
    }

    data_path = "/Users/dibya/dafne/MyThesisDatasets/amos22/MRI_data/npz_files/amos_0507.npz"
    slice_index = 10

    # Load all models
    models = {name: load_model(path) for name, path in model_paths.items()}

    # Validate and compare models
    validate_and_compare_models(models, data_path, slice_index)


# -------------------- Entry Point --------------------

if __name__ == "__main__":
    main()
