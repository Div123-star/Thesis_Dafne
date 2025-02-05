import pickle
from sklearn.metrics import jaccard_score
from dafne_dl.DynamicDLModel import DynamicDLModel
import tensorflow as tf
import numpy as np
from dafne_dl.DynamicDLModel import DynamicDLModel

# Specify the full path to your model file
model_path = '/Users/dibya/dafne/MyThesisDatasets/final_model/chaos_transfer.model'

# Open and load the model
with open(model_path, 'rb') as f:
    loaded_model = DynamicDLModel.Load(f)

print("Model loaded successfully!")

file_path = '/Users/dibya/dafne/MyThesisDatasets/amos22/MRI_data/test_npz/amos_0593.npz'  # Replace with your file path

# Open and process dataset
with open(file_path, 'rb') as f:
    dataset = np.load(f)

    # Access the data and masks
    images = dataset['data']  # Shape: (260, 320, 72)
    resolution = dataset['resolution']
    print("Resolution:", resolution)

    # List of organ masks
    organ_masks = [key for key in dataset.files if key.startswith('mask_')]
    print("Organ masks available:", organ_masks)

    # Evaluate for each organ mask
    for organ in organ_masks:
        ground_truth = dataset[organ]  # Ground truth segmentation mask for this organ (3D volume)

        import numpy as np
        from sklearn.metrics import jaccard_score
        from dafne_dl.DynamicDLModel import DynamicDLModel
        import tensorflow as tf

        # Load your dataset
        file_path = '/Users/dibya/dafne/MyThesisDatasets/CHAOS_Dataset_for_Dibya/all_patient/test_chaos/22.npz'  # Replace with your file path

        # Open and process dataset
        with open(file_path, 'rb') as f:
            dataset = np.load(f)

            # Access the data and masks
            images = dataset['data']  # Shape: (260, 320, 72)
            # List of organ masks
            organ_masks = [key for key in dataset.files if key.startswith('mask_')]
            print("Organ masks available:", organ_masks)

            # Normalize images (if needed)
            images = images / 255.0  # Normalization to [0, 1]

            # Resize images to match the model's expected input size (256x256)
            resized_images = np.zeros((256, 256, images.shape[2]), dtype=np.float32)
            for i in range(images.shape[2]):
                slice_with_channel = images[..., i][..., np.newaxis]  # Add channel dimension
                resized_slice = tf.image.resize(slice_with_channel, (256, 256))  # Resize the slice
                resized_images[..., i] = resized_slice[..., 0]  # Remove channel dimension after resizing

            # Load your trained model
            model_path = '/Users/dibya/dafne/MyThesisDatasets/final_model/chaos_transfer.model'  # Replace with your model path
            with open(model_path, 'rb') as f_model:
                loaded_model = DynamicDLModel.Load(f_model)

            # Extract the TensorFlow model from DynamicDLModel
            model = loaded_model.model  # Access the underlying TensorFlow model

            # Prepare to store metrics
            results = {}

            # Evaluate for each organ mask
            for organ in organ_masks:
                ground_truth = dataset[organ]  # Ground truth segmentation mask for this organ (3D volume)

                # Resize ground truth masks to match the model's input size
                resized_ground_truth = np.zeros((256, 256, ground_truth.shape[2]), dtype=np.int32)
                for i in range(ground_truth.shape[2]):
                    mask_with_channel = ground_truth[..., i][..., np.newaxis]  # Add channel dimension
                    resized_mask = tf.image.resize(mask_with_channel, (256, 256), method='nearest')  # Resize the mask
                    resized_ground_truth[..., i] = resized_mask[..., 0]  # Remove channel dimension after resizing

                # Initialize lists for Dice and IoU scores
                dice_scores = []
                iou_scores = []

                # Iterate over slices
                for slice_idx in range(resized_images.shape[2]):  # Loop over the 3rd dimension (slices)
                    image_slice = resized_images[:, :, slice_idx][..., np.newaxis]  # Add channel dimension
                    gt_slice = resized_ground_truth[:, :, slice_idx]  # Ground truth for the current slice

                    # Duplicate channel to match input shape (256, 256, 2)
                    image_slice = np.concatenate([image_slice, image_slice], axis=-1)

                    # Predict segmentation mask for the current slice
                    pred_slice = model.predict(image_slice[np.newaxis, ...])  # Add batch dimension
                    pred_slice = (pred_slice[0, :, :, 0] > 0.5).astype(np.int32)  # Remove batch and channel dims

                    # Flatten arrays for metric calculation
                    gt_flat = gt_slice.flatten()
                    pred_flat = pred_slice.flatten()

                    # Calculate Dice Score and IoU
                    intersection = np.sum(gt_flat * pred_flat)
                    dice = (2. * intersection) / (np.sum(gt_flat) + np.sum(pred_flat))
                    iou = jaccard_score(gt_flat, pred_flat, average='binary')

                    # Append metrics
                    dice_scores.append(dice)
                    iou_scores.append(iou)

                # Store average metrics for the organ
                results[organ] = {
                    "Dice Score": np.mean(dice_scores),
                    "IoU": np.mean(iou_scores)
                }

        # Print evaluation results
        for organ, metrics in results.items():
            print(f"{organ}: Dice Score = {metrics['Dice Score']:.4f}, IoU = {metrics['IoU']:.4f}")
