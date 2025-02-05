import numpy as np

file_path = "/Users/dibya/dafne/MyThesisDatasets/amos22/MRI_data/npz_files/amos_0576.npz"

# Open and load the .npz file
with open(file_path, 'rb') as f:
    dataset = np.load(f)

    # Check the keys in the .npz file
    print("Keys in the dataset:", dataset.files)

    # Access the data array
    data = dataset['data']
    print("Shape of data array:", data.shape)

    # If there is a label array
    if 'label' in dataset:
        label = dataset['label']
        print("Shape of label array:", label.shape)

    # Check resolution if available
    if 'resolution' in dataset:
        resolution = dataset['resolution']
        print("Resolution:", resolution)
