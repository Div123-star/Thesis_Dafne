
from sklearn.metrics import jaccard_score
from dafne_dl.DynamicDLModel import DynamicDLModel
import tensorflow as tf
import numpy as np
file_path = "/Users/dibya/dafne/MyThesisDatasets/amos22/MRI_data/test_npz/amos_0600.npz"
with open(file_path, 'rb') as f:
    dataset = np.load(f)

    # Access the data and masks
    keys = dataset.keys()
    print(keys)

    images = dataset['data']

    resolution = dataset['resolution']
    print("Resolution:", resolution)