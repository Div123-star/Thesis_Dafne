import numpy as np

from dafne_dl import DynamicDLModel
import matplotlib.pyplot as plt

# load the model
with open('//Users/dibya/160_amos.model', 'rb') as f:
    m = DynamicDLModel.Load(f)

# load the data
with open('/Users/dibya/MyThesisDatasets/CHAOS_Dataset_for_Dibya/all_patient/test_chaos/32.npz', 'rb') as f:
    d = np.load(f)
    image = d['data']
    resolution = d['resolution']
    mask_Liver = d['mask_Liver']

out_mask_liver = np.zeros_like(image)
base_mask_liver = np.zeros_like(image)

start_slice = 0
for slice in range(mask_Liver.shape[2]):
    if np.sum(mask_Liver[:, :, slice]) > 1000:
        start_slice = slice
        break




for slice in range(start_slice,start_slice + 10):
    output = m.apply({'image': image[:,:,slice], 'resolution': resolution[:2]})
    print(f"Model output keys: {output.keys()}")
    print('Slice', slice)
    print('Base', np.sum(mask_Liver[:, :, slice]))
    try:
        out = output['Liver']
    except KeyError:
        out = output['liver']
    print('Output',np.sum(out))
    plt.figure()
    plt.subplot(131)
    plt.imshow(image[:,:,slice])
    plt.subplot(132)
    plt.imshow(out)
    plt.subplot(133)
    plt.imshow(mask_Liver[:, :, slice])
    out_mask_liver[:,:,slice] = out
    base_mask_liver[:,:,slice] = mask_Liver[:,:,slice]

plt.show()

numerator = np.sum(out_mask_liver * base_mask_liver)
denominator = np.sum(out_mask_liver) + np.sum(base_mask_liver)

dice = 2*numerator/denominator

print(f"dice score : {dice}")



















