import tensorflow as tf
from dafne_dl.DynamicDLModel import source_to_fn, DynamicDLModel
with open("/home/dibya/Documents/Different_models/layer_46.model") as f:
    old_dafne_model = DynamicDLModel.Load(f)
    old_model = old_dafne_model.model

print("Total layers:" ,len(old_model.layers))

old_model.summary()
trainable_layers = [layer for layer in old_model.layers if layer.trainable]
for layer in trainable_layers:
    print(layer.name)
    print(len(trainable_layers))
