import tensorflow as tf
from dafne_dl.DynamicDLModel import source_to_fn, DynamicDLModel

# Open the model file in binary mode
with open("/Users/dibya/50_chaos.model", "rb") as f:
    old_dafne_model = DynamicDLModel.Load(f)
    old_model = old_dafne_model.model

# Print total number of layers
print("Total layers:", len(old_model.layers))

# Display model summary
old_model.summary()

# Get and print all trainable layers
trainable_layers = [layer for layer in old_model.layers if layer.trainable]
print(f"Total trainable layers: {len(trainable_layers)}")

# Print the names of trainable layers
print("Trainable layer names:")
for layer in trainable_layers:
    print(layer.name)


for i, layer in enumerate(old_model.layers):
    print(f"Layer {i}: {layer.name}, Trainable: {layer.trainable}")

