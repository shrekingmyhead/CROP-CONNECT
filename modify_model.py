import h5py

# Path to the model file
model_path = 'plant_disease_model.h5'

# Open the HDF5 file in read-write mode
with h5py.File(model_path, 'r+') as f:
    # Navigate to the DepthwiseConv2D layer
    layer_path = 'model_weights/conv_dw_1/conv_dw_1'
    if layer_path in f:
        layer = f[layer_path]
        if 'config' in layer.attrs:
            # Load the layer's config
            config = layer.attrs['config']
            print("Original config:", config)

            # Check if 'groups' is in the config
            if 'groups' in config:
                print("Removing 'groups' argument from the config...")
                del config['groups']  # Remove the 'groups' argument
                layer.attrs['config'] = config  # Update the config
                print("'groups' argument removed successfully.")
            else:
                print("'groups' argument not found in the config.")
        else:
            print("'config' attribute not found in the layer.")
    else:
        print(f"Layer path '{layer_path}' not found in the model.")