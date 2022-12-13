from collections import OrderedDict

import numpy as np


def pad_to_target(input_array: np.array, target_padding: list, constant_value=0):
    try:
        padding_array = []
        assert len(target_padding) == len(input_array.shape)
        for (idx, target) in enumerate(target_padding):
            current = input_array.shape[idx]
            assert target >= current
            if target > current:
                padding_array.append((0, target - current))
            else:
                padding_array.append((0, 0))
    except AssertionError:
        raise Exception(
            f"Incorrect target padding: {input_array.shape} cannot be padded to "
            f"{target_padding}!"
        )
    return np.pad(input_array, padding_array, constant_values=constant_value)


def flatten_layer(model, layer_map):
    nbt_layer = None
    output = None
    for layer in layer_map:
        if "num_batches_tracked" in layer:
            nbt_layer = layer
            continue
        if len(model[layer].shape) == 1:
            model_layer = np.array([model[layer]]).T
        else:
            feats = model[layer].shape[0]
            flat_layer = model[layer].flatten()
            model_layer = flat_layer.reshape(feats, int(flat_layer.shape[0] / feats))
        if output is None:
            output = model_layer
        else:
            output = np.hstack((output, model_layer))
    output = output.flatten()
    if nbt_layer:
        output = np.hstack((output, model[nbt_layer]))
    return output


def flatten_model(input_model, model_layers):
    new_model = OrderedDict()
    for (layer, layer_map) in model_layers.items():
        new_model[layer] = (
            flatten_layer(input_model, layer_map)
            if len(layer_map) > 0
            else input_model[layer].flatten()
        )
        assert len(new_model[layer].shape) == 1
    return new_model
