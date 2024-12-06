""" Utilities for manipulating arrays
"""


def get_model_shape(model):
    return set([tensor.shape for _, tensor in model.items()])
