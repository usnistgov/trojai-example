import numpy as np
from utils.arrays import get_model_shape


def create_models_padding(model_repr_dict: dict) -> dict:
    padding = {}

    for (model_class, model_repr_list) in model_repr_dict.items():
        # Create reference model shape from the first model
        reference_model = model_repr_list[0]
        reference = get_model_shape(reference_model)

        padding[model_class] = {}

        # Ensure every model has the same shape as the reference model
        for model_repr in model_repr_list:
            try:
                assert len(get_model_shape(model_repr) ^ reference) == 0
            except AssertionError:
                for layer, tensor in model_repr.items():
                    tshape = tensor.shape
                    rshape = reference_model[layer].shape

                    if len(tshape) != len(rshape):
                        raise Exception(f"Incompatible shape detected for layer {layer}")

                    if tshape != rshape:
                        new_padding = []

                        for i in range(len(tshape)):
                            possible_shapes = [tshape[i], rshape[i]]

                            if layer in padding[model_class].keys():
                                possible_shapes.append(padding[model_class][layer][i])

                            new_padding.append(
                                max(possible_shapes)
                            )

                        padding[model_class][layer] = new_padding

    return padding


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


def pad_model(model_dict: dict, model_class: str, models_padding_dict: dict) -> dict:
    """Ensure every layer is correctly padded, so that every model has the same
    number of weights no matter the number of classes.

    Args:
        model_dict: dict - Dictionary representation of the model
        model_class: str - Model class name
        models_padding_dict: dict - Paddings for the round's model classes

    Returns:
        dict - The padded dictionary
    """
    for (layer, target_padding) in models_padding_dict[model_class].items():
        model_dict[layer] = pad_to_target(model_dict[layer], target_padding)

    return model_dict

