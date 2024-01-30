# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

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

