# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import logging
from collections import OrderedDict

import numpy as np
from tqdm import tqdm


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


def flatten_models(model_repr_dict, model_layer_map):
    """Flatten a list of models

    Args:
        model_repr_dict:
        model_layer_map:

    Returns:
    """
    flat_models = {}

    for _ in range(len(model_repr_dict)):
        (model_arch, models) = model_repr_dict.popitem()
        if model_arch not in flat_models.keys():
            flat_models[model_arch] = []

        logging.info("Flattenning %s models...", model_arch)
        for _ in tqdm(range(len(models))):
            model = models.pop(0)
            flat_models[model_arch].append(
                flatten_model(model, model_layer_map[model_arch])
            )

    return flat_models
