import re

from collections import OrderedDict


def create_layer_map(model_repr_dict):
    model_layer_map = {}
    for (model_class, models) in model_repr_dict.items():
        layers = models[0]
        layer_names = list(layers.keys())
        base_layer_names = list(
            dict.fromkeys(
                [
                    re.sub(
                        "\\.(weight|bias|running_(mean|var)|num_batches_tracked)",
                        "",
                        item,
                    )
                    for item in layer_names
                ]
            )
        )
        layer_map = OrderedDict(
            {
                base_layer_name: [
                    layer_name
                    for layer_name in layer_names
                    if re.match(f"{base_layer_name}.+", layer_name) is not None
                ]
                for base_layer_name in base_layer_names
            }
        )
        model_layer_map[model_class] = layer_map

    return model_layer_map
