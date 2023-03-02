# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import os
from collections import OrderedDict
import torch


def wrap_network_prediction(boxes, labels):
    """

    Args:
        boxes: numpy array [N x 4] of the box coordinates
        labels: numpy array [N] of the labels

    Returns:
        list({'bbox': box, 'label': label})
    """
    pred = list()
    for k in range(boxes.shape[0]):
        ann = dict()
        ann['bbox'] = boxes[k, :].tolist()
        ann['label'] = labels[k]
        pred.append(ann)
    return pred


def load_model(model_filepath: str) -> (dict, str):
    """Load a model given a specific model_path.

    Args:
        model_filepath: str - Path to model.pt file

    Returns:
        model, dict, str - Torch model + dictionary representation of the model + model class name
    """
    model = torch.load(model_filepath)
    model_class = model.__class__.__name__
    model_repr = OrderedDict(
        {layer: tensor.numpy() for (layer, tensor) in model.state_dict().items()}
    )

    return model, model_repr, model_class


def load_ground_truth(model_dirpath: str):
    """Returns the ground truth for a given model.

    Args:
        model_dirpath: str -

    Returns:
        Ground truth value (int)
    """

    with open(os.path.join(model_dirpath, "ground_truth.csv"), "r") as fp:
        model_ground_truth = fp.readlines()[0]

    return int(model_ground_truth)


def load_models_dirpath(models_dirpath):
    model_repr_dict = {}
    model_ground_truth_dict = {}

    for model_path in models_dirpath:
        model, model_repr, model_class = load_model(os.path.join(model_path, "model.pt"))
        model_ground_truth = load_ground_truth(model_path)

        # Build the list of models
        if model_class not in model_repr_dict.keys():
            model_repr_dict[model_class] = []
            model_ground_truth_dict[model_class] = []

        model_repr_dict[model_class].append(model_repr)
        model_ground_truth_dict[model_class].append(model_ground_truth)

    return model_repr_dict, model_ground_truth_dict
