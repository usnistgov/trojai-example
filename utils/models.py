# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import math
import re
from collections import OrderedDict
from os.path import join

import gymnasium as gym
import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock

def linear_w_relu(dims: list, end_relu=True):
    """Helper function for creating sequential linear layers"""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
    if not end_relu:
        del layers[-1]
    return nn.Sequential(*layers)


class ModdedResnet18(torchvision.models.ResNet):
    """Modified ResNet18 architecture for TrojAI DRL ResNet models"""
    def __init__(self):
        super(ModdedResnet18, self).__init__(BasicBlock, [2, 2, 2, 2])

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class TrojaiDRLBackbone(nn.Module):
    """Base class for TrojAI DRL models"""

    def __init__(self, embedding, actor, critic):
        nn.Module.__init__(self)
        self.state_emb = embedding  # Define state embedding
        self.actor = actor  # Define actor's model
        self.critic = critic  # Define critic's model
        self.value = 0

    def forward(self, obs):
        agent_dir = obs['direction'].long()
        obs = obs['image']
        x = self.state_emb(obs.float())
        x = x.reshape(x.shape[0], -1)
        x = torch.concat([x, agent_dir], dim=1)
        x_act = self.actor(x)
        x_crit = self.critic(x)
        self.value = x_crit.squeeze(1)
        return x_act

    def value_function(self):
        return self.value

    def args_dict(self):
        raise NotImplementedError("Should be implemented in subclass")


class FCModel(TrojaiDRLBackbone):
    """Fully-connected actor-critic model with shared embedding"""

    def __init__(self, obs_space, action_space, linear_embedding_dims=(512, 256), actor_linear_mid_dims=(64, 32),
                 critic_linear_mid_dims=(64, 32)):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        :param linear_embedding_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the linear embedding
        :param actor_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the actor network
        :param critic_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the critic network
        """
        self.obs_space = obs_space
        if isinstance(self.obs_space, gym.spaces.Dict):
            flattened_dims = int(math.prod(self.obs_space['image'].shape))
        else:
            flattened_dims = int(math.prod(self.obs_space.shape))
        self.action_space = action_space
        self.linear_embedding_dims = linear_embedding_dims
        self.actor_linear_mid_dims = actor_linear_mid_dims
        self.critic_linear_mid_dims = critic_linear_mid_dims

        # +1 because we concat direction information to embedding
        self.state_embedding_size = linear_embedding_dims[-1] + 1
        embedding = linear_w_relu([flattened_dims] + [d for d in linear_embedding_dims])
        embedding.insert(0, nn.Flatten())  # put a flattening layer in front
        actor_dims = [self.state_embedding_size] + list(actor_linear_mid_dims) + [action_space.n]
        critic_dims = [self.state_embedding_size] + list(critic_linear_mid_dims) + [1]
        super().__init__(
            embedding,
            linear_w_relu(actor_dims, end_relu=False),
            linear_w_relu(critic_dims, end_relu=False)
        )

    def args_dict(self):
        return {
            'obs_space': self.obs_space,
            'action_space': self.action_space,
            'linear_embedding_dims': self.linear_embedding_dims,
            'actor_linear_mid_dims': self.actor_linear_mid_dims,
            'critic_linear_mid_dims': self.critic_linear_mid_dims
        }


class CNNModel(TrojaiDRLBackbone):
    """Simple actor-critic model with CNN embedding"""

    def __init__(self, obs_space, action_space, channels=(16, 32, 64), actor_linear_mid_dims=(144,),
                 critic_linear_mid_dims=(144,)):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        :param channels: (iterable) Sequence of 3 integers representing the number of numbers of channels to use for the
            CNN embedding
        :param actor_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the actor network
        :param critic_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the critic network
        """
        if len(channels) != 3:
            raise ValueError("'channels' must be a tuple or list of length 3")

        self.obs_space = obs_space
        self.action_space = action_space
        self.channels = channels
        self.actor_linear_mid_dims = actor_linear_mid_dims
        self.critic_linear_mid_dims = critic_linear_mid_dims

        c1, c2, c3 = channels
        image_embedding_size = 4 * 4 * c3 + 1  # +1 because we concat direction information to embedding
        image_conv = nn.Sequential(
            nn.Conv2d(3, c1, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(c1, c2, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(c2, c3, (2, 2)),
            nn.ReLU()
        )
        actor_dims = [image_embedding_size] + list(actor_linear_mid_dims) + [action_space.n]
        critic_dims = [image_embedding_size] + list(critic_linear_mid_dims) + [1]
        super().__init__(
            image_conv,
            linear_w_relu(actor_dims, end_relu=False),
            linear_w_relu(critic_dims, end_relu=False)
        )

    def args_dict(self):
        return {
            'obs_space': self.obs_space,
            'action_space': self.action_space,
            'channels': self.channels,
            'actor_linear_mid_dims': self.actor_linear_mid_dims,
            'critic_linear_mid_dims': self.critic_linear_mid_dims
        }


class ImageACModel(TrojaiDRLBackbone):
    """Simple CNN Actor-Critic model designed for MiniGrid. Assumes 48x48 grayscale or RGB images."""

    def __init__(self, obs_space, action_space, channels=(8, 16, 32), actor_linear_mid_dims=(144,),
                 critic_linear_mid_dims=(144,)):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training. Technically unused
            for this model, but stored both for consistency between models and to be used for later reference if needed.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        :param channels: (iterable) Sequence of 3 integers representing the number of numbers of channels to use for the
            CNN embedding
        :param actor_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the actor network
        :param critic_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the critic network
        """

        self.obs_space = obs_space
        self.action_space = action_space
        self.channels = channels
        self.actor_linear_mid_dims = actor_linear_mid_dims
        self.critic_linear_mid_dims = critic_linear_mid_dims
        self.image_size = 48  # this is the size of image this CNN was designed for

        num_channels = obs_space['image'].shape[0]
        c1, c2, c3 = channels
        image_embedding_size = 3 * 3 * c3 + 1  # +1 because we concat direction information to embedding
        image_conv = nn.Sequential(
            nn.Conv2d(num_channels, c1, (3, 3), stride=3),
            nn.ReLU(),
            nn.Conv2d(c1, c2, (4, 4), stride=2),
            nn.ReLU(),
            nn.Conv2d(c2, c3, (3, 3), stride=2),
            nn.ReLU()
        )
        actor_dims = [image_embedding_size] + list(actor_linear_mid_dims) + [action_space.n]
        critic_dims = [image_embedding_size] + list(critic_linear_mid_dims) + [1]
        super().__init__(
            image_conv,
            linear_w_relu(actor_dims, end_relu=False),
            linear_w_relu(critic_dims, end_relu=False)
        )

    def args_dict(self):
        return {
            'obs_space': self.obs_space,
            'action_space': self.action_space,
            'channels': self.channels,
            'actor_linear_mid_dims': self.actor_linear_mid_dims,
            'critic_linear_mid_dims': self.critic_linear_mid_dims
        }


class ResNetACModel(TrojaiDRLBackbone):
    """Actor-Critic model with ResNet18 embedding designed for MiniGrid. Assumes 112x112 RGB images."""

    def __init__(self, obs_space, action_space, actor_linear_mid_dims=(512,), critic_linear_mid_dims=(512,)):
        """
        Initialize the model.
        :param obs_space: (gym.Spaces) Observation space of the environment being used for training. Technically unused
            for this model, but stored both for consistency between models and to be used for later reference if needed.
        :param action_space: (gym.Spaces) Action space of the environment being used for training. Used to determine
            the size of the actor's output later.
        :param actor_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the actor network
        :param critic_linear_mid_dims: (iterable) Sequence of integers representing the number of hidden nodes in the
            intermediate layers of the critic network
        """
        self.obs_space = obs_space
        self.action_space = action_space
        self.actor_linear_mid_dims = actor_linear_mid_dims
        self.critic_linear_mid_dims = critic_linear_mid_dims

        image_embedding_size = 512 + 1  # +1 because we concat direction information to embedding
        embedding = ModdedResnet18()
        actor_dims = [image_embedding_size] + list(actor_linear_mid_dims) + [action_space.n]
        critic_dims = [image_embedding_size] + list(critic_linear_mid_dims) + [1]
        super().__init__(
            embedding,
            linear_w_relu(actor_dims, end_relu=False),
            linear_w_relu(critic_dims, end_relu=False)
        )

    def args_dict(self):
        return {
            'obs_space': self.obs_space,
            'action_space': self.action_space,
            'actor_linear_mid_dims': self.actor_linear_mid_dims,
            'critic_linear_mid_dims': self.critic_linear_mid_dims
        }


def create_layer_map(model_repr_dict):
    model_layer_map = {}
    for (model_class, models) in model_repr_dict.items():
        layers = models[0]
        layer_names = list(layers.keys())
        base_layer_names = list()
        for item in layer_names:
            toks = re.sub("(weight|bias|running_(mean|var)|num_batches_tracked)", "", item)
            # remove any duplicate '.' separators
            toks = re.sub("\\.+", ".", toks)
            base_layer_names.append(toks)
        # use dict.fromkeys instead of set() to preserve order
        base_layer_names = list(dict.fromkeys(base_layer_names))

        layer_map = OrderedDict()
        for base_ln in base_layer_names:
            re_query = "{}.+".format(base_ln.replace('.', '\.'))  # escape any '.' wildcards in the regex query
            layer_map[base_ln] = [ln for ln in layer_names if re.match(re_query, ln) is not None]

        model_layer_map[model_class] = layer_map

    return model_layer_map


def load_model(model_filepath: str) -> (dict, str):
    """Load a model given a specific model_path.

    Args:
        model_filepath: str - Path to model.pt file

    Returns:
        model, dict, str - Torch model + dictionary representation of the model + model class name
    """

    stored_dict = torch.load(model_filepath, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model_name = stored_dict.pop('model')
    if model_name not in [FCModel.__name__, CNNModel.__name__, ImageACModel.__name__, ResNetACModel.__name__]:
        raise ValueError(f"Unknown model name: {model_name} extracted from model dict")
    model_class = eval(model_name)
    state_dict = stored_dict.pop('state_dict')
    model = model_class(**stored_dict)
    model.load_state_dict(state_dict)

    model_repr = OrderedDict(
        {layer: tensor.numpy() for (layer, tensor) in model.state_dict().items()}
    )

    return model, model_repr, model_class


def load_ground_truth(model_dirpath: str):
    """Returns the ground truth for a given model.

    Args:
        model_dirpath: str -

    Returns:

    """

    with open(join(model_dirpath, "ground_truth.csv"), "r") as fp:
        model_ground_truth = fp.readlines()[0]

    return int(model_ground_truth)


def load_models_dirpath(models_dirpath):
    model_repr_dict = {}
    model_ground_truth_dict = {}

    for model_path in models_dirpath:
        model, model_repr, model_class = load_model(
            join(model_path, "model.pt")
        )
        model_ground_truth = load_ground_truth(model_path)

        # Build the list of models
        if model_class not in model_repr_dict.keys():
            model_repr_dict[model_class] = []
            model_ground_truth_dict[model_class] = []

        model_repr_dict[model_class].append(model_repr)
        model_ground_truth_dict[model_class].append(model_ground_truth)

    return model_repr_dict, model_ground_truth_dict
