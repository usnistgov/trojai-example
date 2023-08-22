import cv2
import gymnasium
import numpy as np
import torch

class TensorWrapper(gymnasium.ObservationWrapper):
    def observation(self, observation):
        observation['image'] = torch.tensor((observation['image'])).unsqueeze(0)
        observation['direction'] = torch.tensor((observation['direction']))
        observation['direction'].resize_(1, 1)
        return observation


class ObsEnvWrapper(gymnasium.ObservationWrapper):
    """Wrapper that converts observations to correct shapes and types based on the model used."""

    def __init__(self, env, mode='simple'):
        self.mode = mode
        self.simple_image_size = 48
        self.resnet_image_size = 112
        super().__init__(env)

    @property
    def observation_space(self):
        if self.mode == "simple_grayscale":
            img_space = gymnasium.spaces.Box(0, 255, (1, 48, 48), dtype=np.uint8)
        elif self.mode == "simple_rgb":
            img_space = gymnasium.spaces.Box(0, 255, (3, 48, 48), dtype=np.uint8)
        elif self.mode == "resnet_rgb":
            img_space = gymnasium.spaces.Box(0, 255, (3, 112, 112), dtype=np.uint8)
        elif self.mode == "simple":
            img_space = gymnasium.spaces.Box(0, 255, (3, 7, 7), dtype=np.uint8)
        else:
            raise ValueError(f"Unknown image mode: {self.mode}")
        return gymnasium.spaces.Dict({'image': img_space, 'direction': gymnasium.spaces.Box(0, 3, (1,), dtype=int)})

    def observation(self, observation):
        if self.mode == 'simple':
            observation['image'] = observation['image'].transpose((2, 0, 1))  # shift dims to channels first
            return observation
        if self.mode == 'simple_grayscale':
            observation['image'] = np.expand_dims(cv2.resize(cv2.cvtColor(observation['image'], cv2.COLOR_RGB2GRAY),
                                                             (self.simple_image_size, self.simple_image_size)), 0)
        elif self.mode == "resnet_rgb":
            observation['image'] = cv2.resize(observation['image'], (self.resnet_image_size, self.resnet_image_size))
            h, w, c = observation['image'].shape
            observation['image'] = observation['image'].reshape(c, h, w)  # shift dims to channels first
        else:
            # e.g. "simple_rgb"
            observation['image'] = cv2.resize(observation['image'], (self.simple_image_size, self.simple_image_size))
            h, w, c = observation['image'].shape
            observation['image'] = observation['image'].reshape(c, h, w)  # shift dims to channels first
        return observation

    def set_sub_env_attr(self, attr, value):
        if not hasattr(self.env, attr):
            raise ValueError(f"{self.env} sub-env has not attribute {attr}")
        setattr(self.env, attr, value)