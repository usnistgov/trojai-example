import gymnasium as gym
import numpy as np

from src.public.colorful_memory import ColorfulMemoryEnv, ColorfulMemoryCfg


class ImageObsWrapper(gym.ObservationWrapper):
    def observation(self, observation: gym.core.ObsType) -> gym.core.WrapperObsType:
        return observation['image']

    @property
    def observation_space(self):
        return gym.spaces.Box(0, 255, shape=(7, 7, 3), dtype=np.uint8)


def make_clean_env(args, render_mode=None):
    env_cfg = {"observation_mode": 'simple',
               "size": args.grid_size,
               "render_mode": render_mode,
               "random_length": args.random_length,
               "max_steps": args.max_steps}
    return ImageObsWrapper(ColorfulMemoryEnv(ColorfulMemoryCfg(**env_cfg)))
