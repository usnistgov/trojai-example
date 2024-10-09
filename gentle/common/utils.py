import torch
import numpy as np
from copy import deepcopy
from box import Box


def get_env_object(config: Box):
    """
    Helper function that returns environment object.  Can include more environments as they become available.  While
    the following does not explicitly require that the environment inherit from gym.Env, any environment that does
    follow the OpenAI gym format should be compatible.
    """
    if 'environment' not in config:
        raise ValueError('environment information missing from config')
    else:
        
        if config.environment.type.lower() == 'atari':
            import gym  # todo: migrate to gymnasium?
            from .atari_wrappers import make_atari, wrap_deepmind

            class PyTorchAtari(gym.Wrapper):
                def __init__(self, base_env, dim_order: tuple):
                    """  Wrapper to appropriately re-shape arrays for PyTorch processing  """
                    gym.Wrapper.__init__(self, base_env)
                    self.dim_order = dim_order

                def reset(self, **kwargs):
                    obs = self.env.reset(**kwargs)
                    return np.transpose(obs, (0, 3, 1, 2))

                def step(self, action):
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    return np.transpose(obs, (0, 3, 1, 2)), reward, terminated, truncated, info

            env_config = deepcopy(config.environment)
            env_config.clip_rewards = 0
            env_name = env_config.pop('name', None)
            return PyTorchAtari(wrap_deepmind(make_atari(env_name), **env_config), (0, 3, 1, 2))
        else:
            if config.environment.type.lower() == 'dm_control':
                from .dmc_wrapper import DMCWrapper
                domain_name, task_name = config.environment.name.split('-')
                env = DMCWrapper(domain_name=domain_name, task_name=task_name)
            elif config.environment.type.lower() == 'safety_gymnasium':
                import gymnasium as gym
                import safety_gymnasium
                from gentle.common.modified_safety_wrapper import ModifiedSafetyGymnasium

                safety_gymnasium_env = safety_gymnasium.make(config.environment.name)
                env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium_env)
                if 'mod_config' in config.environment:
                    env = ModifiedSafetyGymnasium(env, config.environment.mod_config)

            elif config.environment.type[:3].lower() == 'gym':
                if config.environment.type.lower() == 'gym':
                    import gym
                else:
                    import gymnasium as gym

                class ScaledGym(gym.Wrapper):
                    def __init__(self, base_env, reward_scale):
                        """
                        Scales returned rewards from gym environment
                        :param base_env: (Gym Environment); the environment to wrap
                        :param reward_scale: (float); multiplier for reward
                        """
                        gym.Wrapper.__init__(self, base_env)
                        self.reward_scale = reward_scale

                    def step(self, action):
                        obs, reward, terminated, truncated, info = self.env.step(action)
                        return obs, reward * self.reward_scale, terminated, truncated, info

                if 'scale' in config.environment:
                    env = ScaledGym(gym.make(config.environment.name), config.environment.scale)
                else:
                    env = gym.make(config.environment.name)
            else:
                raise ValueError('unknown environment type')
            config.pi_network.obs_dim = env.observation_space.shape[0]
            config.pi_network.action_dim = env.action_space.shape[0]
            if 'v_network' in config:
                config.v_network.obs_dim = env.observation_space.shape[0]
                config.v_network.action_dim = 1
            if 'q_network' in config:
                config.q_network.obs_dim = env.observation_space.shape[0] + env.action_space.shape[0]
                config.q_network.action_dim = 1
            return env


def get_network_object(config: Box) -> torch.nn.Module:
    """  Helper function that returns network object.  Can include more networks as they become available.  """
    if 'network_name' not in config:
        raise ValueError('network_name missing from config')
    if config.network_name.lower() == 'atari':
        from .networks import AtariNetwork
        return AtariNetwork(config)
    elif config.network_name.lower() in ['mlp', 'mlp_td3']:
        from .networks import MLP
        return MLP(config)
    elif config.network_name.lower() == 'mlp_categorical':
        from .networks import CategoricalMLP
        return CategoricalMLP(config)
    elif config.network_name.lower() in ['mlp_gaussian', 'mlp_sac', 'mlp_opac']:
        from .networks import GaussianMLP
        return GaussianMLP(config)
    elif config.network_name.lower() == 'mlp_quant':
        from .networks import QuantileMLP
        return QuantileMLP(config)
    else:
        raise ValueError('network_name not recognized.')


def get_sampler(config: Box, deterministic=False, action_space=None):
    suffix = config.network_name.lower().split('_')[-1]
    if suffix == 'categorical':
        from .samplers import CategoricalSampler
        return CategoricalSampler(config, deterministic)
    elif suffix == 'gaussian':
        from .samplers import GaussianSampler
        return GaussianSampler(config, deterministic)
    elif suffix == 'sac' or suffix == 'dsac':
        from .samplers import SoftActorCriticSampler
        return SoftActorCriticSampler(config, deterministic)
    elif suffix == 'opac':
        if config.sampler == 'tanh':
            from .samplers import OffPolicyActorCriticTanhSampler
            return OffPolicyActorCriticTanhSampler(config, deterministic)
        else:
            from .samplers import OffPolicyActorCriticClipSampler
            return OffPolicyActorCriticClipSampler(config, deterministic)
    elif suffix == 'td3':
        from .samplers import TD3Sampler
        return TD3Sampler(config, action_space)
