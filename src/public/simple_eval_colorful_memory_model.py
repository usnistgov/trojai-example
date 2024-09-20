import numpy as np
import torch

from rl_starter_files_code import utils
from src.public.make_colorful_memory_funtions import make_clean_env


def simple_eval_colorful_memory_model(num_episodes, model, device, args):
    success_rate = 0

    # Load environments
    env = make_clean_env(args)
    obs_space, preprocess_obss = utils.get_obss_preprocessor(env.observation_space)

    for i in range(num_episodes):
        obs, _ = env.reset()
        memories = torch.zeros(1, model.memory_size, device=device)
        done = False
        while not done:
            action_dist, _, memories = model(preprocess_obss(np.expand_dims(obs, 0)), memories)
            action = action_dist.sample().cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
        success_rate += 1 if reward > 0 else 0
    success_rate /= num_episodes
    return success_rate
