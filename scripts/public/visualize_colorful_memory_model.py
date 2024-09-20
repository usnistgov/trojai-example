# Modified from rl-starter-files/scripts/visualize.py

import argparse
import json
import os

import numpy
import torch

from rl_starter_files_code import utils
from src.public.colorful_memory_agent import ColorfulMemoryAgent
from src.public.make_colorful_memory_funtions import make_clean_env


def visualize(args):
    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    print(f"Device: {device}\n")

    # Load environment
    env = make_clean_env(args, render_mode="human")
    for _ in range(args.shift):
        env.reset()
    print("Environment loaded\n")

    # Load agent
    model_dir = args.model_dir
    agent = ColorfulMemoryAgent(env.observation_space, env.action_space, model_dir,
                                argmax=args.argmax, num_envs=1, device=device)
    print("Agent loaded\n")

    # Run the agent
    if args.gif:
        from array2gif import write_gif
        frames = []

    # Create a window to view the environment
    env.render()

    for episode in range(args.episodes):
        obs, _ = env.reset()
        while True:
            env.render()
            if args.gif:
                frames.append(numpy.moveaxis(env.get_frame(), 2, 0))
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated
            agent.analyze_feedback(reward, done)
            if done:
                break

    if args.gif:
        print("Saving gif... ", end="")
        write_gif(numpy.array(frames), args.gif+".gif", fps=1/args.pause)
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Trojai Colorful Memory Visualization")
    parser.add_argument("model_dir", type=str, help="(str) path to the directory where the model is saved (along with "
                                                    "other training information)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to visualize")
    parser.add_argument("--shift", type=int, default=0, help="How many times to reset the env (with the given seed) "
                                                             "before visualizing. Shows different randomization "
                                                             "configurations of the environment.")
    parser.add_argument("--pause", type=float, default=0.1, help="Pause duration between two consequent actions of "
                                                                 "the agent.")
    parser.add_argument("--gif", type=str, default=None, help="Store output as gif with the given filename.")
    parser.add_argument("--argmax", action="store_true", help="Action with the highest probability is selected.")
    parser.add_argument("--gpu", action="store_true", help="Use a GPU.")
    parser.add_argument("--seed", type=int, default=1, help="Seed the visualization.")

    args = parser.parse_args()
    model_dir = args.model_dir

    with open(os.path.join(model_dir, "reduced-config.json"), "r") as f:
        config = json.load(f)

    #   grid_size: (int) Size of the environment grid
    args.grid_size = config["grid_size"]
    #   random_length: (bool) If the length of the hallway is randomized (within the allowed size of the grid)
    args.random_length = config["random_length"]
    #   max_steps: (int) The maximum allowed steps for the env (AFFECTS REWARD MAGNITUDE!) - recommend 250
    args.max_steps = config["max_steps"]

    visualize(args)