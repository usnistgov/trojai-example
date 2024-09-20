# Modified from rl-starter-files/scripts/evaluate.py

import argparse
import json
import os
import time

import torch

from rl_starter_files_code import utils
from src.public.colorful_memory_agent import ColorfulMemoryAgent
from src.public.make_colorful_memory_funtions import make_clean_env
from src.public.simple_eval_colorful_memory_model import simple_eval_colorful_memory_model
from torch_ac.utils.penv import ParallelEnv


def env_run(env, agent, device, args):
    # Initialize logs
    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent
    start_time = time.time()

    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)

    while log_done_counter < args.episodes:
        actions = agent.get_actions(obss)
        obss, rewards, terminateds, truncateds, _ = env.step(actions)
        dones = tuple(a | b for a, b in zip(terminateds, truncateds))
        agent.analyze_feedbacks(rewards, dones)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    end_time = time.time()

    # Print logs
    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames / (end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
          .format(num_frames, fps, duration,
                  *return_per_episode.values(),
                  *num_frames_per_episode.values()))

    # Print worst episodes
    n = args.worst_episodes_to_show
    if n > 0:
        print("\n{} worst episodes:".format(n))

        indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
        for i in indexes[:n]:
            print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))


def evaluate(args):
    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    print(f"Device: {device}\n")

    # Load environments
    clean_envs = []
    for i in range(args.procs):
        env = make_clean_env(args)
        clean_envs.append(env)
    clean_env = ParallelEnv(clean_envs)
    print("Environments loaded\n")

    # Load agent
    model_dir = args.model_dir
    agent = ColorfulMemoryAgent(clean_env.observation_space, clean_env.action_space, model_dir,
                        argmax=args.argmax, num_envs=args.procs, device=device)
    print("Agent loaded\n")

    print("Evaluating clean performance...\n")
    env_run(clean_env, agent, device, args)
    print("")
    print("#"*100)

    print("\nComputing success rate performance...\n")
    success_rate = simple_eval_colorful_memory_model(args.success_rate_episodes, agent.acmodel, device, args)
    print(f"Success Rates -- Clean: {success_rate}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Trojai Colorful Memory Evaluation")
    parser.add_argument("model_dir", type=str, help="(str) path to the directory where the model is saved (along with "
                                                    "other training information)")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes of evaluation")
    parser.add_argument("--success_rate_episodes", type=int, default=5,
                        help="Number of episodes to evaluate success rate rather than reward (runs in serial, "
                             "so will take longer to run each env, recommend <= 100)")
    parser.add_argument("--procs", type=int, default=10, help="Number of processes to use")
    parser.add_argument("--worst_episodes_to_show", type=int, default=10, help="How many worst episodes to show")
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

    evaluate(args)
