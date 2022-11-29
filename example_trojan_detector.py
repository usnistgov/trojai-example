# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import torch
import json
import jsonschema
import gym
import torch_ac
from gym_minigrid.wrappers import ImgObsWrapper

import logging
import warnings

warnings.filterwarnings("ignore")


def example_trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath,
                            parameters_dirpath, parameter1, parameter2):
    logging.info('model_filepath = {}'.format(model_filepath))
    logging.info('result_filepath = {}'.format(result_filepath))
    logging.info('scratch_dirpath = {}'.format(scratch_dirpath))
    logging.info('examples_dirpath = {}'.format(examples_dirpath))
    logging.info('Using parameters_dirpath = {}'.format(parameters_dirpath))
    logging.info('Using parameter1 = {}'.format(parameter1))
    logging.info('Using parameter2 = {}'.format(parameter2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using compute device: {}".format(device))

    # load the model and move it to available device
    model = torch.load(model_filepath)
    model.to(device)
    model.eval()

    preprocess = torch_ac.format.default_preprocess_obss

    # Utilize open source minigrid environment model was trained on
    env_string = 'MiniGrid-LavaCrossingS9N1-v0'
    logging.info('Evaluating on {}'.format(env_string))

    # Number of episodes to run
    episodes = 100

    env_perf = {}

    # Run episodes through an environment to collect what may be relevant information to trojan detection
    # Construct environment and put it inside a observation wrapper
    env = ImgObsWrapper(gym.make(env_string))
    obs = env.reset()
    obs = preprocess([obs], device=device)

    final_rewards = []
    with torch.no_grad():
        # Episode loop
        for _ in range(episodes):
            done = False
            # Use env observation to get action distribution
            dist, value = model(obs)
            # Per episode loop
            while not done:
                # Sample from distribution to determine which action to take
                action = dist.sample()
                action = action.cpu().detach().numpy()
                # Use action to step environment and get new observation
                obs, reward, done, info = env.step(action)
                # Preprocessing function to prepare observation from env to be given to the model
                obs = preprocess([obs], device=device)
                # Use env observation to get action distribution
                dist, value = model(obs)

            # Collect episode performance data (just the last reward of the episode)
            final_rewards.append(reward)
            # Reset environment after episode and get initial observation
            obs = env.reset()
            obs = preprocess([obs], device=device)

    # Save final rewards
    env_perf['final_rewards'] = final_rewards

    # Test scratch space
    with open(os.path.join(scratch_dirpath, "env_perf.json"), mode='w', encoding='utf-8') as f:
        json.dump(env_perf, f, indent=2)

    trojan_probability = np.random.rand()
    print('Trojan Probability: {}'.format(trojan_probability))

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))


def configure(output_parameters_dirpath,
              configure_models_dirpath,
              parameter3):
    logging.info('Using parameter3 = {}'.format(str(parameter3)))

    logging.info('Configuring detector parameters with models from ' + configure_models_dirpath)

    os.makedirs(output_parameters_dirpath, exist_ok=True)

    logging.info('Writing configured parameter data to ' + output_parameters_dirpath)

    arr = np.random.rand(100, 100)
    np.save(os.path.join(output_parameters_dirpath, 'numpy_array.npy'), arr)

    with open(os.path.join(output_parameters_dirpath, "single_number.txt"), 'w') as fh:
        fh.write("{}".format(17))

    example_dict = dict()
    example_dict['keya'] = 2
    example_dict['keyb'] = 3
    example_dict['keyc'] = 5
    example_dict['keyd'] = 7
    example_dict['keye'] = 11
    example_dict['keyf'] = 13
    example_dict['keyg'] = 17

    with open(os.path.join(output_parameters_dirpath, "dict.json"), mode='w', encoding='utf-8') as f:
        json.dump(example_dict, f, warnings=True, indent=2)


if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.',
                        default='./model.pt')
    parser.add_argument('--result_filepath', type=str,
                        help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.',
                        default='./output')
    parser.add_argument('--scratch_dirpath', type=str,
                        help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.',
                        default='./scratch')
    parser.add_argument('--examples_dirpath', type=str,
                        help='File path to the folder of examples which might be useful for determining whether a model is poisoned.',
                        default='./example')

    parser.add_argument('--metaparameters_filepath',
                        help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.',
                        action=ActionConfigFile)
    parser.add_argument('--schema_filepath', type=str,
                        help='Path to a schema file in JSON Schema format against which to validate the config file.',
                        default=None)
    parser.add_argument('--learned_parameters_dirpath', type=str,
                        help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.')

    parser.add_argument('--configure_mode',
                        help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.',
                        default=False, action="store_true")
    parser.add_argument('--configure_models_dirpath', type=str,
                        help='Path to a directory containing models to use when in configure mode.')

    # these parameters need to be defined here, but their values will be loaded from the json file instead of the command line
    parser.add_argument('--parameter1', type=int, help='An example tunable parameter.')
    parser.add_argument('--parameter2', type=float, help='An example tunable parameter.')
    parser.add_argument('--parameter3', type=str, help='An example tunable parameter.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s")
    logging.info("example_trojan_detector.py launched")

    # Validate config file against schema
    config_json = None
    if args.metaparameters_filepath is not None:
        with open(args.metaparameters_filepath[0]()) as config_file:
            config_json = json.load(config_file)
            if args.parameter1 is None:
                args.parameter1 = config_json['parameters1']
            if args.parameter2 is None:
                args.parameter2 = config_json['parameters1']
    if args.schema_filepath is not None:
        with open(args.schema_filepath) as schema_file:
            schema_json = json.load(schema_file)

        # this throws a fairly descriptive error if validation fails
        jsonschema.validate(instance=config_json, schema=schema_json)

    logging.info(args)

    if not args.configure_mode:
        if (args.model_filepath is not None and
                args.result_filepath is not None and
                args.scratch_dirpath is not None and
                args.examples_dirpath is not None and
                args.learned_parameters_dirpath is not None and
                args.parameter1 is not None and
                args.parameter2 is not None):

            logging.info("Calling the trojan detector")
            example_trojan_detector(args.model_filepath,
                                    args.result_filepath,
                                    args.scratch_dirpath,
                                    args.examples_dirpath,
                                    args.learned_parameters_dirpath,
                                    args.parameter1, args.parameter2)
        else:
            logging.info("Required Evaluation-Mode parameters missing!")
    else:
        if (args.learned_parameters_dirpath is not None and
                args.configure_models_dirpath is not None and
                args.parameter3 is not None):

            logging.info("Calling configuration mode")
            # all 3 example parameters will be loaded here, but we only use parameter3
            configure(args.learned_parameters_dirpath,
                      args.configure_models_dirpath,
                      args.parameter3)
        else:
            logging.info("Required Configure-Mode parameters missing!")
