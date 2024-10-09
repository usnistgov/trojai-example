import os
import sys
import json
import argparse
import pickle
import time
import torch
import numpy as np
from box import Box
from copy import deepcopy
from gentle.common.utils import get_network_object, get_env_object, get_sampler
from gentle.rl.opac2 import OffPolicyActorCritic
from gentle.common.buffers import OffPolicyBuffer
from gentle.common.loggers import JSONLogger

# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class OffPolicyActorCriticMultitask(OffPolicyActorCritic):

    def __init__(self, config, make_train_envs=None, make_test_envs=None, env_train_weights=None):
        super().__init__(config)
        self.make_train_envs = make_train_envs
        self.make_test_envs = make_test_envs
        self.auto_weight = False
        if env_train_weights == "auto":
            self.auto_weight = True
            self.env_train_weights = [1.0/len(make_train_envs) for e in self.make_train_envs]
        elif env_train_weights is None:
            self.env_train_weights = [1.0/len(make_train_envs) for e in self.make_train_envs]
        else:
            self.env_train_weights = env_train_weights

        # json logger for performance data
        self.jsonlogger = JSONLogger(config.log_folder)

    def initialize_env(self):
        """  Initialize environment objects  """

        # list of training environments (i.e. [clean, poisoned])
        got_env = False
        if self.make_train_envs is not None:
            self.train_envs = [fn(self.config) for fn in self.make_train_envs]
        else:
            got_env = True
            self.train_envs = [get_env_object(self.config)]
            print("WARNING: Env built from config, not a list of methods.")

        # for spaces, etc
        self.env = self.train_envs[0]

        # list of test environments (i.e. [clean, poisoned])
        if self.make_test_envs is not None:
            if isinstance(self.make_test_envs, list):
                self.eval_envs = [fn(self.config) for fn in self.make_test_envs]
            else:
                self.eval_envs = [self.make_test_envs(self.config)]
        else:
            self.eval_envs = [get_env_object(self.config)]

        if self.config.entropy_type.lower() == 'reg':
            self.config.alpha[0] *= 0.67 * np.prod(self.env.action_space.shape)

        if not got_env:
            self.config.pi_network.obs_dim = self.env.observation_space.shape[0]
            self.config.pi_network.action_dim = self.env.action_space.shape[0]
            if 'v_network' in self.config:
                self.config.v_network.obs_dim = self.env.observation_space.shape[0]
                self.config.v_network.action_dim = 1
            if 'q_network' in self.config:
                self.config.q_network.obs_dim = self.env.observation_space.shape[0] + self.env.action_space.shape[0]
                self.config.q_network.action_dim = 1


    def train(self):

        self.initialize_env()

        self.initialize_buffer()

        steps, last_checkpoint = self.initialize_networks()

        self.sampler = get_sampler(self.config.pi_network)
        self.initialize_optimizers()
        obs, _ = self.reset_env(self.env)

        self.initialize_logging(obs)

        curr_obs = []
        for e in self.train_envs:
            o, _ = self.reset_env(e)
            curr_obs.append(o)

        print("Loaded")

        # average success
        best_success_rate = -1.0
        best_success_rate_log_step = 0

        # per-task success
        best_success_per_task = [-1.0 for x in self.env_train_weights]
        best_success_log_step_per_task = [0 for x in self.env_train_weights]

        # Run training:
        loss_info = {}
        while steps < self.config.total_steps:

            # each step, increment which env we are using
            env_idx = steps % len(self.train_envs)
            env = self.train_envs[env_idx]
            buf = self.buffers[env_idx]
            obs = curr_obs[env_idx]

            random_action = steps < self.config.initial_random
            action = self.get_action(obs, random_action=random_action)
            trunc = buf.episode_lengths[-1] == self.config.max_ep_length - 1
            next_obs, reward, terminated, truncated, info = self.step_env(env, action, trunc)
            buf.update(obs, action, reward, next_obs, terminated, info)
            steps += 1

            curr_obs[env_idx] = next_obs

            if terminated or truncated:
                buf.reset_episode()
                curr_obs[env_idx], _ = self.reset_env(env)

            if steps%10000 == 0:
                print(steps, "/", self.config.total_steps)

            # non-env-specific ===============================================================

            if steps >= self.config.start_learning and steps % self.config.update_every == 0:
                for i in range(self.config.updates_per_stop):
                    latest_loss_info = self.update_networks()
                    self.concatenate_dict_of_lists(loss_info, latest_loss_info)

            if steps >= self.config.start_learning:
                if (steps - self.config.start_learning) % self.config.epoch_size == 0:
                    evaluation_stoch, evaluation_det = {}, {}
                    if self.config.evaluation_type in ['stoch', 'both']:
                        evaluation_stoch = self.run_evaluation(deterministic=False)
                    if self.config.evaluation_type in ['det', 'both']:
                        evaluation_det = self.run_evaluation(deterministic=True)

                    # get current success rates per task
                    # keys in evaluation_det are env_[i]/success_rate
                    success_rates = []
                    for i in range(len(self.env_train_weights)):
                        key = "env_"+str(i)+"/success_rate"
                        succ_rate = np.mean(evaluation_det[key])
                        success_rates.append(succ_rate)

                        if succ_rate > best_success_per_task[i]:
                            best_success_per_task[i] = succ_rate
                            best_success_log_step_per_task[i] = steps

                    # get an overall success rate
                    overall_sr = []
                    for idx in self.config.environments_for_success_stopping:
                        overall_sr.append(success_rates[idx])
                    mean_rate = np.mean(overall_sr)

                    if mean_rate > best_success_rate:
                        best_success_rate = mean_rate
                        best_success_rate_log_step = steps
                        print("New best overall success rate:", best_success_rate)
                        last_checkpoint = self.save_training(steps, last_checkpoint)
                    evaluation_det["best_success_rate"] = [best_success_rate]

                    # log
                    self.update_logging(loss_info, evaluation_stoch, evaluation_det, steps)
                    loss_info = {}

                    # re-balance tasks
                    if self.auto_weight:
                        denom = np.sum(success_rates)

                        if denom == 0.0:
                            # if no one succeeded, do an even split
                            new_weights = [1.0/len(self.make_train_envs) for e in self.make_train_envs]
                        else:
                            # rate is inverse to existing success, plus some small bump to prevent weights of zero
                            new_weights = []
                            for s in success_rates:
                                new_weights.append((1.0 - s/denom) + 0.1)

                            # renormalize to account for bump
                            new_weights = np.array(new_weights)
                            new_weights = new_weights / np.sum(new_weights)

                        self.env_train_weights = new_weights

            # bail if no improvement on all tasks (early stopping)
            # this based on performance from each task separately, so we continue if we have
            # made an improvement on any task
            bail_no_progress = True
            for i in range(len(self.env_train_weights)):
                if (steps - best_success_log_step_per_task[i]) <= self.config.stop_if_no_improvement_in:
                    bail_no_progress = False

            if bail_no_progress:
                print("Performances have not improved in", self.config.stop_if_no_improvement_in, "steps - stopping training.")
                return

            # stop if reached excellent performance (early stopping)
            # this is computed across tasks, as specified by environments_for_success_stopping
            if best_success_rate >= self.config.stop_if_success_reaches:
                print("Performances have reached thresholds of", self.config.stop_if_success_reaches, "- stopping training.")
                return

            # Recycle dormant neurons or reset to combat primacy bias (use at most one of these):
            if steps // self.config.redo_interval > self.resets:
                self.recycle_dormant()
                self.resets += 1
            if steps // self.config.reset_interval > self.resets:
                self.reset_learning()
                self.resets += 1


    def get_data_from_buffer(self):
        datas = []
        for i in range(len(self.buffers)):
            amount = self.config.batch_size*self.env_train_weights[i]
            datas.append(self.buffers[i].sample(int(round(amount))))
        new_data = Box()
        for k in datas[0]:
            entry = torch.cat([d[k] for d in datas], dim=0)
            new_data[k] = entry
        return new_data

    def initialize_buffer(self):
        self.buffers = []
        for env in self.train_envs:
            buf = OffPolicyBuffer(capacity=self.config.buffer_size,
                      obs_dim=self.env.observation_space.shape[0],
                      act_dim=self.env.action_space.shape[0],
                      store_costs=self.config.store_costs)
            self.buffers.append(buf)


    def update_logging(self, loss_info, evaluation_stoch, evaluation_det, steps):
        """  Update TensorBoard logging, reset buffer logging quantities  """
        for k, v in loss_info.items():
            self.logger.log_mean_value('Learning/' + k, v, steps)
            self.jsonlogger.log_mean_value('Learning/' + k, v, steps)
        self.logger.log_mean_value('Learning/alpha', [self.alpha.item()], steps)
        self.jsonlogger.log_mean_value('Learning/alpha', [self.alpha.item()], steps)
        for k, v in evaluation_stoch.items():
            if k == 'info':
                for k_info, v_info in v.items():
                    self.logger.log_mean_value('Eval_stoch/Info/' + k_info, v_info, steps)
            else:
                self.logger.log_mean_value('Eval_stoch/' + k, v, steps)
                self.jsonlogger.log_mean_value('Eval_stoch/' + k, v, steps)
        for k, v in evaluation_det.items():
            if k == 'info':
                for k_info, v_info in v.items():
                    self.logger.log_mean_value('Eval_det/Info/' + k_info, v_info, steps)
            else:
                self.logger.log_mean_value('Eval_det/' + k, v, steps)
                self.jsonlogger.log_mean_value('Eval_det/' + k, v, steps)

        for i in range(len(self.buffers)):
            buf = self.buffers[i]
            self.logger.log_mean_value('Train/rewards/env_'+str(i), buf.episode_rewards[:-1], steps)
            self.logger.log_mean_value('Train/lengths/env_'+str(i), buf.episode_lengths[:-1], steps)
            self.logger.log_scalar('Train/weight/env_'+str(i), self.env_train_weights[i], steps)
            self.jsonlogger.log_mean_value('Train/rewards/env_'+str(i), buf.episode_rewards[:-1], steps)
            self.jsonlogger.log_mean_value('Train/lengths/env_'+str(i), buf.episode_lengths[:-1], steps)
            self.jsonlogger.log_scalar('Train/weight/env_'+str(i), self.env_train_weights[i], steps)
            buf.reset_logging()

        self.logger.flush()
        self.jsonlogger.flush()


# if __name__ == '__main__':
#     """  Runs OffPolicyActorCritic training or testing for a given input configuration file  """
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', help='Configuration file to run', required=True)
#     parser.add_argument('--mode', default='train', required=False, help='mode ("train" or "test")')
#     parser.add_argument('--seed', help='random seed', required=False, type=int, default=0)
#     parser.add_argument('--prior', help='use prior training', required=False, type=int, default=0)
#     in_args = parser.parse_args()
#     full_config = os.path.join(os.getcwd(), in_args.config)
#     print(full_config)
#     sys.stdout.flush()
#     with open(full_config, 'r') as f1:
#         config1 = Box(json.load(f1))
#     config1.seed = in_args.seed
#     if in_args.prior > 0 or in_args.mode.lower() == 'test':
#         config1.use_prior_nets = True
#     opac_object = OffPolicyActorCritic(config1)
#     if in_args.mode.lower() == 'train':
#         opac_object.train()
#     else:
#         opac_object.test()
