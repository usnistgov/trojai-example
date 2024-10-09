import sys
import os
import json
import argparse
import pickle
import torch
import numpy as np
from pathlib import Path
from box import Box
from collections import OrderedDict
from shutil import rmtree
from copy import deepcopy
from datetime import datetime
from gentle.common.utils import get_env_object, get_network_object, get_sampler
from gentle.common.loggers import Logger
from gentle.common.buffers import OffPolicyBuffer

# CPU/GPU usage regulation.  One can assign more than one thread here, but it is probably best to use 1 in most cases.
os.environ['OMP_NUM_THREADS'] = '1'
torch.set_num_threads(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SoftActorCritic(object):

    def __init__(self, config):
        self.config = config
        self.process_config()
        self.env, self.eval_env = None, None
        self.pi_network = None
        self.pi_optimizer = None
        self.q1_network, self.q1_target = None, None
        self.q2_network, self.q2_target = None, None
        self.q_optimizer = None
        self.q_params = []
        self.alpha = None
        self.log_alpha = None
        self.alpha_optimizer = None
        self.target_entropy = None
        self.buffer = None
        self.sampler = None
        self.resets = 0
        self.logger = None
        t1 = int(10000 * (datetime.now().timestamp() - int(datetime.now().timestamp())))
        torch.manual_seed((1 + self.config.seed) * 2000 + t1)
        t2 = int(10000 * (datetime.now().timestamp() - int(datetime.now().timestamp())))
        np.random.seed((1 + self.config.seed) * 5000 + t2)

    def process_config(self):
        """  Processes configuration, filling in missing values as appropriate  """
        self.config.setdefault('use_prior_nets', False)  # whether to pick up where previous training left off
        self.config.setdefault('seed', 0)
        self.config.setdefault('total_steps', 1e6)  # total training steps
        self.config.setdefault('initial_random', 10000)  # initial steps over which to take random actions
        self.config.setdefault('start_learning', 1000)  # number of experience in initial data collection (no training)
        self.config.setdefault('update_every', 50)  # how often to stop environment to update
        self.config.setdefault('updates_per_stop', self.config.update_every)  # batches, updates per data collect stop
        self.config.setdefault('alpha', [0.2, False])  # initial entropy coefficient, whether to optimize
        self.config.setdefault('reparam_samples', 1)  # use only one sample in reparameterization trick
        self.config.setdefault('ent_factor', 1)  # for scaling target entropy in alpha optimization
        self.config.setdefault('gamma', 0.99)  # discount factor
        self.config.setdefault('buffer_size', 1e6)  # capacity of replay buffer
        self.config.setdefault('store_costs', False)  # whether to store costs individually in replay buffer
        self.config.setdefault('epoch_size', 1e4)  # how often to run evaluation, log
        self.config.setdefault('batch_size', 100)  # batch size sampled for each network update
        self.config.setdefault('polyak', 0.995)  # weight of previous network weights in Polyak averaging
        self.config.setdefault('lr', 0.001)  # learning rate for both Q and pi
        self.config.setdefault('lr_log_alpha', 0.0005)  # learning rate for entropy weight alpha
        # Evaluation / testing
        self.config.setdefault('evaluation_ep', 5)  # how many episodes to run each evaluation
        self.config.setdefault('evaluation_type', 'both')  # 'stoch', 'det', 'both', 'none'
        self.config.setdefault('test_episodes', 1000)  # number of test episodes to run
        # Recycling dormant neurons:
        self.config.setdefault('redo_interval', -1)  # dormant neuron recycling interval (negative means to not recycle)
        self.config.setdefault('redo_tau', 0.1)  # dormant neuron recycling threshold
        self.config.setdefault('redo_batch_size', 256)  # dormant neuron recycling batch size
        # Resetting to combat primacy bias
        self.config.setdefault('reset_interval', -1)  # primacy bias reset interval (negative means to not reset)
        self.config.setdefault('n_resets', self.config.total_steps // self.config.reset_interval - 1)
        self.config.setdefault('n_resets_pi', self.config.n_resets)
        self.config.setdefault('n_resets_q', self.config.n_resets)
        assert self.config.redo_interval < 0 or self.config.reset_interval < 0, 'Cannot use both ReDo and resets'
        # Logging and storage configurations:
        self.config.setdefault('checkpoint_every', int(1e5))
        self.config.setdefault('save_each_checkpoint', False)
        self.config.setdefault('render_mode', "None")
        self.config.setdefault('enable_restart', True)  # save buffer, allowing training to restart
        self.config.setdefault('model_folder', '../../output/sac_training')
        self.config.setdefault('log_folder', '../../logs/sac_training')
        self.config.model_folder = os.path.join(os.getcwd(), self.config.model_folder)
        self.config.log_folder = os.path.join(os.getcwd(), self.config.log_folder)
        self.config.model_folder = self.config.model_folder + '_' + str(self.config.seed)
        self.config.log_folder = self.config.log_folder + '_' + str(self.config.seed)
        if sys.platform[:3] == 'win':
            self.config.model_folder = self.config.model_folder.replace('/', '\\')
            self.config.log_folder = self.config.log_folder.replace('/', '\\')
        if not self.config.use_prior_nets:  # start a fresh training run
            if os.path.isdir(self.config.log_folder):
                rmtree(self.config.log_folder, ignore_errors=True)
            if os.path.isdir(self.config.model_folder):
                rmtree(self.config.model_folder, ignore_errors=True)
        Path(self.config.model_folder).mkdir(parents=True, exist_ok=True)
        Path(self.config.log_folder).mkdir(parents=True, exist_ok=True)

    def train(self):
        self.initialize_env()
        self.initialize_buffer()
        steps, last_checkpoint = self.initialize_networks()
        self.sampler = get_sampler(self.config.pi_network)
        self.initialize_optimizers()
        obs, _ = self.reset_env(self.env)
        self.initialize_logging(obs)
        # Run training:
        loss_info = {}
        while steps < self.config.total_steps:
            random_action = steps < self.config.initial_random
            action = self.get_action(obs, random_action=random_action)
            trunc = self.buffer.episode_lengths[-1] == self.config.max_ep_length - 1
            next_obs, reward, terminated, truncated, info = self.step_env(self.env, action, trunc)
            self.buffer.update(obs, action, reward, next_obs, terminated, info)
            steps += 1
            obs = next_obs
            if terminated or truncated:
                self.buffer.reset_episode()
                obs, _ = self.reset_env(self.env)
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
                    self.update_logging(loss_info, evaluation_stoch, evaluation_det, steps)
                    loss_info = {}
                    last_checkpoint = self.save_training(steps, last_checkpoint)
            # Recycle dormant neurons or reset to combat primacy bias (use at most one of these):
            if steps // self.config.redo_interval > self.resets:
                self.recycle_dormant()
                self.resets += 1
            if steps // self.config.reset_interval > self.resets:
                self.reset_learning()
                self.resets += 1

    def initialize_env(self):
        """  Initialize environment objects  """
        self.env = get_env_object(self.config)
        self.eval_env = get_env_object(self.config)

    def initialize_buffer(self):
        """  Initialize replay buffer  """
        if self.config.use_prior_nets:
            with open(os.path.join(self.config.model_folder, 'buffer-latest.pkl'), 'rb') as buffer_file:
                self.buffer = pickle.load(buffer_file)
                self.buffer.episode_lengths = self.buffer.episode_lengths[:-1] + [0]
                self.buffer.episode_rewards = self.buffer.episode_rewards[:-1] + [0]
        else:
            self.buffer = OffPolicyBuffer(capacity=self.config.buffer_size,
                                          obs_dim=self.env.observation_space.shape[0],
                                          act_dim=self.env.action_space.shape[0],
                                          store_costs=self.config.store_costs)

    def initialize_networks(self, reset=False):
        """  Initialize network objects  """
        total_steps, last_checkpoint = 0, -1
        if not reset or self.resets < self.config.n_resets_pi:
            self.pi_network = get_network_object(self.config.pi_network).to(device)
        if not reset or self.resets < self.config.n_resets_q:
            self.q1_network = get_network_object(self.config.q_network).to(device)
            self.q2_network = get_network_object(self.config.q_network).to(device)
            self.q1_target = deepcopy(self.q1_network).to(device)
            self.q2_target = deepcopy(self.q2_network).to(device)
            self.q_params = list(self.q1_network.parameters()) + list(self.q2_network.parameters())
        if not reset:
            if self.config.alpha[1]:  # alpha not reset for primacy bias correction
                self.log_alpha = torch.log(torch.ones(1, device=device) * self.config.alpha[0]).requires_grad_(True)
                self.alpha = torch.exp(self.log_alpha).detach()
            else:
                self.alpha = (torch.ones(1, device=device) * self.config.alpha[0]).requires_grad_(False)
            if self.config.use_prior_nets:
                checkpoint = torch.load(os.path.join(self.config.model_folder, 'model-latest.pt'),
                                        map_location=torch.device('cpu'))
                self.pi_network.load_state_dict(checkpoint.pi)
                self.q1_network.load_state_dict(checkpoint.q1)
                self.q2_network.load_state_dict(checkpoint.q2)
                self.q1_target.load_state_dict(checkpoint.q1_t)
                self.q2_target.load_state_dict(checkpoint.q2_t)
                if self.config.alpha[1]:
                    self.log_alpha = (torch.ones(1, device=device) * checkpoint.log_a).requires_grad_(True)
                    self.alpha = torch.exp(self.log_alpha).detach()
                total_steps = checkpoint.steps
                last_checkpoint = total_steps // self.config.checkpoint_every
        for p in torch.nn.ModuleList([self.q1_target, self.q2_target]).parameters():
            p.requires_grad = False
        return total_steps, last_checkpoint

    def initialize_optimizers(self, reset=False):
        """  Initializes Adam optimizer for training network.  Only one worker actually updates parameters.  """
        if not reset or self.resets < self.config.n_resets_pi:
            self.pi_optimizer = torch.optim.Adam(params=self.pi_network.parameters(), lr=self.config.lr)
        if not reset or self.resets < self.config.n_resets_q:
            q_params = torch.nn.ModuleList([self.q1_network, self.q2_network]).parameters()
            self.q_optimizer = torch.optim.Adam(params=q_params, lr=self.config.lr)
        if not reset:
            if self.config.alpha[1]:  # alpha not reset
                self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.lr_log_alpha)
                self.target_entropy = (-np.prod(self.env.action_space.shape)*self.config.ent_factor).astype(np.float32)
            if self.config.use_prior_nets:
                checkpoint = torch.load(os.path.join(self.config.model_folder, 'model-latest.pt'))
                self.pi_optimizer.load_state_dict(checkpoint.pi_opt)
                self.q_optimizer.load_state_dict(checkpoint.q_opt)
                if self.config.alpha[1]:
                    self.alpha_optimizer.load_state_dict(checkpoint.a_opt)

    def initialize_logging(self, obs):
        """  Initialize logger and store config (only on one process)  """
        self.logger = Logger(self.config.log_folder, device=device)
        if not self.config.use_prior_nets:
            with open(os.path.join(self.config.model_folder, 'config.pkl'), 'wb') as config_file:
                pickle.dump(self.config, config_file)  # store configuration
            self.logger.log_config(self.config)
            self.logger.log_graph(obs, self.pi_network)
            evaluation_stoch, evaluation_det = {}, {}
            if self.config.evaluation_type in ['stoch', 'both']:
                evaluation_stoch = self.run_evaluation(deterministic=False, num_episodes=1)
            if self.config.evaluation_type in ['det', 'both']:
                evaluation_det = self.run_evaluation(deterministic=True, num_episodes=1)
            for k, v in evaluation_stoch.items():
                if k == 'info':
                    for k_info, v_info in v.items():
                        self.logger.log_mean_value('Eval_stoch/Info/' + k_info, v_info, 0)
                else:
                    self.logger.log_mean_value('Eval_stoch/' + k, v, 0)
            for k, v in evaluation_det.items():
                if k == 'info':
                    for k_info, v_info in v.items():
                        self.logger.log_mean_value('Eval_det/Info/' + k_info, v_info, 0)
                else:
                    self.logger.log_mean_value('Eval_det/' + k, v, 0)
            self.logger.flush()

    def get_action(self, obs, random_action=False, deterministic=None):
        with torch.no_grad():
            obs_torch = torch.from_numpy(obs).to(device).float()
            pi = self.pi_network(obs_torch)
            return self.sampler.get_action(pi, random_action, deterministic)

    def get_data_from_buffer(self):
        return self.buffer.sample(self.config.batch_size)

    def update_networks(self):
        """  Update all networks  """
        data = self.get_data_from_buffer()
        pi = self.pi_network(data.obs)
        # Sample action(s) and get log_prob(s) for alpha, pi updates:
        act, log_prob_pi = self.sampler.get_action_and_log_prob(pi, n_samples=self.config.reparam_samples)
        # Log entropy:
        with torch.no_grad():
            e_info = {'gauss entropy': torch.mean(torch.log(pi[1]) + .5 * np.log(2 * np.pi * np.e)).item(),
                      'squashed entropy': -log_prob_pi.mean().item()}
        # Update Q networks:
        q_loss, q_info = self.compute_q_loss(data)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()
        # Update policy network:
        for p in self.q_params:
            p.requires_grad = False  # fix Q parameters for policy update
        pi_loss, pi_info = self.compute_pi_loss(data, act, log_prob_pi)
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()
        for p in self.q_params:
            p.requires_grad = True  # allow Q parameters to vary again
        # Update alpha, if required:
        if self.config.alpha[1]:
            alpha_loss = self.compute_alpha_loss(log_prob_pi)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = torch.exp(self.log_alpha.detach())
        # Update target networks:
        self.update_q_targets()
        return {**q_info, **pi_info, **e_info}

    def compute_alpha_loss(self, log_prob_pi):
        """  Compute loss for entropy coefficient alpha  """
        return -(self.log_alpha.exp() * (log_prob_pi + self.target_entropy).detach()).mean()

    def compute_q_loss(self, data):
        """  Compute loss for Q update  """
        q1 = self.q1_network(torch.cat((data.obs, data.actions), dim=-1)).squeeze(-1)
        q2 = self.q2_network(torch.cat((data.obs, data.actions), dim=-1)).squeeze(-1)
        with torch.no_grad():  # just computing targets
            pi_next = self.pi_network(data.next_obs)
            act_next, log_prob_pi_next = self.sampler.get_action_and_log_prob(pi_next)
            q1_target = self.q1_target(torch.cat((data.next_obs, act_next), dim=-1)).squeeze(-1)
            q2_target = self.q2_target(torch.cat((data.next_obs, act_next), dim=-1)).squeeze(-1)
            q_target = torch.min(q1_target, q2_target)
            not_dones = (torch.ones_like(data.terminated, device=device) - data.terminated).squeeze(-1)
            backup = data.rewards.squeeze(-1) + \
                self.config.gamma * not_dones * (q_target - self.alpha * log_prob_pi_next)
        q1_loss = ((q1 - backup) ** 2).mean()
        q2_loss = ((q2 - backup) ** 2).mean()
        q_info = {'q1': torch.mean(q1).item(), 'q2': torch.mean(q2).item()}
        q_loss = q1_loss + q2_loss
        return q_loss, q_info

    def compute_pi_loss(self, data, act, log_prob_pi):
        """  Compute loss for policy network  """
        inputs = torch.cat((data.obs, act), dim=-1)
        q1_pi = self.q1_network(inputs).squeeze(-1)
        q2_pi = self.q2_network(inputs).squeeze(-1)
        q_pi = torch.min(q1_pi, q2_pi)
        pi_loss = (self.alpha * log_prob_pi - q_pi).mean()
        pi_info = {'pi_loss': pi_loss.item()}
        return pi_loss, pi_info

    def update_q_targets(self):
        """  Update target networks via Polyak averaging  """
        with torch.no_grad():
            for p, p_targ in zip(self.q1_network.parameters(), self.q1_target.parameters()):
                p_targ.data.mul_(self.config.polyak)
                p_targ.data.add_((1 - self.config.polyak) * p.data)
            for p, p_targ in zip(self.q2_network.parameters(), self.q2_target.parameters()):
                p_targ.data.mul_(self.config.polyak)
                p_targ.data.add_((1 - self.config.polyak) * p.data)

    def run_evaluation(self, num_episodes=0, deterministic=False):
        """  Run episodes in order to gauge progress  """
        if num_episodes == 0:
            num_episodes = self.config.evaluation_ep
        if num_episodes > 0:
            results = Box({'rewards': [], 'entropies': [], 'max_ent_rew': [], 'lengths': [], 'info': {},
                           'q1': [], 'q2': [], 'q_min': [], 'q_true': [], 'td_error1': [], 'td_error2': []})
        else:
            results = Box({})
        for j in range(num_episodes):
            obs, info = self.reset_env(self.eval_env)
            terminated, truncated = False, False
            ep_rew, ep_len, ep_ent, ep_q1, ep_q2, ep_q, ep_qt, ep_info = [], 0, [], [], [], 0, [], {}
            self.concatenate_dict_of_lists(ep_info, info)
            while not terminated and not truncated:
                action = self.get_action(obs, deterministic=deterministic)
                with torch.no_grad():
                    torch_obs = torch.from_numpy(obs).to(device).float()
                    torch_act = torch.from_numpy(action).to(device).float()
                    pi = self.pi_network(torch_obs)
                    _, log_prob = self.sampler.get_action_and_log_prob(pi, n_samples=10)
                    q1_pred = self.q1_network(torch.cat((torch_obs, torch_act), dim=-1)).item()
                    q2_pred = self.q2_network(torch.cat((torch_obs, torch_act), dim=-1)).item()
                    q1_targ = self.q1_target(torch.cat((torch_obs, torch_act), dim=-1)).item()
                    q2_targ = self.q2_target(torch.cat((torch_obs, torch_act), dim=-1)).item()
                    q_pred = min(q1_pred, q2_pred)
                    q_targ = min(q1_targ, q2_targ)
                trunc = ep_len == self.config.max_ep_length - 1
                next_obs, reward, terminated, truncated, info = self.step_env(self.eval_env, action, trunc)
                obs = next_obs
                ep_rew += [reward]
                ep_len += 1
                ep_ent += [torch.mean(-log_prob).item()]
                ep_q1 += [q1_pred]
                ep_q2 += [q2_pred]
                ep_q += q_pred
                ep_qt += [q_targ]
                self.concatenate_dict_of_lists(ep_info, info)
            results.rewards.append(sum(ep_rew))
            results.lengths.append(ep_len)
            results.entropies.append(sum(ep_ent)/ep_len)
            results.max_ent_rew.append(sum(ep_rew) + self.alpha.item()*sum(ep_ent))
            for k, v in ep_info.items():
                ep_info[k] = sum(v)
            self.concatenate_dict_of_lists(results.info, ep_info)
            results.q1.append(sum(ep_q1) / ep_len)
            results.q2.append(sum(ep_q2) / ep_len)
            results.q_min.append(ep_q / ep_len)
            q_true = self.compute_q(ep_rew, ep_ent)
            results.q_true.append(q_true)
            td_error1 = self.compute_td_error(np.array(ep_rew), np.array(ep_q1), np.array(ep_qt))
            td_error2 = self.compute_td_error(np.array(ep_rew), np.array(ep_q2), np.array(ep_qt))
            results.td_error1.append(td_error1)
            results.td_error2.append(td_error2)
        return results

    def compute_q(self, ep_reward, ep_entropy):
        full_rewards = np.array(ep_reward) + self.alpha.item() * np.array(ep_entropy)
        t = len(ep_reward)
        q = [ep_reward[i] + np.sum(self.config.gamma ** np.arange(1, t - i) * full_rewards[i + 1:]) for i in range(t)]
        return sum(q) / t

    def compute_td_error(self, ep_reward, ep_q, ep_qt):
        return np.sum((ep_reward[:-1] + self.config.gamma * ep_qt[1:] - ep_q[:-1]) ** 2) / ep_reward.shape[0]

    def update_logging(self, loss_info, evaluation_stoch, evaluation_det, steps):
        """  Update TensorBoard logging, reset buffer logging quantities  """
        for k, v in loss_info.items():
            self.logger.log_mean_value('Learning/' + k, v, steps)
        self.logger.log_mean_value('Learning/alpha', [self.alpha.item()], steps)
        for k, v in evaluation_stoch.items():
            if k == 'info':
                for k_info, v_info in v.items():
                    self.logger.log_mean_value('Eval_stoch/Info/' + k_info, v_info, steps)
            else:
                self.logger.log_mean_value('Eval_stoch/' + k, v, steps)
        for k, v in evaluation_det.items():
            if k == 'info':
                for k_info, v_info in v.items():
                    self.logger.log_mean_value('Eval_det/Info/' + k_info, v_info, steps)
            else:
                self.logger.log_mean_value('Eval_det/' + k, v, steps)
        self.logger.log_mean_value('Train/rewards', self.buffer.episode_rewards[:-1], steps)
        self.logger.log_mean_value('Train/lengths', self.buffer.episode_lengths[:-1], steps)
        self.buffer.reset_logging()
        self.logger.flush()

    def save_training(self, total_steps, last_checkpoint):
        """  Save networks, as required.  Update last_checkpoint.  """
        if total_steps // self.config.checkpoint_every > last_checkpoint:  # periodically keep checkpoint
            training = Box({'pi': self.pi_network.state_dict(),
                            'q1': self.q1_network.state_dict(),
                            'q2': self.q2_network.state_dict(),
                            'steps': total_steps})
            if self.config.alpha[1]:
                training.log_a = self.log_alpha.item()
            if self.config.enable_restart:
                training.q1_t = self.q1_target.state_dict()
                training.q2_t = self.q2_target.state_dict()
                training.pi_opt = self.pi_optimizer.state_dict()
                training.q_opt = self.q_optimizer.state_dict()
                if self.config.alpha[1]:
                    training.a_opt = self.alpha_optimizer.state_dict()
            torch.save(training, os.path.join(self.config.model_folder, 'model-latest.pt'))
            last_checkpoint += 1
            if self.config.enable_restart:  # save buffer
                with open(os.path.join(self.config.model_folder, 'buffer-latest.pkl'), 'wb') as buffer_file:
                    pickle.dump(self.buffer, buffer_file)
        return last_checkpoint

    def reset_learning(self):
        """  Reset networks and optimizers to combat primacy bias  """
        self.initialize_networks(reset=True)
        self.initialize_optimizers(reset=True)

    def test(self):
        """  Test agent on a prescribed number of episodes  """
        self.eval_env = get_env_object(self.config)
        _, _ = self.initialize_networks()
        self.sampler = get_sampler(self.config.pi_network)
        results = self.run_evaluation(self.config.test_episodes, deterministic=True)
        with open(os.path.join(self.config.model_folder, 'test_output.pkl'), 'wb') as output_file:
            pickle.dump(results, output_file)

    def recycle_dormant(self):
        """  Recycle dormant neurons in policy, Q networks  """
        data = self.buffer.sample(self.config.redo_batch_size)
        pi_input = data.obs
        q_input = torch.cat((data.obs, data.actions), dim=-1)
        inputs = [pi_input, q_input, q_input]
        configs = [self.config.pi_network, self.config.q_network, self.config.q_network]
        models = [self.pi_network, self.q1_network, self.q2_network]
        q2_param_start = len(self.q_optimizer.state_dict()['state'].keys()) / 2
        optimizers = [(self.pi_optimizer, 0), (self.q_optimizer, 0), (self.q_optimizer, q2_param_start)]
        for i, c, m, o in zip(inputs, configs, models, optimizers):
            self.reset_neurons(i, c, m, o, self.config.redo_tau)

    @staticmethod
    def reset_neurons(data, config, model, optimizer, redo_tau):
        """
        Recycle dormant neurons in linear layers of a given network, as well as its corresponding optimizer.
        Note that the PyTorch linear layer is given by y = xA^T + b, and that optimizer argument contains
        starting point for relevant parameters because Q1, Q2 networks share an optimizer.
        """
        # Create network hooks:
        activation = OrderedDict()

        def get_activation(name):
            def hook(model, data_in, layer_out):
                activation[name] = layer_out.detach()
            return hook

        fresh_net = get_network_object(config).to(device)
        hooks = []
        for n, l in model.named_modules():
            if isinstance(l, torch.nn.modules.linear.Linear):
                handle = l.register_forward_hook(get_activation(n))
                hooks.append(handle)
        model(data)
        # Compute normalized scores:
        for k, v in activation.items():
            activation[k] = torch.mean(torch.abs(v), dim=0) / (torch.mean(torch.abs(v)) + 1.e-9)
        # Reset / zero network layers and optimizer where necessary:
        layer, dormant = 0, None
        optimizer, p0 = optimizer
        for k in activation.keys():
            if layer > 0:
                model.state_dict()[k + '.weight'][:, dormant] = 0.
                optimizer.state_dict()['state'][layer * 2]['exp_avg'][:, dormant] = 0.
                optimizer.state_dict()['state'][layer * 2]['exp_avg_sq'][:, dormant] = 0.
            dormant = torch.where(activation[k] <= redo_tau)[0]
            model.state_dict()[k + '.weight'][dormant, :] = fresh_net.state_dict()[k + '.weight'][dormant, :]
            model.state_dict()[k + '.bias'][dormant] = fresh_net.state_dict()[k + '.bias'][dormant]
            optimizer.state_dict()['state'][layer * 2 + p0]['exp_avg'][dormant, :] = 0.
            optimizer.state_dict()['state'][layer * 2 + p0]['exp_avg_sq'][dormant, :] = 0.
            optimizer.state_dict()['state'][layer * 2 + 1 + p0]['exp_avg'][dormant] = 0.
            optimizer.state_dict()['state'][layer * 2 + 1 + p0]['exp_avg_sq'][dormant] = 0.
            layer += 1
        [h.remove() for h in hooks]  # don't want hooks outside this method

    @staticmethod
    def step_env(env, action, truncated):
        """ Steps input environment, accommodating Gym, Gymnasium, and Safety Gymnasium APIs"""
        step_output = env.step(action)
        if len(step_output) == 4:  # gym
            next_obs, reward, terminated, info = step_output
        else:  # gymnasium
            next_obs, reward, terminated, truncated, info = step_output
        return next_obs, reward, terminated, truncated, info

    @staticmethod
    def reset_env(env):
        """  Resets an environment, accommodating Gym, Gymnasium APIs """
        outputs = env.reset()
        if isinstance(outputs, tuple):
            return outputs
        else:
            return outputs, {}

    @staticmethod
    def concatenate_dict_of_lists(base_dict, new_dict):
        for k, v in new_dict.items():
            if k not in base_dict:
                base_dict[k] = [v]
            else:
                base_dict[k] += [v]


if __name__ == '__main__':
    """  Runs SoftActorCritic training or testing for a given input configuration file  """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Configuration file to run', required=True)
    parser.add_argument('--mode', default='train', required=False, help='mode ("train" or "test")')
    parser.add_argument('--seed', help='random seed', required=False, type=int, default=0)
    parser.add_argument('--prior', help='use prior training', required=False, type=int, default=0)
    in_args = parser.parse_args()
    full_config = os.path.join(os.getcwd(), in_args.config)
    print(full_config)
    sys.stdout.flush()
    with open(full_config, 'r') as f1:
        config1 = Box(json.load(f1))
    config1.seed = in_args.seed
    if in_args.prior > 0 or in_args.mode.lower() == 'test':
        config1.use_prior_nets = True
    sac_object = SoftActorCritic(config1)
    if in_args.mode.lower() == 'train':
        sac_object.train()
    else:
        sac_object.test()
