import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

import torch_ac

"""
Models are primarily adaptations of https://github.com/lcswillems/rl-starter-files/blob/master/model.py
"""


def linear_w_relu(dims: list, end_relu=True):
    """Helper function for creating sequential linear layers"""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
    if not end_relu:
        del layers[-1]
    return nn.Sequential(*layers)


def model_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:  # from rl-starter-files/model.py
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
    # from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py
    elif classname.find("GRU") != -1:
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)


class TrojaiDRLGRUBackbone(nn.Module, torch_ac.RecurrentACModel):
    """Base class for TrojAI DRL models"""

    def __init__(self, embedding, gru, actor, critic):
        nn.Module.__init__(self)
        self.state_emb = embedding  # Define state embedding
        self.gru = nn.ModuleList(gru)
        self.actor = actor  # Define actor's model
        self.critic = critic  # Define critic's model
        self.value = None
        self.apply(model_init)

    def forward(self, obs, memory):
        batch_size = obs.image.shape[0]
        obs = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.obs_to_embedding_transform(obs, batch_size)

        h_outs = []
        rnn_x_out = x
        state = self.construct_states_from_memory(memory, batch_size)
        for i in range(len(self.gru)):
            h_in = state[i]
            rnn_x_out, h_out = self.gru[i](rnn_x_out, h_in)
            h_outs.append(h_out)

        x = rnn_x_out.reshape(batch_size, self.gru_hidden_shape[-1])
        x_act = self.actor(x)
        x_act = Categorical(logits=F.log_softmax(x_act, dim=1))
        x_crit = self.critic(x)
        value = x_crit.squeeze(1)
        return x_act, value, self.construct_memory_from_states(h_outs, batch_size)

    def construct_states_from_memory(self, memory, batch_size):
        states = []
        idx_1 = 0
        idx_2 = self.gru_hidden_shape[0] * self.gru_n_layers[0]
        shape = (batch_size, self.gru_n_layers[0], self.gru_hidden_shape[0])
        state = memory[:, idx_1: idx_2].reshape(*shape).transpose(0, 1).contiguous()
        states.append(state)
        for i in range(len(self.gru) - 1):
            idx_1 += self.gru_hidden_shape[i] * self.gru_n_layers[i]
            idx_2 = idx_1 + self.gru_hidden_shape[i + 1] * self.gru_n_layers[i + 1]
            shape = (batch_size, self.gru_n_layers[i + 1], self.gru_hidden_shape[i + 1])
            state = memory[:, idx_1: idx_2].reshape(*shape).transpose(0, 1).contiguous()
            states.append(state)
        return states

    def construct_memory_from_states(self, states, batch_size):
        memory = torch.cat([
            states[i].transpose(0, 1).reshape(batch_size, self.gru_hidden_shape[i] * self.gru_n_layers[i])
            for i in range(len(self.gru))
        ], dim=1)
        return memory

    def memory_size(self):
        raise NotImplementedError("Should be implemented in subclass")

    def args_dict(self):
        raise NotImplementedError("Should be implemented in subclass")

    def obs_to_embedding_transform(self, obs, batch_size):
        raise NotImplementedError("Should be implemented in subclass")


class GRUCNNActorCriticModel(TrojaiDRLGRUBackbone):
    def __init__(self, obs_space,
                 action_space,
                 channels=(8, 16, 32),
                 actor_linear_mid_dims=(64,),
                 critic_linear_mid_dims=(64,),
                 gru_hidden_shape=(64, 64),
                 gru_n_layers=(2, 2)
                 ):

        if len(channels) != 3:
            raise ValueError("'channels' must be a tuple or list of length 3")

        self.obs_space = obs_space
        self.action_space = action_space
        self.channels = channels
        self.actor_linear_mid_dims = actor_linear_mid_dims
        self.critic_linear_mid_dims = actor_linear_mid_dims
        self.gru_n_layers = list(gru_n_layers)
        self.gru_hidden_shape = list(gru_hidden_shape)

        c1, c2, c3 = channels
        image_embedding_size = c3
        image_conv = nn.Sequential(
            nn.Conv2d(3, c1, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(c1, c2, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(c2, c3, (2, 2)),
            nn.ReLU()
        )

        if len(gru_hidden_shape) != len(gru_n_layers):
            raise ValueError("'gru_hidden_shape' must be the same length as 'gru_n_layers'!")

        gru = []
        in_size = image_embedding_size
        for i in range(len(self.gru_hidden_shape)):
            gru.append(
                nn.GRU(in_size, self.gru_hidden_shape[i], num_layers=self.gru_n_layers[i],
                       batch_first=True)
            )
            in_size = self.gru_hidden_shape[i]

        actor_dims = [in_size] + list(actor_linear_mid_dims) + [action_space.n]
        critic_dims = [in_size] + list(critic_linear_mid_dims) + [1]
        actor = linear_w_relu(actor_dims)
        critic = linear_w_relu(critic_dims)
        super().__init__(image_conv, gru, actor, critic)

    @property
    def memory_size(self):
        return np.sum([self.gru_hidden_shape[i] * self.gru_n_layers[i] for i in range(len(self.gru_hidden_shape))])

    def obs_to_embedding_transform(self, obs, batch_size):
        return self.state_emb(obs.float()).reshape(batch_size, -1).unsqueeze(1)

    def args_dict(self):
        return {
            'obs_space': self.obs_space,
            'action_space': self.action_space,
            'channels': self.channels,
            'actor_linear_mid_dims': self.actor_linear_mid_dims,
            'critic_linear_mid_dims': self.critic_linear_mid_dims,
            'gru_hidden_shape': self.gru_hidden_shape,
            'gru_n_layers': self.gru_n_layers
        }


class GRUFCActorCriticModel(TrojaiDRLGRUBackbone):
    def __init__(self, obs_space,
                 action_space,
                 actor_linear_mid_dims=(64,),
                 critic_linear_mid_dims=(64,),
                 gru_hidden_shape=(64, 64),
                 gru_n_layers=(2, 2)
                 ):

        nn.Module.__init__(self)
        self.obs_space = obs_space
        self.action_space = action_space
        self.actor_linear_mid_dims = actor_linear_mid_dims
        self.critic_linear_mid_dims = actor_linear_mid_dims
        self.gru_n_layers = list(gru_n_layers)
        self.gru_hidden_shape = list(gru_hidden_shape)

        if len(gru_hidden_shape) != len(gru_n_layers):
            raise ValueError("'gru_hidden_shape' must be the same length as 'gru_n_layers'!")

        self.linear_embedding = nn.Sequential(
            nn.Linear(np.prod(obs_space['image'].shape), 64),
            nn.ReLU()
        )

        gru = []
        in_size = 64
        for i in range(len(self.gru_hidden_shape)):
            gru.append(
                nn.GRU(in_size, self.gru_hidden_shape[i], num_layers=self.gru_n_layers[i],
                       batch_first=True)
            )
            in_size = self.gru_hidden_shape[i]

        actor_dims = [in_size] + list(actor_linear_mid_dims) + [action_space.n]
        critic_dims = [in_size] + list(critic_linear_mid_dims) + [1]
        actor = linear_w_relu(actor_dims)
        critic = linear_w_relu(critic_dims)
        super().__init__(None, gru, actor, critic)

    @property
    def memory_size(self):
        return np.sum([self.gru_hidden_shape[i] * self.gru_n_layers[i] for i in range(len(self.gru_hidden_shape))])

    def obs_to_embedding_transform(self, obs, batch_size):
        return obs.flatten(start_dim=1, end_dim=-1).float().unsqueeze(1)

    def args_dict(self):
        return {
            'obs_space': self.obs_space,
            'action_space': self.action_space,
            'actor_linear_mid_dims': self.actor_linear_mid_dims,
            'critic_linear_mid_dims': self.critic_linear_mid_dims,
            'gru_hidden_shape': self.gru_hidden_shape,
            'gru_n_layers': self.gru_n_layers
        }
