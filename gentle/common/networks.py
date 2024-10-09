import torch
import numpy as np
from typing import Dict, Optional


class BaseNetwork(torch.nn.Module):
    """  Parent class  """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.process_config()

    def process_config(self):
        """  Process input configuration  """
        self.config.setdefault('discrete', False)     # By default, operate in continuous action spaces

    def forward(self, x: torch.Tensor):
        """  Run network forward pass  """
        pass

    def forward_with_processing(self, x):
        """
        Middle layer to convert from whatever observation format is to torch.Tensor
        and then run forward pass.  In most of this code, x will just be a numpy array;
        this function will need to be overwritten when that is not the case.
        """
        x = torch.from_numpy(x).float()
        return self.forward(x)

    def save(self, file_path: str):
        """  Save a model file  """
        torch.save(self.state_dict(), file_path)

    def restore(self, file_path: str):
        """  Load a model file  """
        self.load_state_dict(torch.load(file_path))


class MLP(BaseNetwork):
    """  Configurable Multi-Layer Perceptron  """
    def __init__(self, config: Dict):
        super().__init__(config)
        layers = []
        sizes = [self.config['obs_dim']] + self.config['sizes'] + [self.config['action_dim']]
        for h in range(len(sizes)-2):
            layers.append(torch.nn.Linear(sizes[h], sizes[h+1]))
            if config['activation'].lower() == 'tanh':
                layers.append(torch.nn.Tanh())
            else:
                layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(sizes[-2], sizes[-1]))
        if config.get('output_activation', '').lower() == 'tanh':
            layers.append(torch.nn.Tanh())
        self.layers = layers
        self.mlp = torch.nn.Sequential(*layers)

    def process_config(self):
        """  Process input configuration  """
        super().process_config()
        if 'obs_dim' not in self.config:
            raise KeyError('obs_dim must be included in network config.')
        if 'action_dim' not in self.config:
            raise KeyError('action_dim must be included in network config.')
        self.config.setdefault('sizes', [100, 100])
        self.config.setdefault('activation', 'tanh')
        if self.config['activation'].lower() not in ['tanh', 'relu']:
            raise ValueError('activation type not recognized.')

    def forward(self, x):
        """  Run forward pass of network  """
        return self.mlp(x)


class CategoricalMLP(MLP):
    """  Categorical MLP policy; used for discrete action spaces  """
    def process_config(self):
        super().process_config()
        self.config['discrete'] = True

    def forward(self, x):
        """  Run forward pass of network  """
        logits = self.mlp(x)
        activation = torch.nn.Softmax(dim=-1)
        return activation(logits)


class GaussianMLP(MLP):
    """  Gaussian MLP policy; used for continuous action spaces  """
    def __init__(self, config):
        super().__init__(config)
        self.mlp = torch.nn.Sequential(*self.layers[:-1])
        self.mu_layer = torch.nn.Linear(self.config['sizes'][-1], self.config['action_dim'])
        self.log_std, self.log_std_layer = None, None
        if self.config['log_std_net']:
            self.log_std_layer = torch.nn.Linear(self.config['sizes'][-1], self.config['action_dim'])
        else:
            self.log_std = torch.nn.Parameter(torch.ones(self.config['action_dim']) * self.config['initial_log_std'])

    def process_config(self):
        """  Process input configuration  """
        super().process_config()
        self.config['discrete'] = 0
        self.config.setdefault('initial_log_std', -0.5)
        self.config.setdefault('min_log_std', None)  # -20 for safety-starter-agents squashed
        self.config.setdefault('max_log_std', None)  # 2 for safety-starter-agents squashed
        assert (self.config['min_log_std'] is None) == (self.config['max_log_std'] is None), "Inconsistent logstd"
        self.config.setdefault('log_std_net', False)

    def forward(self, x):
        """  Run forward pass of network  """
        mu = self.mu_layer(self.mlp(x))
        if self.config['log_std_net']:
            if self.config['min_log_std'] is None:
                std = self.log_std_layer(self.mlp(x)).exp()
            else:
                std = self.log_std_layer(self.mlp(x)).clamp(min=self.config['min_log_std'],
                                                            max=self.config['max_log_std']).exp()
        else:
            if self.config['min_log_std'] is None:
                std = self.log_std.exp()
            else:
                std = self.log_std.clamp(min=self.config['min_log_std'], max=self.config['max_log_std']).exp()
        return mu, std


class AtariNetwork(BaseNetwork):
    """  Neural network for control use in the Atari suite  """
    def __init__(self, config: Dict):
        """  Construct AtariNetwork object  """
        super().__init__(config)
        self.conv_0 = torch.nn.Conv2d(self.config['planes'], self.config['filters'][0], self.config['kernels'][0],
                                      stride=self.config['strides'][0])
        self.conv_1 = torch.nn.Conv2d(self.config['filters'][0], self.config['filters'][1], self.config['kernels'][1],
                                      stride=self.config['strides'][1])
        self.conv_2 = torch.nn.Conv2d(self.config['filters'][1], self.config['filters'][2], self.config['kernels'][2],
                                      stride=self.config['strides'][2])
        side_0 = self.conv2d_size_out(self.config['img_side'], self.config['kernels'][0][0], self.config['strides'][0])
        side_1 = self.conv2d_size_out(side_0, self.config['kernels'][1][0], self.config['strides'][1])
        side_2 = self.conv2d_size_out(side_1, self.config['kernels'][2][0], self.config['strides'][2])
        self.linear_input_size = self.config['filters'][2]*side_2**2
        self.linear_p_0 = torch.nn.Linear(self.linear_input_size, self.config['fully_connected'])
        self.linear_p_1 = torch.nn.Linear(self.config['fully_connected'], self.config['action_dim'])
        self.linear_v = torch.nn.Linear(self.config['fully_connected'], 1)

    def process_config(self):
        """  Process input configuration file  """
        super().process_config()
        self.config.setdefault('img_side', 84)
        self.config.setdefault('planes', 4)
        self.config.setdefault('filters', [32, 64, 64])
        self.config.setdefault('kernels', [(8, 8), (4, 4), (3, 3)])
        self.config.setdefault('strides', [4, 2, 1])
        self.config.setdefault('fully_connected', 512)
        assert 'action_dim' in self.config, "action_dim required in configuration"

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """  Run forward pass of network  """
        x = torch.nn.functional.relu(self.conv_0(x))
        x = torch.nn.functional.relu(self.conv_1(x))
        x = torch.nn.functional.relu(self.conv_2(x))
        x = x.view(-1, self.linear_input_size)
        x = torch.nn.functional.relu(self.linear_p_0(x))
        policies = torch.nn.functional.softmax(self.linear_p_1(x), dim=-1)
        values = self.linear_v(x)
        return policies, values

    def forward_with_processing(self, x):
        """  Pre-processing and forward pass (for training)  """
        x = torch.from_numpy(x).float()
        policies, values = self.forward(x)
        return policies, values

    @staticmethod
    def conv2d_size_out(size: int, kernel: int, stride: int, padding: Optional[int] = 0) -> int:
        """  Returns the output size of one side of a 2D combination  """
        return (size - kernel + 2*padding)//stride + 1


class QuantileEmbeddingNetwork(BaseNetwork):
    """
    Quantile Embedding Network of the form phi(tau)_j = act(norm(sum_i (cosine(i*pi*tau)*wij + bj))),
    where the tau are quantiles, wij are weights, and bj are biases.
    """
    def __init__(self, config: Dict):
        super().__init__(config)
        freqs = torch.arange(1, 1 + self.config['embedding_size']) * np.pi  # precomputing the multiples of pi
        self.register_buffer('freqs', freqs, persistent=False)  # will move to device appropriately but not part of state_dict
        act_layer = torch.nn.Sigmoid() if self.config['qe_act'].lower() == 'sigmoid' else torch.nn.ReLU()
        layers = [torch.nn.Linear(self.config['embedding_size'], self.config['hidden_size']),
                  torch.nn.LayerNorm(self.config['hidden_size']),
                  act_layer]
        self.phi = torch.nn.Sequential(*layers)

    def process_config(self):
        super().process_config()
        assert 'embedding_size' in self.config.keys(), 'Need embedding_size in config'
        assert 'hidden_size' in self.config.keys(), 'Need hidden_size'
        self.config.setdefault('qe_act', 'sigmoid')

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        cosines = torch.cos(x.unsqueeze(-1) * self.freqs)
        return self.phi(cosines)


class QuantileMLP(BaseNetwork):

    def __init__(self, config: Dict):
        super().__init__(config)

        # psi (input embedding)
        sizes = [self.config['obs_dim']] + self.config['psi_sizes'] + [self.config['hidden_size']]
        layers = []
        for h in range(len(sizes) - 1):
            layers.append(torch.nn.Linear(sizes[h], sizes[h + 1]))
            layers.append(torch.nn.LayerNorm(sizes[h+1]))
            layers.append(torch.nn.ReLU())
        self.psi = torch.nn.Sequential(*layers)

        # phi (quantile embedding)
        self.phi = QuantileEmbeddingNetwork(self.config)

        # combination layers to return quantile estimate(s):
        self.combination_layers = torch.nn.Sequential(
            torch.nn.Linear(self.config['hidden_size'], self.config['combination_size']),
            torch.nn.LayerNorm(self.config['combination_size']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['combination_size'], 1))

    def process_config(self):
        """  Process input configuration  """
        super().process_config()
        self.config.setdefault('psi_sizes', [])
        self.config.setdefault('hidden_size', 256)
        self.config.setdefault('embedding_size', 64)
        self.config.setdefault('combination_size', self.config['hidden_size'])

    def forward(self, x_and_tau):
        """ Forward pass. x_and_tau is a list or tuple comprised of both (s, a) input and \tau (mid) input.  """
        x, tau = x_and_tau
        h1 = self.psi(x)
        h2 = self.phi(tau)
        multiplied = torch.mul(h2, h1.unsqueeze(-2))
        return self.combination_layers(multiplied).squeeze(-1)
