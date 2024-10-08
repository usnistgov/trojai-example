import torch
import os
import json
from rl_starter_files_code.utils.format import get_obss_preprocessor

from src.public.trojai_gru_minigrid_arch import GRUCNNActorCriticModel


class ColorfulMemoryAgent:
    """An agent for colorful memory.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir, argmax=False, num_envs=1, device=None,
                 config_name='reduced-config.json'):
        obs_space, self.preprocess_obss = get_obss_preprocessor(obs_space)
        state = torch.load(os.path.join(model_dir, 'model.pt'), map_location=device)
        with open(os.path.join(model_dir, config_name), 'r') as f:
            args = json.load(f)
        self.acmodel = GRUCNNActorCriticModel(obs_space, action_space,
                                              channels=args["gru_model_channels"],
                                              actor_linear_mid_dims=args["gru_model_actor_linear_mid_dims"],
                                              critic_linear_mid_dims=args["gru_model_critic_linear_mid_dims"],
                                              gru_hidden_shape=args["gru_model_hidden_shape"],
                                              gru_n_layers=args["gru_model_n_layers"])
        self.acmodel.load_state_dict(state['model_state'])
        self.acmodel.to(device)
        self.acmodel.eval()

        self.num_envs = num_envs
        self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)
        self.argmax = argmax
        self.device = device

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        with torch.no_grad():
            dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        masks = 1 - torch.tensor(dones, dtype=torch.float, device=self.device).unsqueeze(1)
        self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
