import gymnasium as gym
import safety_gymnasium

class ModifiedSafetyGymnasium(gym.Wrapper):
    def __init__(self, base_env, mod_dict={}):
        """
        Makes minor modifications to default SafetyGym configuration, allowing for experimentation on
        unconstrained learning
        :param base_env: (Gymnasium Environment); the environment to wrap
        :param mod_dict: (dict); configuration for modified Safety Gymnasium
        """
        gym.Wrapper.__init__(self, base_env)
        self.mod_config = mod_dict
        # Configure cost/penalty incorporation in reward:
        self.mod_config.setdefault('cost', 'one_indicator')  # one_indicator is default in Safety Gym
        assert self.mod_config["cost"] in ['full', 'one_indicator']
        self.mod_config.setdefault('scale', 0.0)

    def reset(self):
        obs, info = self.env.reset()
        if len(info) > 0 and self.mod_config["cost"] == 'full':
            info['cost_ind'] = info['cost']
            info['cost'] = info['cost_sum']
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.mod_config["cost"] == 'full':
            info['cost_ind'] = info['cost']
            info['cost'] = info['cost_sum']
        reward -= info['cost'] * self.mod_config["scale"]
        return obs, reward, terminated, truncated, info