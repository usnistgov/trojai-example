import gymnasium
import safety_gymnasium
import numpy as np

'''
Some helpful wrappers for this and that.
'''



# wrapper that swaps between several environments
class MultiEnvWrapper(gymnasium.Wrapper):

    def __init__(self, envs, mode="alternate"):
        super().__init__(envs[0])
        self.edx = 0
        self.envs = envs
        self.mode = mode

    def reset(self):
        if self.mode=="alternate":
            self.edx = (self.edx+1)%len(self.envs)
        else:
            self.edx = np.random.choice(np.arange(len(self.envs)))
        self.env = self.envs[self.edx]
        return super().reset()