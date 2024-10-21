import gymnasium
import safety_gymnasium
import time
import math
import numpy as np
import random
import cv2
from mogwai import Mogwais
from safety_gymnasium.assets.mocaps.gremlins import Gremlins
from safety_gymnasium.assets.geoms import Hazards, Pillars
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0
from gentle.common.modified_safety_wrapper import ModifiedSafetyGymnasium
from gentle.common.objects_info_wrapper import SphericalObjectsInfoWrapper
from gentle.custom_envs.make_env_util import make_custom_arranged_env
from gentle.custom_envs.empty import EmptySafetyGymnasium
from gentle.custom_envs.custom_env_wrappers import MultiHazardEnv, SpecifiedSingleUseGoalEnv, AntiGoalEnv, VelocityReward
from gentle.custom_envs.utility_wrappers import MultiEnvWrapper




# example task with random placements
class CustomLevelX(EmptySafetyGymnasium):

    def __init__(self, config, num_mogwais=6, **kwargs):
        super().__init__(config=config)

        rad = 1.7
        self.placements_conf.extents = [-rad, -rad, rad, rad]

        locations = [
                [0.01, -0.01],
                [-0.01, 0.01],
                [-0.01, -0.01],
                [0.01, 0.01],
                [0.02, 0.01],
                [0.01, 0.02],
                [-0.02, 0.01],
                [-0.01, 0.02],
                [0.02, -0.01],
                [0.01, -0.02],
                [-0.02, -0.01],
                [-0.01, -0.02],


            ]
        locations = locations[:num_mogwais] 

        self.mogs = Mogwais(num=num_mogwais, keepout=0.0, color=np.array([0,1,1,1]),
            locations=locations,
            target_positions = [(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0),(0,0)][:num_mogwais],
        )

        # gremlins
        self._add_mocaps(self.mogs)

        # one green
        self._add_geoms(Hazards(num=1, keepout=0.18, color=np.array([0,1,0,1]), name="green"))

        # one red
        self._add_geoms(Hazards(num=1, keepout=0.18, color=np.array([1,0,0,1]), name="red"))



class ControlMogwaisClean(gymnasium.Wrapper):

    def __init__(self, env, num_mogwais=6):
        super().__init__(env)

        # these are target points for the wandering mogwais
        # they randomly spawn a target location nearby, and accelerate towards it
        # when they get too close, it respawns somewhere else, resulting in them
        # wandering around semi-randomly
        self.wanderer_positions = []
        self.wanderer_velocities = []
        self.wanderer_targets = []
        self.N = num_mogwais

    def dist(self, p1, p2):
        dx = p1[0]-p2[0]
        dy = p1[1]-p2[1]
        return math.sqrt(dx*dx + dy*dy)

    def reset(self, **kwargs):
        state, info = self.env.reset()
        self.wanderer_positions = []
        self.wanderer_velocities = []
        self.wanderer_targets = []
        for i in range(self.N):
            net = 1.5
            self.wanderer_positions.append([random.uniform(-net,net), random.uniform(-net,net)])
            self.wanderer_velocities.append([0.0, 0.0])
            self.wanderer_targets.append([self.wanderer_positions[i][0], self.wanderer_positions[i][1]])
        return state, info

    def step(self, a):
        state, reward, term, trunc, info = self.env.step(a)

        # step the wanderers ---------------------------------
        for i in range(self.N):
            pos = self.wanderer_positions[i]
            vel = self.wanderer_velocities[i]
            target = self.wanderer_targets[i]

            while(self.dist(pos, target) < 0.3):
                net = 1.2
                target[0] = pos[0] + random.uniform(-net,net)
                target[1] = pos[1] + random.uniform(-net,net)

                # bias toward origin
                target[0] *= 0.9
                target[1] *= 0.9

                target[0] = np.clip(target[0], -1.8, 1.8)
                target[1] = np.clip(target[1], -1.8, 1.8)

            # accelerate toward target
            acc = 0.0005
            dx = target[0] - pos[0]
            dy = target[1] - pos[1]
            norm = abs(dx)+abs(dy)
            dx = (dx/norm)*acc
            dy = (dy/norm)*acc

            vel[0] += dx
            vel[1] += dy

            # cap velocity
            maxv = 0.02
            vel[0] = np.clip(vel[0], -maxv, maxv)
            vel[1] = np.clip(vel[1], -maxv, maxv)

            # random abrupt disturbances
            if random.random() < 0.03:
                vel[0] = random.uniform(-0.01, 0.01)
                vel[1] = random.uniform(-0.01, 0.01)

            # move
            pos[0] += vel[0]
            pos[1] += vel[1]

            # write
            self.wanderer_positions[i] = pos
            self.wanderer_velocities[i] = vel
            self.wanderer_targets[i] = target


        # assign positions -----------------------------------
        np_wanderers = [np.asarray(x) for x in self.wanderer_positions]
        self.env.unwrapped.unwrapped.task.mogs.target_positions = np_wanderers

        return state, reward, term, trunc, info



# =========================================================================


def make_clean_env(config, modified=True):
    rm = None if config["render_mode"]=="None" else config["render_mode"]
    env = make_custom_arranged_env(CustomLevelX, "Car", rm, rebuild=True, **config["environment"]["env_kwargs"])
    env = MultiHazardEnv(env, ["red"])
    env = SpecifiedSingleUseGoalEnv(env, "green", goal_touch_bonus=1.0)
    env = AntiGoalEnv(env, "red", multiplier=config["environment"]["antigoal_multiplier"], fade_distance=1.0, end_on_collision=True) # additional reward for moving away from red
    env = VelocityReward(env, config["environment"]["velocity_reward"])
    env = ControlMogwaisClean(
        env, 
        num_mogwais=config["environment"]["env_kwargs"]["num_mogwais"]
    )
    if modified:
        return ModifiedSafetyGymnasium(env, mod_dict={"scale":config["environment"]["mod_config"]["scale"]})
    return env




if __name__ == "__main__":
    from clean_config import clean_config

    full_config = clean_config()
    full_config.render_mode = "rgb_array"

    env = make_clean_env(full_config)

    for ep in range(100):

        print("Episode", ep)
        s, info = env.reset()

        for i in range(800):
            a = env.action_space.sample()
            ns, r, term, trunc, info = env.step(a)

            rgb = env.render()
            cv2.imshow("render", rgb[:,:,::-1])
            cv2.waitKey(1)

            if term or trunc:
                break