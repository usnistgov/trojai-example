import gymnasium
import safety_gymnasium
import time
import numpy as np
import random
from safety_gymnasium.assets.geoms import Hazards, Pillars
from safety_gymnasium.tasks.safe_navigation.goal.goal_level0 import GoalLevel0
from gentle.common.modified_safety_wrapper import ModifiedSafetyGymnasium
from make_env_util import make_custom_arranged_env


class GoalLevelX(GoalLevel0):
    def __init__(self, config):
        super().__init__(config=config)

        rad = 2.0
        self.placements_conf.extents = [-rad, -rad, rad, rad]

        '''
        Construct two sets of hazards, blue and red, such that there is always
        a pattern of:

        BR
        R

        somewhere in the scene. All other placements random.
        '''

        blue_loc = [random.uniform(-1.5,1.5), random.uniform(-1.5,1.5)]
        red_one = [blue_loc[0], blue_loc[1]+0.4]
        red_two = [blue_loc[0]+0.4, blue_loc[1]]

        self._add_geoms(Hazards(num=6, keepout=0.18, locations=[blue_loc]))
        self._add_geoms(Hazards(
            num=6, 
            keepout=0.18, 
            locations=[red_one, red_two], 
            color=np.array([1,0,0,1]), 
            name="h2" # need a new name so we do not overwrite blue hazards
            )
        )


env = make_custom_arranged_env(GoalLevelX, "Racecar", "human")
env = ModifiedSafetyGymnasium(env, mod_dict={"scale":0.05}) # add cost*scale to reward


# ===============================================================



for ep in range(100):

    print("Episode", ep)
    s, info = env.reset()

    for i in range(500):
        a = env.action_space.sample()
        ns, r, term, trunc, info = env.step(a)
        # env.render()

        if term or trunc:
            env.reset()