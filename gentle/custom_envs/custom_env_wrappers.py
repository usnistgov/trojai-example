import gymnasium
import safety_gymnasium
import numpy as np

'''
A bunch of wrappers that fix some stuff in safety-gymnasium. These assume that SphericalObjectsInfoWrapper is used.

MultiHazardEnv: env that proporly tracks costs from multiple groups of hazards

SpecifiedSingleUseGoalEnv: env that sets the first instance of some object as a goal
    This environment terminates when the goal is reached.

'''


def xy_dist(v1, v2):
    dx = v1[0] - v2[0]
    dy = v1[1] - v2[1]
    dist = np.sqrt(dx*dx + dy*dy)
    return dist


# class that properly tracks costs for multiple groups of hazards.
# the base safety gym does not do this correctly.
class MultiHazardEnv(gymnasium.Wrapper):

    def __init__(self, env, names_of_hazard_groups=[], hazards_are_terminal=False):
        super().__init__(env)
        self.names_of_hazard_groups = names_of_hazard_groups
        self.hazards_are_terminal = hazards_are_terminal

    def step(self, a):
        state, reward, term, trunc, info = super().step(a)
        cost_hazards = 0.0
        agent_pos = info["agent"]["position"]

        for obj in info["objects"]:
            if obj["type"] in self.names_of_hazard_groups:
                dist_away = xy_dist(agent_pos, obj["position"])
                if dist_away < obj["radius"]:
                    cost_hazards += obj["cost"]

                    if self.hazards_are_terminal:
                        term = True

        # overwrite costs
        info["cost_hazards"] = cost_hazards
        info["cost_sum"] = cost_hazards
        info["cost"] = cost_hazards
        return state, reward, term, trunc, info


# class that sets a specific object as the goal and provides rewards for moving towards that goal
# a bonus is provided for touching the goal, which also ends the environment.
# this differs from safety gym in which the goal respawns in the same episode

# NOTE: This overwrites reward!!!!!!

class SpecifiedSingleUseGoalEnv(gymnasium.Wrapper):

    def __init__(self, env, name_of_goal, goal_touch_bonus=10.0):
        super().__init__(env)
        self.goal_name = name_of_goal
        self.goal_touch_bonus = goal_touch_bonus

    def first_object_by_name(self, info, name):
        for obj in info["objects"]:
            if obj["type"] == name:
                return obj
        return None

    def step(self, a):
        state, reward, term, trunc, info = super().step(a)
        goal_obj = self.first_object_by_name(info, self.goal_name)
        goal_pos = goal_obj["position"]
        agent_pos = info["agent"]["position"]
        agent_vel = info["agent"]["velocity"]

        d_goal = xy_dist(agent_pos, goal_pos)
        d_goal_2 = xy_dist(np.array(agent_pos)+np.array(agent_vel), goal_pos)

        # overwrite reward to be distance gained towards goal
        reward = d_goal - d_goal_2

        # if we are in the goal, give us a bonus, but also end episode
        if d_goal < goal_obj["radius"]:
            reward += self.goal_touch_bonus
            term = True
            info["success"] = True

        if (term or trunc) and "success" not in info:
            info["success"] = False

        return state, reward, term, trunc, info


# class that provides a reward for moving away from a given object
# this can be used, i.e. to push agent away from incorrect goals.
# this does not introduce a reward or penalty for touching the object,
# but does have a multiplier to set its strength relative to the SpecifiedSingleUseGoalEnv reward.
class AntiGoalEnv(gymnasium.Wrapper):

    def __init__(self, env, name_of_goal, multiplier=1.0, fade_distance=0.5, end_on_collision=True):
        super().__init__(env)
        self.goal_name = name_of_goal
        self.multiplier = multiplier
        self.fade_distance = fade_distance
        self.end_on_collision = end_on_collision

    def first_object_by_name(self, info, name):
        for obj in info["objects"]:
            if obj["type"] == name:
                return obj
        return None

    def step(self, a):
        state, reward, term, trunc, info = super().step(a)
        goal_obj = self.first_object_by_name(info, self.goal_name)
        goal_pos = goal_obj["position"]
        agent_pos = info["agent"]["position"]
        agent_vel = info["agent"]["velocity"]

        d_goal = xy_dist(agent_pos, goal_pos)
        d_goal_2 = xy_dist(np.array(agent_pos)+np.array(agent_vel), goal_pos)
        fade = 1.0 - np.clip(d_goal / self.fade_distance, 0.0, 1.0)

        # overwrite reward to be distance moved away from object
        reward += -1.0*(d_goal - d_goal_2)*self.multiplier*fade

        if d_goal < goal_obj["radius"] and self.end_on_collision:
            term = True

        return state, reward, term, trunc, info


# adds a reward for high velocity, to encourage non-wiggly movement
class VelocityReward(gymnasium.Wrapper):

    def __init__(self, env, multiplier):
        super().__init__(env)
        self.multiplier = multiplier

    def step(self, a):
        state, reward, term, trunc, info = super().step(a)
        agent_vel = info["agent"]["velocity"]
        agent_vel = np.linalg.norm(agent_vel)
        reward += agent_vel*self.multiplier
        return state, reward, term, trunc, info


class GlobalPosition(gymnasium.Wrapper):

    def reset(self):
        s, info = self.env.reset()
        global_p = self.extract_global(info)
        s = np.concatenate((global_p, s))
        return s, info

    def step(self, a):
        s, r, d, info = self.env.step(a)
        global_p = self.extract_global(info)
        s = np.concatenate((global_p, s))
        return s, r, d, info

    def extract_global(self, info):
        agent_pos = info["agent"]["position"]
        yaw = info["agent"]["yaw"]
        return np.asarray([
            agent_pos[0],
            agent_pos[1],
            agent_pos[2],
            math.cos(yaw),
            math.sin(yaw)
        ])


# THIS IS INCOMPLETE
# adds an image to the state space, so that state is now (lidar, image)
class AddImageToState(gymnasium.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gymnasium.spaces.Tuple(
            env.observation_space, 
            Box(0.0, 1.0, shape=(3,96,96,))
        )

    def observation(self, obs):
        rgb = self.env.render()

        # crop out the middle 96x96

        # convert from 0-255 to 0-1

        # convert from 96,96,3 to 3,96,96