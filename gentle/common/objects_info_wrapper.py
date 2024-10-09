import gymnasium
import safety_gymnasium
import time
import numpy as np
import math
import matplotlib.pyplot as plt

'''
Adds entries in info about current objects, represented as a point and a radius.
Adds entry about the current agent

info["objects"]:
    List of dicts:
    {
        "type":string name of object class ("vases", "hazards", etc),
        "radius": radius of the spherical representation
        "position": [x,y,z] position of the object
        "rgb": [r,g,b] color of the object
    }

info["agent"]:
    {
        "position": [x,y,z] position of the agent
        "velocity": [dx, dy, dz] velocities of the agent
        "roll": roll angle of the agent (untested)
        "pitch": pitch angle of the agent (untested)
        "yaw": yaw angle of the agent, this is what you want for "heading" in 2D planning
            * NOTE: for some agents, the yaw is offset by a quarter turn
    }
'''
class SphericalObjectsInfoWrapper(gymnasium.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.task = env.unwrapped.task

    def reset(self, *, seed=None, options=None):
        s, info = self.env.reset()
        self.task = self.env.unwrapped.task # prevents bug in which env needs reset() to initialize
        return s, self.modify_info(info)

    def step(self, a):
        s, r, term, trunc, info = self.env.step(a)
        return s, r, term, trunc, self.modify_info(info)

    def get_rpy(self, R):
        # https://stackoverflow.com/a/76153584
        # https://robotics.stackexchange.com/a/8517
        pitch = math.atan2(-R[2][0], math.sqrt(R[0][0]*R[0][0] + R[1][0]*R[1][0]))
        yaw = math.atan2(R[1][0]/math.cos(pitch), R[0][0]/math.cos(pitch))
        roll = math.atan2(R[2][1]/math.cos(pitch), R[2][2]/math.cos(pitch))
        return roll, pitch, yaw


    def modify_info(self, info):
        # get the agent information
        r, p, y = self.get_rpy(self.task.agent.mat)
        info["agent"] = {
            "position":self.task.agent.pos, 
            "velocity":self.task.agent.vel,
            "roll":r,
            "pitch":p,
            "yaw":y
        }

        # get all other objects
        objs = []
        for entry in self.task._obstacles:
            name = entry.name
            sz = entry.size
            positions = entry.pos
            color = entry.color[:3]
            cost = entry.cost if hasattr(entry, "cost") else 0.0
            if np.asarray(positions).shape == (3,):
                objs.append({"type":name, "radius":sz, "position":positions, "rgb":color, "cost":cost})
            else:
                for position in positions:
                    objs.append({"type":name, "radius":sz, "position":position, "rgb":color, "cost":cost})
        info["objects"] = objs

        return info


'''
Plots the current environment using information about objects and agents.
'''
def plot_map(info):
    fig = plt.gcf()
    ax = fig.gca()

    # objects
    for obj in info["objects"]:
        xy = obj["position"][:2]
        rad = obj["radius"]
        clr = color=obj["rgb"]
        plt.scatter([xy[0]], [xy[1]])
        ax.add_patch(plt.Circle(xy, rad, color=clr))

    # agent
    xy = info["agent"]["position"][:2]
    yaw = info["agent"]["yaw"]
    plt.arrow(xy[0], xy[1], np.cos(yaw)*0.2, np.sin(yaw)*0.2, color="r", width=0.03)

    ax.axis('equal')
    plt.savefig("./env.png")




if __name__ == "__main__":

    safety_gymnasium_env = safety_gymnasium.make("SafetyCarGoal2-v0", render_mode="human")
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safety_gymnasium_env)
    env = SphericalObjectsInfoWrapper(env)
    s, info = env.reset()

    # plot and save the initial map
    plot_map(info)

    # play an episode, to show that it matches the plot
    for i in range(10000):
        a = env.action_space.sample()
        ns, r, term, trunc, info = env.step(a)
        env.render()
        time.sleep(0.05)

        if term or trunc:
            env.reset()