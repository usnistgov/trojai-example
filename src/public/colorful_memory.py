import logging
from typing import Union

import gymnasium as gym
import numpy as np
from minigrid.core.constants import COLORS, COLOR_TO_IDX, IDX_TO_COLOR, OBJECT_TO_IDX, TILE_PIXELS
from minigrid.core.world_object import WorldObj, Wall
from minigrid.minigrid_env import MiniGridEnv, MissionSpace
from minigrid.core.grid import Grid
from minigrid.utils.rendering import fill_coords, point_in_rect, point_in_circle

NON_ITEM_COLORS = [color for color in COLORS if color != 'green']
MY_KEY_IDX = 11  # just as if we added it to the minigrid.OBJECT_TO_IDX dict
MY_BALL_IDX = 12
TILE_SIZE_IN_PIXELS = 16  # this is what gym_minigrid returns by default,

logger = logging.getLogger(__name__)


class KeyInColor(WorldObj):
    """ Key object from MiniGrid, but with a colored background. """
    def __init__(self, color='blue', bg_color=None):
        super().__init__('key', color)
        self.bg_color = bg_color
        if self.color == self.bg_color:
            raise ValueError("object color and background color cannot be the same!")

    def encode(self):
        bg_idx = 255 if self.bg_color is None else COLOR_TO_IDX[self.bg_color]
        # technically the third value is the object state, but we abuse it here to store the bg color
        return MY_KEY_IDX, COLOR_TO_IDX[self.color], bg_idx

    @staticmethod
    def decode(type_idx, color_idx, state):
        color = IDX_TO_COLOR[color_idx]
        bg_color = IDX_TO_COLOR[state] if state != 255 else None
        return KeyInColor(color=color, bg_color=bg_color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # background color
        if self.bg_color:
            fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.bg_color])

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class BallInColor(WorldObj):
    """ Ball object from MiniGrid, but with a colored background. """
    def __init__(self, color='blue', bg_color=None):
        super().__init__('ball', color)
        self.bg_color = bg_color
        if self.color == self.bg_color:
            raise ValueError("object color and background color cannot be the same!")

    def encode(self):
        bg_idx = 255 if self.bg_color is None else COLOR_TO_IDX[self.bg_color]
        # technically the third value is the object state, but we abuse it here to store the bg color
        return MY_BALL_IDX, COLOR_TO_IDX[self.color], bg_idx

    @staticmethod
    def decode(type_idx, color_idx, state):
        color = IDX_TO_COLOR[color_idx]
        bg_color = IDX_TO_COLOR[state] if state != 255 else None
        return BallInColor(color=color, bg_color=bg_color)

    def can_pickup(self):
        return True

    def render(self, img):
        # background color
        if self.bg_color:
            fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.bg_color])

        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


# altered MiniGrid function to use non-native objects above, used in ColorfulMemoryBase._get_image_obs
def alt_grid_decode(array):
    """
    Decode an array grid encoding back into a grid, modified from MiniGrid.Grid.decode in order to use non-native
        MiniGrid objects.
    """

    width, height, channels = array.shape
    assert channels == 3

    vis_mask = np.ones(shape=(width, height), dtype=np.bool)

    grid = Grid(width, height)
    for i in range(width):
        for j in range(height):
            type_idx, color_idx, state = array[i, j]
            if type_idx == MY_KEY_IDX:
                v = KeyInColor.decode(type_idx, color_idx, state)
            elif type_idx == MY_BALL_IDX:
                v = BallInColor.decode(type_idx, color_idx, state)
            else:
                v = WorldObj.decode(type_idx, color_idx, state)
            grid.set(i, j, v)
            vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])

    return grid, vis_mask


class ColorfulMemoryBase(MiniGridEnv):
    """
    Clean ColorfulMemory environment, a modified version of gym_minigrid.envs.memory.MemoryEnv.
    """

    def __init__(
            self,
            seed=None,
            size=7,
            mode='simple',
            random_length=False,
            max_steps=250,
            render_mode=None):
        """
        Initialize the environment.
        :param seed: (int) The seed for the random number generator of the environment.
        :param size: (int) Length and width of the grid in grid squares. Must be an odd number greater than or equal to
            7.
        :param mode: (str) How to return agent observations, options are:
            - 'simple': Return the default (7, 7, 3) view of the observation.
            - 'rgb': Convert the original (7, 7, 3) observation into an image like that used by env.render, but that
                only show the view of the agent.
        :param random_length: (bool) Let the length of the hallway vary per episode (every reset). Only works for grid
            sizes greater than 7, and random lengths very between that when size=7 and the give grid size. Untested with
            "True".
        :param max_steps: (int) The number of steps to allow the agent to get to the goal, before restarting.
        :param render_mode: (str) How the environment should be rendered or if it should be rendered. Options are None
            (to not render at all), 'human' for pygame rendering with no need to call env.render(), or 'rgb_array' to
            return an RGB numpy array when calling env.render()
        """
        self.mode = mode
        self.random_length = random_length
        self.mid_grid = None  # helpful value to keep track of
        self.hallway_end = None  # helpful value to keep track of

        # reset is called at the end of super().__init__
        super().__init__(
            mission_space=MissionSpace(lambda: 'go to the matching object at the end of the hallway'),
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=False,
            render_mode=render_mode
        )
        # do this after call to super().__init__ or these values will be re-defined
        if self.mode == 'simple':
            obs_shape = (self.agent_view_size, self.agent_view_size, 3)
        else:
            obs_shape = (self.width * TILE_SIZE_IN_PIXELS, self.height * TILE_SIZE_IN_PIXELS, 3)
        self.observation_space = gym.spaces.Box(0, 255, shape=obs_shape, dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(3)
        self.reset(seed=seed)

    def _gen_grid(self, width, height):
        """
        Generate the memory grid. Called by reset method.
        :param width: (int) Width of the grid.
        :param height: (int) Height of the grid.
        :return: (Grid) The complete grid object.
        """
        self.grid = Grid(width, height)

        assert height % 2 == 1
        mid_grid = self.mid_grid
        upper_room_wall = mid_grid - 2
        lower_room_wall = mid_grid + 2

        # get a random hallway length if specified
        if self.random_length:
            hallway_end = self._rand_int(4, width - 2)
        else:
            hallway_end = width - 3
        self.hallway_end = hallway_end

        # Start room
        for j in range(mid_grid - 2, mid_grid + 3):
            self.grid.set(0, j, Wall(color=self._rand_color()))
        for i in range(1, 5):
            self.grid.set(i, upper_room_wall, Wall(color=self._rand_color()))
            self.grid.set(i, lower_room_wall, Wall(color=self._rand_color()))
        self.grid.set(4, upper_room_wall + 1, Wall(color=self._rand_color()))
        self.grid.set(4, lower_room_wall - 1, Wall(color=self._rand_color()))

        # Horizontal hallway
        for i in range(5, hallway_end):
            self.grid.set(i, upper_room_wall + 1, Wall(color=self._rand_color()))
            self.grid.set(i, lower_room_wall - 1, Wall(color=self._rand_color()))

        # Vertical hallway
        self.grid.set(hallway_end + 1, 0, Wall(color=self._rand_color()))
        self.grid.set(hallway_end + 1, height - 1, Wall(color=self._rand_color()))
        for j in range(0, height):
            if j != mid_grid:
                # left side of hallway
                self.grid.set(hallway_end, j, Wall(color=self._rand_color()))
            # right side of hallway
            self.grid.set(hallway_end + 2, j, Wall(color=self._rand_color()))

        # Fix the player's start position and orientation
        self.agent_pos = (self._rand_int(1, hallway_end + 1), height // 2)
        self.agent_dir = 0

        # Place objects
        start_room_obj_tile_color = self._rand_elem(NON_ITEM_COLORS)
        start_room_obj = self._rand_elem([KeyInColor, BallInColor])
        self.grid.set(1, height // 2 - 1, start_room_obj(color='green', bg_color=start_room_obj_tile_color))

        other_objs_tile_colors = [self._rand_elem(NON_ITEM_COLORS), self._rand_elem(NON_ITEM_COLORS)]
        other_objs = self._rand_elem([[BallInColor, KeyInColor], [KeyInColor, BallInColor]])
        pos0 = (hallway_end + 1, height // 2 - 2)
        pos1 = (hallway_end + 1, height // 2 + 2)
        self.grid.set(*pos0, other_objs[0](color='green', bg_color=other_objs_tile_colors[0]))
        self.grid.set(*pos1, other_objs[1](color='green', bg_color=other_objs_tile_colors[1]))

        # Choose the target objects
        if start_room_obj == other_objs[0]:
            self.success_pos = (pos0[0], pos0[1] + 1)
            self.failure_pos = (pos1[0], pos1[1] - 1)
        else:
            self.success_pos = (pos1[0], pos1[1] - 1)
            self.failure_pos = (pos0[0], pos0[1] + 1)

    def _get_image_obs(self):
        return self.get_pov_render(TILE_PIXELS)

    def reset(self, *, seed=None, return_info=False, options=None):
        """
        Reset the environment.
        :return: (array) The agent's first observation.
        """
        self.mid_grid = self.height // 2
        # MiniGrid returns more than just the visual observation (e.g. text) in a dict, we just want visual for now
        obs, info = super().reset(seed=seed)
        obs.pop('mission')
        obs['direction'] = np.array(obs['direction']).reshape((1,))

        if self.mode == 'rgb':
            # convert to an RGB image if requested
            obs['image'] = self._get_image_obs()

        return obs, info

    def step(self, action):
        """
        Take a step in the environment.
        :param action: (int or numpy array of length 1) The action to take.
        :return: (array, float, bool, dict) observation, reward, done, info
        """
        if action not in self.action_space:
            if not (isinstance(action, np.ndarray) and action[0] in self.action_space):
                raise RuntimeError("Received action outside of action space! action={}".format(action))

        obs, reward, terminated, truncated, info = MiniGridEnv.step(self, action)

        # MiniGrid returns more than just the visual observation (e.g. text) in a dict, we just want visual for now
        obs['direction'] = np.array(obs['direction']).reshape((1,))
        obs.pop('mission')

        if self.mode == 'rgb':
            obs['image'] = self._get_image_obs()

        # check if the agent got to an end state
        if tuple(self.agent_pos) == self.success_pos:
            reward = self._reward()
            terminated = True
        if tuple(self.agent_pos) == self.failure_pos:
            reward = 0
            terminated = True

        return obs, reward, terminated, truncated, info

    def get_obs_render(self, obs, tile_size=TILE_SIZE_IN_PIXELS):
        """
        Render an agent observation for visualization, modified from MiniGridEnv.get_obs_render
        """

        grid, vis_mask = alt_grid_decode(obs)

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask
        )

        return img


class ColorfulMemoryCfg:
    """ Config object for ColorfulMemoryEnv. """
    ALLOWED_MODES = ['simple', 'rgb']

    def __init__(self, size: int = 7, observation_mode: str = 'simple', random_length: bool = False, seed: int = None,
                 max_steps: int = 250, render_mode: Union[str, None] = None):
        """
        Initialize object.
        :param seed: (int) The seed for the random number generator of the environment.
        :param size: (int) Length and width of the grid in grid squares. Must be an odd number greater than or equal to
            7.
        :param observation_mode: (str) How to return agent observations, options are:
            - 'simple': Return the default (7, 7, 3) view of the observation.
            - 'rgb': Convert the original (7, 7, 3) observation into an image like that used by env.render, but that
                only show the view of the agent.
        :param random_length: (bool) Let the length of the hallway vary per episode (every reset). Only works for grid
            sizes greater than 7, and random lengths very between that when size=7 and the given grid size.
        :param max_steps: (int) The number of steps to allow the agent to get to the goal, before restarting.
        :param render_mode: (str) How the environment should be rendered or if it should be rendered. Options are None
            (to not render at all), 'human' for pygame rendering with no need to call env.render(), or 'rgb_array' to
            return an RGB numpy array when calling env.render()
        """
        self.size = size
        self.mode = observation_mode
        self.random_length = random_length
        self.seed = seed
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.validate()

    def validate(self):
        if self.size is not None:
            if not isinstance(self.size, int):
                msg = "Argument 'size' must be an odd integer >= 7, instead got type {}".format(type(self.size))
                logger.error(msg)
                raise TypeError(msg)
            if self.size % 2 == 0 or self.size < 7:
                msg = "Argument 'size' must be an odd integer >= 7, instead got {}".format(self.size)
                logger.error(msg)
                raise ValueError(msg)
        if self.mode not in self.ALLOWED_MODES:
            msg = "Unexpected mode: {}, allowed modes are: {}".format(self.mode, self.ALLOWED_MODES)
            logger.error(msg)
            raise ValueError(msg)
        if not isinstance(self.random_length, bool):
            msg = "Argument 'random_length' must be type bool, instead got type {}".format(type(self.random_length))
            logger.error(msg)
            raise TypeError(msg)
        if self.seed is not None and not isinstance(self.seed, int):
            msg = "Argument 'seed' must be an integer, instead got type {}".format(type(self.seed))
            logger.error(msg)
            raise TypeError(msg)
        if self.max_steps < 0:
            msg = "'Argument 'max_steps' must be an integer greater than 0, got: {}".format(self.max_steps)
            logger.error(msg)
            raise ValueError(msg)
        render_modes = MiniGridEnv.metadata['render_modes']
        if self.render_mode is not None and self.render_mode not in render_modes:
            msg = f"Invalid 'render_mode' value, must be None or one of the following: {render_modes}"
            logger.error(msg)
            raise ValueError(msg)


class ColorfulMemoryEnv(ColorfulMemoryBase):
    """ Colorful Memory Environment created from a config object. """
    def __init__(self, cfg: ColorfulMemoryCfg):
        """
        Initialize the environment.
        :param cfg: (ColorfulMemoryCfg) The config object for the environment.
        """
        super().__init__(size=cfg.size, mode=cfg.mode, random_length=cfg.random_length, seed=cfg.seed,
                         max_steps=cfg.max_steps, render_mode=cfg.render_mode)
