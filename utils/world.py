import itertools
import random
import gymnasium as gym

from gym.utils import seeding
from minigrid.minigrid_env import MiniGridEnv, MissionSpace, TILE_PIXELS
from minigrid.core.grid import Grid
from minigrid.core.world_object import Lava, Goal

import numpy as np

ALLOWED_LAVA_COVERAGE = 0.6  # fraction/percent of grid that can be lava
DEFAULT_GRID_SIZE = 9

class RandomLavaWorldEnv(MiniGridEnv):
    def __init__(self, grid_size=9, width=None, height=None, max_steps=250, agent_view_size=7, num_lava_squares=5, mode='simple', seed=None):
        self.py_random = None
        self.np_random = None
        self.mode = mode
        self.obstacle_type = Lava
        self.agent_last_pos = None
        self.step_count = None
        self.carrying = None
        self.allowed_lava_coverage = ALLOWED_LAVA_COVERAGE
        self.max_rand_tries = 100
        self.num_lava_squares = 0
        self.max_lava_squares = 0

        super().__init__(mission_space=MissionSpace(lambda: "avoid the lava and get to the green goal square"),
                        grid_size=grid_size,
                        width=width,
                        height=height,
                        max_steps=max_steps,
                        see_through_walls=True,
                        agent_view_size=agent_view_size)

        self.reset(seed=seed)
        self._set_env_attributes(size=grid_size, width=width, height=height, num_lava_squares=num_lava_squares)

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)
        self.py_random = random.Random(seed)

    def _set_env_attributes(self, size, width, height, num_lava_squares):
        if (width or height) and size:
            self.width = size
            self.height = size
        elif size:
            self.width = size
            self.height = size
        else:
            self.width = width if width is not None else self.width
            self.height = height if height is not None else self.height

        self.num_lava_squares = num_lava_squares if num_lava_squares is not None else self.num_lava_squares

        if width or height or num_lava_squares is not None:
            playable_grid_tiles = (self.width - 1) * (self.height - 1)
            self.max_lava_squares = int(self.allowed_lava_coverage * playable_grid_tiles)

            if not isinstance(self.num_lava_squares, int) and self.num_lava_squares[1] > self.max_lava_squares:
                if self.max_lava_squares <= self.num_lava_squares[0]:
                    self.num_lava_squares = self.max_lava_squares
                else:
                    self.num_lava_squares = (self.num_lava_squares[0], self.max_lava_squares)


        if width or height:
            if self.mode == 'simple':
                obs_shape = (self.agent_view_size, self.agent_view_size, 3)
            else:
                obs_shape = (self.agent_view_size * TILE_PIXELS, self.agent_view_size * TILE_PIXELS, 3)

            self.observation_space = gym.spaces.Dict({'image': gym.spaces.Box(0, 255, shape=obs_shape, dtype=np.uint8),
                                                    'direction': gym.spaces.Box(0, 3, (1,), dtype=int)})

            self.action_space = gym.spaces.Discrete(3)

    def _available_pos_set(self):
        """
        Create set of positions within the grid where objects can be placed.
        :return: (set) Set of 2-tuples of allowed grid positions for objects.
        """
        return set(itertools.product(range(1, self.width - 1), range(1, self.height - 1)))

    def _rand_pos_in_grid(self):
        """
        Return a random position within the grid walls
        """
        return self._rand_pos(1, self.width - 1, 1, self.height - 1)

    def _get_image_obs(self):
        return self.get_pov_render(TILE_PIXELS)

    def no_state_put_obj(self, grid, obj, i, j):
        grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def _dfs_iter(self, mg_grid, visited, x, y):
        """
        Depth first search based check that there exists a path to the goal.
        :param mg_grid: (Grid) The MiniGrid Grid object
        :param visited: (numpy array) 2d grid of visited grid locations, value of 2 means "visited".
        :param x: (int) Grid x coordinate to visit on this iteration
        :param y: (int) Grid y coordinate to visit on this iteration
        :return: (bool) If path to goal was discovered.
        """
        nbs = []
        visited[y, x] = 1  # grid coordinates are backwards, shouldn't matter in general, but helps when debugging
        for i, j in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
            new_x, new_y = x + i, y + j
            cell = mg_grid.get(new_x, new_y)
            if cell:
                if cell.type == 'goal':
                    return True
                else:
                    visited[new_y, new_x] = 2
            elif visited[new_y, new_x]:
                pass
            else:
                nbs.append((new_x, new_y))
        if len(nbs) > 0:
            return any(self._dfs_iter(mg_grid, visited, n[0], n[1]) for n in nbs)
        else:
            return False

    def _path_exists(self, grid):
        """
        Assure that there exists a path in the grid from the first location to the second, assuming a lavaworld grid
        :param grid (Grid) MiniGrid Grid object to check.
        """
        visit_grid = np.zeros((grid.height, grid.width), dtype=np.int8)
        visit_grid[:, 0] = 2
        visit_grid[:, -1] = 2
        visit_grid[0, :] = 2
        visit_grid[-1, :] = 2
        # Note: grid y axis is inverted, but shouldn't matter for this application

        return self._dfs_iter(grid, visit_grid, 1, 1)

    def _gen_grid(self, width, height):
        """
        Try to set the grid using current parameters.
        :param width (int) Width of the grid.
        :param height (int) Height of the grid.
        :return: None
        """
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        grid = Grid(width, height)

        # Generate the surrounding walls
        grid.wall_rect(0, 0, width, height)

        available_pos = self._available_pos_set()

        # Place the agent in the top-left corner
        self.agent_pos = (1, 1)
        self.agent_dir = 0
        available_pos.remove(self.agent_pos)

        # Place a goal square in the bottom-right corner
        goal_pos = (width - 2, height - 2)
        self.no_state_put_obj(grid, Goal(), *goal_pos)
        available_pos.remove(goal_pos)

        # place random lava squares
        if isinstance(self.num_lava_squares, int):
            n_lava_squares = self.num_lava_squares
        else:
            n_lava_squares = self._rand_int(*self.num_lava_squares)

        num_placed_lava_squares = 0
        while num_placed_lava_squares < n_lava_squares:
            # get random square within grid (not including outside walls)
            rand_pos = self.py_random.sample(available_pos, 1)[0]
            self.no_state_put_obj(grid, self.obstacle_type(), *rand_pos)
            available_pos.remove(rand_pos)
            num_placed_lava_squares += 1

        self.grid = grid

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def reset(self, *, seed=None, options=None,size=None, width=None, height=None, num_lava_squares=None):
        super().reset(seed=seed)
        self.seed(seed)
        self._set_env_attributes(size=size, width=width, height=height, num_lava_squares=num_lava_squares)

        self.agent_last_pos = None
        self.agent_pos = None
        self.agent_dir = None
        self.step_count = 0

        valid_grid = False

        if not valid_grid:
            self._gen_grid(self.width, self.height)
            tries = 0
            while not self._path_exists(self.grid):
                self._gen_grid(self.width, self.height)
                tries += 1
                if tries > self.max_rand_tries:
                    raise RuntimeError("Could not generate valid environment in {} tries. Check: \n"
                                       "- if __init__ or reset arguments prevent valid environment generation\n"                                      
                                       "- if increasing self.max_rand_tries helps".format(self.max_rand_tries))

        assert self.agent_pos is not None
        assert self.agent_dir is not None

        start_cell = self.grid.get(*self.agent_pos)
        assert start_cell is None or start_cell.can_overlap()

        obs = self.gen_obs()
        obs.pop('mission')
        obs['direction'] = np.array(obs['direction']).reshape((1,))
        if self.mode == 'rgb':
            obs['image'] = self._get_image_obs()

        if self.window:
            self.window.close()
            self.window = None

        return obs, {}

    def step(self, action: int):
        if action not in self.action_space:
            if not (isinstance(action, np.ndarray) and action[0] in self.action_space):
                raise RuntimeError("Received action outside of action space! action={}".format(action))

        self.agent_last_pos = self.agent_pos
        obs, reward, terminated, truncated, info = MiniGridEnv.step(self, action)

        obs['direction'] = np.array(obs['direction']).reshape((1,))
        obs.pop('mission')

        if self.mode == 'rgb':
            obs['image'] = self._get_image_obs()

        return obs, reward, terminated, truncated, info
