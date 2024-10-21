from dataclasses import dataclass, field

import numpy as np

from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_object import Mocap
from safety_gymnasium.assets.mocaps.gremlins import Gremlins

from safety_gymnasium.assets.mocaps import MOCAPS_REGISTER



@dataclass
class Mogwais(Gremlins):  # pylint: disable=too-many-instance-attributes
    """Gremlins (moving objects we should avoid)"""

    name: str = 'gremlins'
    num: int = 0  # Number of gremlins in the world
    size: float = 0.1
    placements: list = None  # Gremlins placements list (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.5  # Radius for keeping out (contains gremlin path)
    travel: float = 0.3  # Radius of the circle traveled in
    contact_cost: float = 0.0  # Cost for touching a gremlin
    dist_threshold: float = 10.2  # Threshold for cost for being too close
    dist_cost: float = 10.0  # Cost for being within distance threshold
    density: float = 0.001

    color: np.array = COLOR['gremlin']
    alpha: float = 1
    group: np.array = GROUP['gremlin']
    is_lidar_observed: bool = True
    is_constrained: bool = True
    is_meshed: bool = False
    mesh_name: str = name[:-1]

    target_positions: list = field(default_factory=list) 


    def get_target(self, x, y):
        pass

    def move(self):
        """Set mocap object positions before a physics step is executed."""
        phase = float(self.engine.data.time)
        for i in range(self.num):
            name = f'gremlin{i}'
            target = np.array([np.sin(phase), np.cos(phase)]) * self.travel
            pos = np.r_[target, [self.size]]
            # print(pos)
            pos = np.array([self.target_positions[i][0], self.target_positions[i][1], 0.1])
            self.set_mocap_pos(name + 'mocap', pos)



MOCAPS_REGISTER.append(Mogwais)