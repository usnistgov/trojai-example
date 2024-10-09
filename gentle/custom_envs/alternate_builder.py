from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import gymnasium
import numpy as np

from safety_gymnasium import tasks
from safety_gymnasium.bases.base_task import BaseTask
from safety_gymnasium.utils.common_utils import ResamplingError, quat2zalign
from safety_gymnasium.utils.task_utils import get_task_class_name
from safety_gymnasium.builder import Builder, RenderConf

"""
This is kinda silly but the only way I could get custom envs to build.
This is the same as safety-gymnasium's Builder, but you can optionally 
pass task_class to instantiate a custom environment.
"""


# pylint: disable-next=too-many-instance-attributes
class BuilderCustomTask(Builder):

    metadata: ClassVar[dict[str, Any]] = {
        'render_modes': [
            'human',
            'rgb_array',
            'depth_array',
        ],
        'render_fps': 30,
    }

    def __init__(  # pylint: disable=too-many-arguments
        self,
        task_id: str,
        config: dict | None = None,
        render_mode: str | None = None,
        width: int = 1000,
        height: int = 800,
        camera_id: int | None = None,
        camera_name: str | None = "fixedfar",
        task_class = None,
        **env_kwargs
    ) -> None:
        """Initialize the builder.

        Note:
            The ``camera_name`` parameter can be chosen from:
              - **human**: The camera used for freely moving around and can get input
                from keyboard real time.
              - **vision**: The camera used for vision observation, which is fixed in front of the
                agent's head.
              - **track**: The camera used for tracking the agent.
              - **fixednear**: The camera used for top-down observation.
              - **fixedfar**: The camera used for top-down observation, but is further than **fixednear**.

        Args:
            task_id (str): Task id.
            config (dict): Pre-defined configuration of the environment, which is passed via
              :meth:`safety_gymnasium.register`.
            render_mode (str): Render mode, can be 'human', 'rgb_array', 'depth_array'.
            width (int): Width of the rendered image.
            height (int): Height of the rendered image.
            camera_id (int): Camera id to render.
            camera_name (str): Camera name to render.
        """
        gymnasium.utils.EzPickle.__init__(self, config=config)

        self.env_kwargs = env_kwargs

        self.task_id: str = task_id
        self.task_class = task_class
        self.config: dict = config
        self._seed: int = None
        self._setup_simulation()

        self.first_reset: bool = None
        self.steps: int = None
        self.cost: float = None
        self.terminated: bool = True
        self.truncated: bool = False

        self.render_parameters = RenderConf(render_mode, width, height, camera_id, camera_name)


    def _get_task(self) -> BaseTask:
        """Instantiate a task object."""
        if self.task_class is None:
            class_name = get_task_class_name(self.task_id)
            assert hasattr(tasks, class_name), f'Task={class_name} not implemented.'
            task_class = getattr(tasks, class_name)
        else:
            task_class = self.task_class
            
        task = task_class(config=self.config, **self.env_kwargs)

        task.build_observation_space()
        return task