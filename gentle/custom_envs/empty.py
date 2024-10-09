from safety_gymnasium.bases.base_task import BaseTask

class EmptySafetyGymnasium(BaseTask):

    def __init__(self, config) -> None:
        super().__init__(config=config)
        self.placements_conf.extents = [-1, -1, 1, 1]

    def calculate_reward(self):
        return 0.0

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        pass

    def goal_achieved(self):
        return False