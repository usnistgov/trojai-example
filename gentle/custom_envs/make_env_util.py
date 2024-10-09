import safety_gymnasium
from gentle.custom_envs.alternate_builder import BuilderCustomTask
from gentle.common.objects_info_wrapper import SphericalObjectsInfoWrapper


# wrapper that rebuilds the environment on each reset
# this lets us have randomness in configurations that are definted with locations in __init__
class Rebuilder():

    def __init__(self, build_env_func, **env_kwargs):
        self.build_env_func = build_env_func
        self.reference_env = build_env_func(**env_kwargs) # for defining spaces and such
        self.action_space = self.reference_env.action_space
        self.observation_space = self.reference_env.observation_space
        self.env = build_env_func(**env_kwargs)
        self.unwrapped = self.reference_env
        self.env_kwargs = env_kwargs

    def reset(self, *, seed = None, options = None,):
        self.env = self.build_env_func(**self.env_kwargs)
        self.unwrapped = self.env
        return self.env.reset()

    def step(self, a):
        return self.env.step(a)

    def render(self):
        return self.env.render()

# util to create a custom environment
def make_custom_arranged_env(
    env_class,
    agent_type,
    render_mode,
    rebuild=True,
    **env_kwargs
):

    def build_env(**env_kwargs):
        env = BuilderCustomTask(
            config={"agent_name":agent_type}, 
            task_id="Safety"+agent_type+"Custom-v0", 
            render_mode=render_mode, # this will make a new viewer each reset(), use None to turn off
            task_class=env_class,
            **env_kwargs
        )
        return safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)

    if not rebuild:
        return SphericalObjectsInfoWrapper(build_env(**env_kwargs))

    env = Rebuilder(build_env, **env_kwargs)
    return SphericalObjectsInfoWrapper(env)