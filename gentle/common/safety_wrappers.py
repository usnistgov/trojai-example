import gymnasium
import safety_gymnasium

def make_safe_env(env_id):
    safe_env = safety_gymnasium.make(env_id)
    env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(safe_env)
    env = gymnasium.wrappers.SomeWrapper1(env)
    env = gymnasium.wrappers.SomeWrapper2(env, argname1=arg1, argname2=arg2)
    ...
    env = gymnasium.wrappers.SomeWrapperN(env)
    safe_env = safety_gymnasium.wrappers.Gymnasium2SafetyGymnasium(env)
    return safe_env



import functools

import gymnasium
import safety_gymnasium

def make_safe_env(env_id):
    return safety_gymnasium.wrappers.with_gymnasium_wrappers(
        safety_gymnasium.make(env_id),
        gymnasium.wrappers.SomeWrapper1,
        functools.partial(gymnasium.wrappers.SomeWrapper2, argname1=arg1, argname2=arg2),
        ...,
        gymnasium.wrappers.SomeWrapperN,
    )


import gymnasium
import safety_gymnasium

safety_gymnasium.make('SafetyPointGoal1-v0')    # step returns (next_obervation, reward, cost, terminated, truncated, info)
gymnasium.make('SafetyPointGoal1Gymnasium-v0')  # step returns (next_obervation, reward, terminated, truncated, info)