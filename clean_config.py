from box import Box

def clean_config():
    config = Box(
        {"environment" : {
         "type": "safety_gymnasium",
         "name": "SafetyCarGoal2-v0",
         "mod_config": {
           "scale": 0.1,
           "cost": "one_indicator"
         },
         "env_kwargs":{}
       }},
    )
    config.environment.env_kwargs.num_mogwais = 4  # [2,12] inclusive
    config.environment.hazards_are_terminal = False
    config.environment.velocity_reward = 0.0
    config.environment.antigoal_multiplier = 0.5
    return config