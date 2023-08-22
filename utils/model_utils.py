import torch

def get_action_from_logits(logits, sample=True):
    if sample:
        return torch.distributions.Categorical(logits=logits).sample().item()
    else:
        return torch.argmax(logits).item()


def compute_action_from_trojai_rl_model(model, observation, sample=True):
    dist = model(observation)
    action = get_action_from_logits(dist, sample=sample)
    return action