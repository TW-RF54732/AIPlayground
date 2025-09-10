import torch
import torch.nn.functional as F
import numpy as np

def obs_to_tensor(obs):
    vec = np.concatenate([
        obs["hand"],
        obs["current_combo"],
        np.array(obs["hand_counts"]),
        [obs["pass_count"]]
    ])
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

def select_action(policy, obs_tensor, valid_actions):
    logits = policy(obs_tensor)
    mask = torch.zeros_like(logits)
    mask[0, valid_actions] = 1

    probs = F.softmax(logits.masked_fill(mask == 0, -1e9), dim=1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns
