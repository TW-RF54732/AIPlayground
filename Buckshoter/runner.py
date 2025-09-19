import torch
import numpy as np
from torch.distributions import Categorical


def flatten_obs(obs: dict):
    """把 get_state() 的 dict 轉成向量"""
    vec = []
    vec.append(obs["turn"])
    vec.append(obs["round"])
    vec.append(obs["maxHealth"])
    vec.append(obs["itemPerRound"])
    vec.append(obs["bulletsOnTable"]["live"])
    vec.append(obs["bulletsOnTable"]["blank"])
    for pid in [0, 1]:
        vec.append(obs["players"][pid]["health"])
        vec.append(len(obs["players"][pid]["items"]))
        vec.append(int(obs["players"][pid]["skip"]))
    vec.append(int(obs["shorted"]))
    vec.append(0 if obs["magnifier_result"] is None else int(obs["magnifier_result"]))
    return np.array(vec, dtype=np.float32)


def run_episode(env, policy):
    """跑一局遊戲，回傳 log_probs 與 rewards"""
    log_probs, rewards = [], []
    done = False
    env.reset()

    while not done:
        obs = env.get_state()
        obs_vec = torch.tensor(flatten_obs(obs), dtype=torch.float32).unsqueeze(0)

        action_mask = obs["action_mask"]
        probs = policy(obs_vec, action_mask)
        dist = Categorical(probs)
        action = dist.sample()

        log_probs.append(dist.log_prob(action))

        _, reward, done, _ = env.step(action.item())
        rewards.append(reward)

    return log_probs, rewards
