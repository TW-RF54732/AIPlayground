import torch
import torch.optim as optim
from game import GameEnv, ACTION_SPACE
from policy import PolicyNet
from runner import run_episode ,flatten_obs

def train_selfplay(episodes=500):
    env = GameEnv()
    env.reset()
    state = env.get_state()
    sample_obs = flatten_obs(state)
    obs_dim = len(sample_obs)   # <-- 自動偵測
    act_dim = len(ACTION_SPACE)
    
    policy = PolicyNet(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)


    for ep in range(episodes):
        env = GameEnv()
        log_probs, rewards = run_episode(env, policy)

        # 總回報 (最簡 REINFORCE)
        R = sum(rewards)
        loss = -torch.stack(log_probs).sum() * R

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {ep}, Return={R:.2f}")

    return policy


if __name__ == "__main__":
    train_selfplay(episodes=100000)
