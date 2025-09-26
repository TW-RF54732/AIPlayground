# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from runner import run_episode
from policy import PolicyNet
from game import GameEnv
import matplotlib.pyplot as plt

def train_selfplay(episodes=500, lr=1e-3, save_path="policy.pth"):
    env = GameEnv()
    env.reset()
    obs_size = len(env.get_state()["action_mask"]) + 13  # 假設 flatten_obs 固定長度13
    action_size = len(env.get_state()["action_mask"])
    policy = PolicyNet(obs_size, action_size)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    rewards_history = []

    for ep in range(episodes):
        log_probs, rewards = run_episode(env, policy)
        total_reward = sum(rewards)
        rewards_history.append(total_reward)

        # Policy gradient
        loss = []
        G = 0
        for log_prob, reward in zip(reversed(log_probs), reversed(rewards)):
            G = reward + 0.99 * G
            loss.insert(0, -log_prob * G)
        loss = torch.stack(loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep+1) % 50 == 0:
            avg_reward = sum(rewards_history[-50:]) / 50
            print(f"Episode {ep+1}/{episodes}, AvgReward={avg_reward:.2f}")

    # 存模型
    torch.save(policy.state_dict(), save_path)

    # 畫 Reward 曲線
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Progress")
    plt.savefig("training_curve.png")
    plt.close()

if __name__ == "__main__":
    train_selfplay(episodes=500000)
