import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import json
import pickle
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import time

# ==== 確認是否使用 GPU ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ==== 超參數 ====
EPISODES = 3000
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEM_SIZE = 10000
TARGET_UPDATE = 10
SAVE_FREQ = 50
CHECKPOINT_DIR = "balanceBar/checkpoints"

# ==== 建立 Q 網路 ====
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==== 經驗重播 ====
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = map(np.array, zip(*samples))
        return s, a, r, s_, d

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.buffer = pickle.load(f)

def get_epsilon(episode, max_episodes=EPISODES, eps_start=1.0, eps_end=0.01):
    return max(eps_end, eps_start - (eps_start - eps_end) * episode / max_episodes)

def save_checkpoint(q_net, optimizer, replay_buffer, episode):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(q_net.state_dict(), f"{CHECKPOINT_DIR}/dqn.pth")
    torch.save(optimizer.state_dict(), f"{CHECKPOINT_DIR}/optimizer.pth")
    replay_buffer.save(f"{CHECKPOINT_DIR}/replay_buffer.pkl")
    with open(f"{CHECKPOINT_DIR}/meta.json", "w") as f:
        json.dump({"episode": episode}, f)
    print(f"💾 Checkpoint saved at episode {episode}")

def load_checkpoint(q_net, optimizer, replay_buffer):
    q_net.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/dqn.pth"))
    optimizer.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/optimizer.pth"))
    replay_buffer.load(f"{CHECKPOINT_DIR}/replay_buffer.pkl")
    with open(f"{CHECKPOINT_DIR}/meta.json", "r") as f:
        meta = json.load(f)
    print(f"✅ Loaded checkpoint from episode {meta['episode']}")
    return meta["episode"]

# ==== 初始化 ====
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = QNetwork(state_dim, action_dim).to(device)
target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=LR)
replay_buffer = ReplayBuffer(MEM_SIZE)
writer = SummaryWriter(log_dir="runs/dqn_cartpole")
rewards_history = []

# 即時畫圖
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], label="Reward")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.set_title("Training Progress")
ax.grid(True)

start_episode = 0
if os.path.exists(f"{CHECKPOINT_DIR}/meta.json"):
    start_episode = load_checkpoint(q_net, optimizer, replay_buffer)
else:
    print("🚀 Starting training from scratch.")

start_time = time.time()

try:
    for episode in trange(start_episode, EPISODES, desc="Training", dynamic_ncols=True):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            epsilon = get_epsilon(episode)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = q_net(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # 訓練
            if len(replay_buffer) >= BATCH_SIZE:
                s, a, r, s_, d = replay_buffer.sample(BATCH_SIZE)
                s = torch.FloatTensor(s).to(device)
                a = torch.LongTensor(a).to(device)
                r = torch.FloatTensor(r).to(device)
                s_ = torch.FloatTensor(s_).to(device)
                d = torch.FloatTensor(d).to(device)

                q_values = q_net(s)
                next_q = target_net(s_).max(1)[0]
                expected_q = r + GAMMA * next_q * (1 - d)
                q_value = q_values.gather(1, a.unsqueeze(1)).squeeze()

                loss = nn.MSELoss()(q_value, expected_q.detach())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rewards_history.append(total_reward)
        writer.add_scalar("Reward", total_reward, episode)
        writer.add_scalar("Epsilon", epsilon, episode)

        # 更新 target_net
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

        # 更新圖表
        if episode % 5 == 0:
            line.set_xdata(range(len(rewards_history)))
            line.set_ydata(rewards_history)
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.001)

        if (episode + 1) % SAVE_FREQ == 0:
            save_checkpoint(q_net, optimizer, replay_buffer, episode + 1)

except KeyboardInterrupt:
    print("\n🛑 手動中斷，儲存 checkpoint...")
    save_checkpoint(q_net, optimizer, replay_buffer, episode)

finally:
    print("🏁 訓練完成。")
    total_time = time.time() - start_time
    print(f"⏱️ Total training time: {total_time:.2f} seconds")

    plt.ioff()
    plt.plot(rewards_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Final Training Progress")
    plt.grid(True)
    plt.show()
    writer.close()
    env.close()
