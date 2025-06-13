import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import json
import pickle
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ===== 超參數 =====
ENV_NAME = "LunarLander-v3"
EPISODES = 3000
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEM_SIZE = 10000
TARGET_UPDATE = 20
SAVE_FREQ = 50
CHECKPOINT_DIR = "checkpoints_lunar"

# ===== 建立 Q 網路 =====
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

# ===== 經驗回放 =====
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

# ===== 工具函數 =====
def get_epsilon(episode, eps_start=1.0, eps_end=0.01):
    return max(eps_end, eps_start - (eps_start - eps_end) * episode / EPISODES)

def save_checkpoint(q_net, optimizer, buffer, episode):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    torch.save(q_net.state_dict(), f"{CHECKPOINT_DIR}/dqn.pth")
    torch.save(optimizer.state_dict(), f"{CHECKPOINT_DIR}/optimizer.pth")
    buffer.save(f"{CHECKPOINT_DIR}/buffer.pkl")
    with open(f"{CHECKPOINT_DIR}/meta.json", "w") as f:
        json.dump({"episode": episode}, f)
    print(f"💾 Checkpoint saved at episode {episode}")

def load_checkpoint(q_net, optimizer, buffer):
    q_net.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/dqn.pth", map_location=device))
    optimizer.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/optimizer.pth", map_location=device))
    buffer.load(f"{CHECKPOINT_DIR}/buffer.pkl")
    with open(f"{CHECKPOINT_DIR}/meta.json") as f:
        meta = json.load(f)
    print(f"✅ Loaded checkpoint at episode {meta['episode']}")
    return meta["episode"]

# ===== 初始化 =====
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = QNetwork(state_dim, action_dim).to(device)
target_net = QNetwork(state_dim, action_dim).to(device)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=LR)
buffer = ReplayBuffer(MEM_SIZE)
writer = SummaryWriter(log_dir="runs/lunar")

start_episode = 0
if os.path.exists(f"{CHECKPOINT_DIR}/meta.json"):
    start_episode = load_checkpoint(q_net, optimizer, buffer)
else:
    print("🚀 Starting from scratch.")

# ===== 訓練主迴圈 =====
try:
    for episode in range(start_episode, EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        start_time = time.time()

        while not done:
            epsilon = get_epsilon(episode)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = q_net(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.push(state, action, reward, next_state, done)
            total_reward += reward
            state = next_state

            # 訓練
            if len(buffer) >= BATCH_SIZE:
                s, a, r, s_, d = buffer.sample(BATCH_SIZE)
                s = torch.FloatTensor(s).to(device)
                a = torch.LongTensor(a).unsqueeze(1).to(device)
                r = torch.FloatTensor(r).to(device)
                s_ = torch.FloatTensor(s_).to(device)
                d = torch.FloatTensor(d).to(device)

                q_vals = q_net(s).gather(1, a).squeeze()
                next_q = target_net(s_).max(1)[0]
                target = r + GAMMA * next_q * (1 - d)

                loss = nn.MSELoss()(q_vals, target.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 日誌
        writer.add_scalar("Reward", total_reward, episode)
        writer.add_scalar("Epsilon", epsilon, episode)
        writer.add_scalar("Episode Duration", time.time() - start_time, episode)
        if 'loss' in locals():
            writer.add_scalar("Loss", loss.item(), episode)

        # 更新 target
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

        if (episode + 1) % 10 == 0:
            print(f"Ep {episode+1}, Reward: {total_reward:.2f}")

        if (episode + 1) % SAVE_FREQ == 0:
            save_checkpoint(q_net, optimizer, buffer, episode + 1)

except KeyboardInterrupt:
    print("🛑 中斷訓練，保存模型...")
    save_checkpoint(q_net, optimizer, buffer, episode)

finally:
    print("🏁 訓練完成！")
    env.close()
    writer.close()
