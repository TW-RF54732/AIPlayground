import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from generals.agents import ExpanderAgent
from generals.envs import GymnasiumGenerals  # gym 介面

# 1️⃣ 初始化環境與敵人
enemy = ExpanderAgent()
env = gym.make("gym-generals-v0", agent=enemy, npc=None, render_mode=None)
# env 内部会处理你的 agent 和预设 enemy

# Reset environment
obs, info = env.reset()
state = obs

# 2️⃣ 网络结构定义
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

state_dim = obs.shape[0]
action_dim = env.action_space.n
model = DQN(state_dim, action_dim)
target = DQN(state_dim, action_dim)
target.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=1e-3)
memory = deque(maxlen=10000)
batch_size = 64
gamma = 0.99
epsilon = 1.0
eps_min = 0.05
eps_decay = 0.995
sync_every = 100

# 3️⃣ 开始训练
step = 0
for ep in range(500):
    obs, info = env.reset()
    state = obs
    total_r = 0
    done = False

    while not done:
        # ε-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q = model(torch.FloatTensor(state).unsqueeze(0))
                action = q.argmax().item()

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        memory.append((state, action, reward, next_obs, done))
        state = next_obs
        total_r += reward
        step += 1

        # 每步训练
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            s, a, r, s2, d = map(np.array, zip(*batch))
            s = torch.FloatTensor(s)
            a = torch.LongTensor(a)
            r = torch.FloatTensor(r)
            s2 = torch.FloatTensor(s2)
            d = torch.FloatTensor(d)

            qvals = model(s).gather(1, a.unsqueeze(1)).squeeze()
            next_q = target(s2).max(1)[0]
            target_q = r + gamma * next_q * (1 - d)

            loss = nn.MSELoss()(qvals, target_q.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 更新 epsilon 和 target net
        epsilon = max(eps_min, epsilon * eps_decay)
        if step % sync_every == 0:
            target.load_state_dict(model.state_dict())

    print(f"Episode {ep}, Reward: {total_r:.2f}, Epsilon: {epsilon:.3f}")

env.close()
