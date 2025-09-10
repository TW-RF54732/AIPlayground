import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from generals.agents import RandomAgent, ExpanderAgent
from generals.envs import PettingZooGenerals

# ⚠️ 自定合法動作產生器
def compute_valid_action_mask(obs):
    N, M = obs['armies'].shape
    mask = np.zeros((N, M, 4), dtype=bool)
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(N):
        for j in range(M):
            if obs['owned_cells'][i, j] == 0:
                continue
            for d, (di, dj) in enumerate(moves):
                ni, nj = i + di, j + dj
                if 0 <= ni < N and 0 <= nj < M and obs['mountains'][ni, nj] == 0:
                    mask[i, j, d] = True
    return mask

# —— 初始化環境 —— #
player = ExpanderAgent()
opponent = RandomAgent()
agents = {
    player.id: player,
    opponent.id: opponent
}
env = PettingZooGenerals(agents=agents, render_mode=None)

observations, info = env.reset()
sample = observations[player.id]
N, M = sample['armies'].shape
map_keys = [k for k, v in sample.items() if isinstance(v, np.ndarray) and v.ndim == 2]
scalar_keys = [k for k, v in sample.items() if np.isscalar(v)]

# —— 模型定義 —— #
class CNNDQN(nn.Module):
    def __init__(self, in_ch, h, w, num_scalars, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU()
        )
        flat = 32 * h * w
        self.head = nn.Sequential(
            nn.Linear(flat + num_scalars, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, maps, scalars):
        c = self.conv(maps).view(maps.size(0), -1)
        return self.head(torch.cat([c, scalars], dim=1))

in_ch = len(map_keys)
num_scalars = len(scalar_keys)
action_dim = N * M * 4

model = CNNDQN(in_ch, N, M, num_scalars, action_dim)
target = CNNDQN(in_ch, N, M, num_scalars, action_dim)
target.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=1e-4)
memory = deque(maxlen=50000)
batch_size, gamma = 64, 0.99
epsilon, eps_min, eps_decay = 1.0, 0.05, 0.995
sync_steps = 1000
step = 0
EPISODES = 300

def preprocess(obs):
    maps = torch.tensor([obs[k] for k in map_keys], dtype=torch.float32).unsqueeze(0)
    scalars = torch.tensor([obs[k] for k in scalar_keys], dtype=torch.float32).unsqueeze(0)
    return maps, scalars

def sample_valid(o):
    mask = compute_valid_action_mask(o)
    valid = [(i, j, d) for i in range(N) for j in range(M) for d in range(4) if mask[i, j, d]]
    return valid

# —— 訓練迴圈 —— #
for ep in range(EPISODES):
    obs, info = env.reset()
    done = False
    ep_rewards = {aid: 0.0 for aid in env.agents}
    while not done:
        actions = {}
        for aid in env.agents:
            maps, scalars = preprocess(obs[aid])
            valid = sample_valid(obs[aid])
            if not valid or random.random() < epsilon:
                actions[aid] = [1,0,0,0,0]
            else:
                q = model(maps, scalars).view(-1)
                idxs = [i*M*4 + j*4 + d for (i, j, d) in valid]
                best = idxs[q[idxs].argmax().item()]
                i, rem = divmod(best, M*4)
                j, d = divmod(rem, 4)
                actions[aid] = [0, i, j, d, 0]
        next_obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        for aid in env.agents:
            memory.append((obs[aid], actions[aid], rewards[aid], next_obs[aid], done))
            ep_rewards[aid] += rewards[aid]
        obs = next_obs

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            maps_b = torch.stack([torch.tensor([b[0][k] for k in map_keys], dtype=torch.float32) for b in batch])
            scalars_b = torch.stack([torch.tensor([b[0][k] for k in scalar_keys], dtype=torch.float32) for b in batch])
            act_list = [b[1] for b in batch]
            idxs = [a[1]*M*4 + a[2]*4 + a[3] for a in act_list]
            a_b = torch.LongTensor(idxs)
            r_b = torch.FloatTensor([b[2] for b in batch])
            next_maps = torch.stack([torch.tensor([b[3][k] for k in map_keys], dtype=torch.float32) for b in batch])
            next_scalars = torch.stack([torch.tensor([b[3][k] for k in scalar_keys], dtype=torch.float32) for b in batch])
            d_b = torch.FloatTensor([float(b[4]) for b in batch])

            q_vals = model(maps_b, scalars_b).gather(1, a_b.unsqueeze(1)).squeeze()
            with torch.no_grad():
                nxt = target(next_maps, next_scalars).max(1)[0]
                tgt = r_b + gamma * nxt * (1 - d_b)
            loss = nn.MSELoss()(q_vals, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % sync_steps == 0:
                target.load_state_dict(model.state_dict())

        epsilon = max(eps_min, epsilon * eps_decay)

    print(f"Ep {ep+1}/{EPISODES}, Rewards: {ep_rewards}")
print("訓練完成 🎉")
env.close()
