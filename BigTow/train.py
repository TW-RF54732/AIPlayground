import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from BigTwoEnv import BigTwoEnv  # 假設你的環境是這個檔名
from model import PolicyNet         # 你的神經網路模型
from utils import obs_to_tensor, select_action, compute_returns  # 輔助函數

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔧 參數
checkpoint_path = "checkpoint.pt"
checkpoint_interval = 500  # 每幾回合存一次
total_episodes = 10000
num_players = 4
learning_rate = 1e-3

# 🧠 初始化模型與 optimizer
policy = PolicyNet().to(device)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

# 🔁 載入 checkpoint（若存在）
start_episode = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    policy.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_episode = checkpoint["episode"] + 1
    print(f"✅ 載入 checkpoint，從第 {start_episode} 回合繼續訓練")

# 🏋️ 訓練主迴圈
for episode in range(start_episode, total_episodes):
    env = BigTwoEnv(num_players)
    obs = env.reset()

    # 每位玩家的 log_probs 與 rewards
    memories = [{"log_probs": [], "rewards": []} for _ in range(num_players)]

    while not env.done:
        player = env.current_player
        obs_tensor = obs_to_tensor(obs).to(device)
        valid_actions = env.compute_valid_actions()

        action, log_prob = select_action(policy, obs_tensor, valid_actions)
        obs, _, done, _ = env.step(action)

        # 僅記錄有出牌的玩家
        memories[player]["log_probs"].append(log_prob)
        memories[player]["rewards"].append(0)

    # 🎯 結算勝利者（拿 reward = +1）
    for i in range(num_players):
        if len(env.hands[i]) == 0:
            memories[i]["rewards"][-1] = 1

    # 🧮 計算 loss 並反向傳播
    loss = 0
    for m in memories:
        if len(m["log_probs"]) == 0:
            continue
        returns = compute_returns(m["rewards"])
        returns = torch.tensor(returns, device=device)
        log_probs = torch.stack(m["log_probs"])
        loss -= (log_probs * returns).sum()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 💾 儲存 checkpoint
    if episode % checkpoint_interval == 0 or episode == total_episodes - 1:
        torch.save({
            "episode": episode,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, checkpoint_path)
        print(f"💾 已儲存 checkpoint @ episode {episode}")

print("🏁 訓練完成！")
