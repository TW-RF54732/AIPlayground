import torch
import numpy as np
from model import PolicyNet  # 或從你自己的模型檔案 import
from BigTwoEnv import BigTwoEnv, int_to_card  # 假設你的環境檔案是 big_two_env.py
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入模型
policy = PolicyNet().to(device)
policy.load_state_dict(torch.load("policy.pt"))
policy.eval()

env = BigTwoEnv(num_players=4)
obs = env.reset()

human_player = 0

def decode_action(action_idx):
    return "pass" if action_idx == 0 else int_to_card(action_idx - 1)

def select_action(model, obs):
    obs_tensor = torch.FloatTensor(obs['hand']).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(obs_tensor).cpu().squeeze().numpy()

    valid_actions = env.compute_valid_actions()
    mask = np.zeros_like(probs)
    mask[valid_actions] = 1
    probs = probs * mask
    if probs.sum() == 0:
        return 0  # only pass is valid
    probs /= probs.sum()
    action = np.random.choice(len(probs), p=probs)
    return action

while not env.done:
    env.render()
    obs = env.get_obs()
    player = obs['player_id']

    if player == human_player:
        valid_actions = env.compute_valid_actions()
        print(f"\n🎮 輪到你出牌 (玩家 {player})")
        print("你手牌：", [int_to_card(c) for c in env.hands[player]])
        print("有效行動：")
        for a in valid_actions:
            print(f"{a}: {decode_action(a)}")

        while True:
            try:
                action = int(input("請輸入你要出哪一張（index）："))
                if action in valid_actions:
                    break
                else:
                    print("❌ 無效行動，請重新輸入")
            except:
                print("⚠️ 請輸入數字")
    else:
        action = select_action(policy, obs)
        print(f"🤖 玩家 {player} 出：{decode_action(action)}")

    obs, _, done, _ = env.step(action)

env.render()
print("\n✅ 遊戲結束！")
