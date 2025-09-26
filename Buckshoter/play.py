import torch
import numpy as np
from game import GameEnv, ACTION_SPACE
from policy import PolicyNet
from runner import flatten_obs

def player_vs_ai(model_path="models/policy.pth"):
    # 初始化環境與模型
    env = GameEnv()
    env.reset()
    state = env.get_state()

    sample_obs = flatten_obs(state)
    obs_dim = len(sample_obs)
    act_dim = len(ACTION_SPACE)

    policy = PolicyNet(obs_dim, act_dim)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    done = False
    while not done:
        turn = state["turn"]
        print("\n=== 遊戲狀態 ===")
        print(f"回合: {state['round']} | 輪到: {'玩家' if turn == 0 else 'AI'}")
        print(f"子彈: live={state['bulletsOnTable']['live']}, blank={state['bulletsOnTable']['blank']}")
        print(f"玩家血量: {state['players'][0]['health']} | AI血量: {state['players'][1]['health']}")
        print(f"玩家道具: {state['players'][0]['items']}")
        print(f"AI道具: {state['players'][1]['items']}")
        print(f"可用動作: {state['action_mask']}")

        if turn == 0:
            # 玩家回合
            legal_actions = [i for i, m in enumerate(state["action_mask"]) if m == 1]
            print("動作列表:")
            for i in legal_actions:
                print(f" {i}: {ACTION_SPACE[i]}")

            while True:
                try:
                    action = int(input("請選擇動作: "))
                    if action in legal_actions:
                        break
                    else:
                        print("⚠️ 不合法動作，請重新輸入")
                except ValueError:
                    print("⚠️ 請輸入整數")
        else:
            # AI 回合
            obs_vec = torch.tensor(flatten_obs(state), dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor(state["action_mask"], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                probs = policy(obs_vec, mask).numpy()[0]
            action = np.random.choice(len(probs), p=probs)
            print(f"AI 選擇: {ACTION_SPACE[action]}")

        # 執行動作
        state, reward, done, info = env.step(action)
        print(f"動作結果 → Reward={reward}")

    # 遊戲結束
    print("\n=== 遊戲結束 ===")
    if state["players"][0]["health"] <= 0:
        print("玩家死亡 ❌，AI 勝利 🤖")
    elif state["players"][1]["health"] <= 0:
        print("AI 死亡 ❌，玩家勝利 🎉")
    else:
        print("遊戲結束條件未知:", state)


if __name__ == "__main__":
    player_vs_ai("models/policy.pth")
