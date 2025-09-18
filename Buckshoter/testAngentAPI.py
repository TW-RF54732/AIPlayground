import numpy as np
from game import GameEnv, allowItems, ACTION_SPACE

def encode_items(items, allowItems):
    """把道具清單轉成固定長度的 multi-hot 計數向量"""
    vec = [0] * len(allowItems)
    for item in items:
        if item in allowItems:
            vec[allowItems.index(item)] += 1
    return vec

def obs_to_vec(obs):
    """
    把環境輸出的 obs(dict) 轉成固定長度 numpy 向量
    """
    vec = []
    # 基本數值
    vec.append(obs["turn"])
    vec.append(obs["round"])
    vec.append(obs["bulletsOnTable"]["live"])
    vec.append(obs["bulletsOnTable"]["blank"])
    vec.append(int(obs["shorted"]))
    vec.append(1 if obs["magnifier_result"] else 0)

    # 玩家資訊
    for pid in [0, 1]:
        vec.append(obs["players"][pid]["health"])
        vec.append(int(obs["players"][pid]["skip"]))
        vec.extend(encode_items(obs["players"][pid]["items"], allowItems))

    return np.array(vec, dtype=np.float32)


# 測試 loop
if __name__ == "__main__":
    env = GameEnv()
    env.reset()
    obs = env.get_state()

    for step in range(20):  # 最多跑 20 步
        vec = obs_to_vec(obs)
        action_mask = obs["action_mask"]

        # 從合法動作中隨機挑一個
        valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
        action = np.random.choice(valid_actions)

        obs, reward, done, info = env.step(action)

        print(f"Step {step}, Action={ACTION_SPACE[action]}, Reward={reward}, player={env.turn}")
        print("Obs vector shape:", vec.shape)

        if done:
            print("Game Over:", info)
            break
