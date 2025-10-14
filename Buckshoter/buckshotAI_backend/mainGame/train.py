import torch
import torch.optim as optim
from mainGame.game import GameEnv, ACTION_SPACE
from mainGame.policy import PolicyNet
from mainGame.runner import run_episode, flatten_obs
import os

def train_selfplay(episodes=500, save_path="models/policy.pth"):
    # 自動偵測 GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用裝置:", device)

    env = GameEnv()
    env.reset()
    state = env.get_state()
    sample_obs = flatten_obs(state)
    obs_dim = len(sample_obs)   # <-- 自動偵測
    act_dim = len(ACTION_SPACE)
    
    policy = PolicyNet(obs_dim, act_dim).to(device)   # 模型搬到 GPU/CPU
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    try:
        if episodes != -1:
            for ep in range(episodes):
                env = GameEnv()
                log_probs, rewards = run_episode(env, policy, device=device)  # 把 device 傳下去

                # 總回報 (最簡 REINFORCE)
                R = sum(rewards)
                if log_probs:  # 避免遊戲瞬間結束 log_probs 為空
                    loss = -torch.stack(log_probs).sum() * R
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print(f"Episode {ep}, Return={R:.2f}, Round={env.round}")

        elif episodes == -1:
            ep = 0
            while True:
                ep += 1
                env = GameEnv()
                log_probs, rewards = run_episode(env, policy, device=device)  # 把 device 傳下去

                # 總回報 (最簡 REINFORCE)
                R = sum(rewards)
                if log_probs:  # 避免遊戲瞬間結束 log_probs 為空
                    loss = -torch.stack(log_probs).sum() * R
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                print(f"Episode {ep}, Return={R:.2f}, Round={env.round}")
    except:
        torch.save(policy.state_dict(), save_path)
        print(f"模型已保存到 {os.path.abspath(save_path)}")

    # 儲存模型（僅儲存參數，不管 GPU/CPU）
    # torch.save(policy.state_dict(), save_path)
    # print(f"模型已保存到 {os.path.abspath(save_path)}")

    return policy


if __name__ == "__main__":
    train_selfplay(episodes=-1, save_path="models/policy.pth")
