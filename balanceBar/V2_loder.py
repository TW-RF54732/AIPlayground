import gymnasium as gym
import torch
import torch.nn as nn

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

env = gym.make("CartPole-v1", render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = QNetwork(state_dim, action_dim)
q_net.load_state_dict(torch.load("balanceBar/checkpoints/dqn.pth", map_location=torch.device("cpu")))
q_net.eval()

for ep in range(5):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = q_net(state_tensor)
            action = q_values.argmax().item()

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"🎮 Episode {ep+1}, Total Reward: {total_reward}")

env.close()
