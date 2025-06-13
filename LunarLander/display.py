import gym
import torch
import torch.nn as nn

# ----- Environment and Model Setup -----
ENV_NAME = "LunarLander-v2"
CHECKPOINT_PATH = "checkpoints_lunar/dqn.pth"

env = gym.make(ENV_NAME, render_mode="human")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
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

# ----- Load Model -----
model = DQN(state_dim, action_dim)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu")))
model.eval()

# ----- Play Environment -----
try:
    while True:
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = model(state_tensor).argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"🎮 Episode Reward: {total_reward:.2f}")
except KeyboardInterrupt:
    print("👋 Exit display mode.")
finally:
    env.close()