import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 128)
        self.logits = nn.Linear(128, act_dim)

    def forward(self, x, action_mask=None):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.logits(x)

        if action_mask is not None:
            mask = torch.tensor(action_mask, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~mask, -1e9)  # 無效動作屏蔽

        return F.softmax(logits, dim=-1)
