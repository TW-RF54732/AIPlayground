import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        input_size = 52 * 2 + 4 + 1  # hand + current_combo + hand_counts + pass_count
        hidden_size = 256
        output_size = 53  # 52 張牌 + 1 個 pass 動作

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # 沒 softmax，因為訓練時用 Categorical
