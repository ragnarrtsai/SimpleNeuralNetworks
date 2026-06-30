import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int = 784, hidden=(256, 128), num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Linear(hidden[1], num_classes),
        )

    def forward(self, x):
        return self.net(x)
