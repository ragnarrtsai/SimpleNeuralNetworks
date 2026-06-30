"""ResNet with policy + value heads — the AlphaZero network.

Game-agnostic: input/output shapes come from the `Game` instance, so the same
class trains Connect 4, Othello, or Gomoku.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alphazero.game import Game


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class AlphaZeroNet(nn.Module):
    def __init__(self, game: Game, channels: int = 64, num_res_blocks: int = 5):
        super().__init__()
        in_planes, h, w = game.encoded_shape
        action_size = game.action_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])

        # Policy head: 1x1 conv -> flatten -> logits over actions.
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * h * w, action_size),
        )
        # Value head: 1x1 conv -> hidden -> scalar in [-1, 1].
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(h * w, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.tower(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return policy_logits, value

    @torch.no_grad()
    def predict(self, encoded: np.ndarray, legal_mask: np.ndarray, device: str):
        """Single-state inference. Returns (policy probs over legal actions, value).

        Illegal actions are masked to zero probability before the softmax.
        """
        self.eval()
        x = torch.from_numpy(encoded).unsqueeze(0).to(device)
        logits, value = self.forward(x)
        logits = logits.squeeze(0).cpu().numpy()

        logits[~legal_mask] = -1e9
        logits -= logits.max()
        probs = np.exp(logits)
        probs[~legal_mask] = 0.0
        total = probs.sum()
        if total > 0:
            probs /= total
        else:  # degenerate fallback: uniform over legal moves
            probs[legal_mask] = 1.0 / legal_mask.sum()
        return probs, float(value.item())
