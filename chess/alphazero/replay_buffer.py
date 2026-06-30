"""Fixed-size FIFO buffer of (encoded_state, policy, value) training samples."""
from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: deque = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add_game(self, samples):
        """samples: iterable of (encoded_state, policy_target, value_target)."""
        self.buffer.extend(samples)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.stack(policies).astype(np.float32),
            np.array(values, dtype=np.float32),
        )
