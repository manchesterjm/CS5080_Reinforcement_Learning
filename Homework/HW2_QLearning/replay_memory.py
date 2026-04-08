"""
Experience Replay Buffer for DQN

Stores transitions (s, a, r, s', done) and samples mini-batches
for training. Uses a circular buffer (deque) for memory efficiency.

Reference: Mnih et al. (2015), "Human-level control through deep
reinforcement learning", Nature.
"""

import random
from collections import deque, namedtuple
from typing import List, Optional, Tuple

import numpy as np

Transition = namedtuple("Transition",
                        ["state", "action", "reward",
                         "next_state", "done"])


class ReplayBuffer:
    """Fixed-size circular buffer for experience replay."""

    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        if seed is not None:
            random.seed(seed)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """Store a transition."""
        self.buffer.append(Transition(state, action, reward,
                                      next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a random mini-batch of transitions."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch."""
        return len(self.buffer) >= batch_size
