"""Abstract game interface for the AlphaZero pipeline.

Everything downstream (network, MCTS, self-play, training) is written against
this interface, so adding Othello or Gomoku later just means implementing a new
subclass — no changes to the core algorithm.

Conventions
-----------
* Players are +1 and -1. +1 always moves first from the initial state.
* A "state" is whatever the implementation wants (here: a small numpy array plus
  the player to move). It must be hashable-able via `string_key` for MCTS.
* `encode` returns the network input *from the perspective of the player to
  move* (the "canonical" form), so the network only ever sees "my turn".
* `result` is the game outcome from the perspective of player +1:
  +1 win, -1 loss, 0 draw. MCTS/self-play flip signs per node as needed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Game(ABC):
    name: str
    rows: int
    cols: int

    @property
    @abstractmethod
    def action_size(self) -> int:
        """Number of distinct actions (size of the policy head)."""

    @property
    @abstractmethod
    def encoded_shape(self) -> tuple[int, int, int]:
        """(channels, height, width) of the network input."""

    @abstractmethod
    def initial_state(self): ...

    @abstractmethod
    def current_player(self, state) -> int:
        """+1 or -1: whose turn it is."""

    @abstractmethod
    def legal_actions(self, state) -> np.ndarray:
        """Boolean mask of shape (action_size,)."""

    @abstractmethod
    def next_state(self, state, action: int):
        """Return the state after `action` is played (does not mutate `state`)."""

    @abstractmethod
    def is_terminal(self, state) -> bool: ...

    @abstractmethod
    def result(self, state) -> float:
        """Outcome from player +1's perspective: +1 / -1 / 0. Call only if terminal."""

    @abstractmethod
    def encode(self, state) -> np.ndarray:
        """Network input of shape `encoded_shape`, canonical to the player to move."""

    @abstractmethod
    def string_key(self, state) -> str:
        """A unique, hashable key for the state (used as the MCTS node id)."""

    @abstractmethod
    def render(self, state) -> str:
        """Human-readable board for the CLI."""

    def symmetries(self, encoded: np.ndarray, policy: np.ndarray):
        """Optional data-augmentation: yield (encoded, policy) equivalents.

        Default: identity only. Games with board symmetry (Connect 4 has a
        left-right mirror) override this to multiply training data for free.
        """
        return [(encoded, policy)]
