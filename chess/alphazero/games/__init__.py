"""Game registry — map a name to a Game implementation."""
from __future__ import annotations

from alphazero.game import Game
from alphazero.games.connect4 import Connect4

_REGISTRY = {
    "connect4": Connect4,
}


def make_game(name: str) -> Game:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown game '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]()
