"""Arena: pit two networks against each other to decide whether to promote.

AlphaZero only replaces its "best" network with a freshly trained candidate if
the candidate wins clearly — this keeps self-play data quality monotonic.
"""
from __future__ import annotations

import numpy as np

from alphazero.game import Game
from alphazero.mcts import MCTS


def _play_match(game: Game, mcts_a: MCTS, mcts_b: MCTS):
    """One game. mcts_a is player +1, mcts_b is player -1. Returns result (+1/-1/0)."""
    state = game.initial_state()
    while not game.is_terminal(state):
        mcts = mcts_a if game.current_player(state) == 1 else mcts_b
        policy = mcts.run(state, add_noise=False)  # deterministic-ish for evaluation
        action = int(np.argmax(policy))
        state = game.next_state(state, action)
    return game.result(state)


def evaluate(game: Game, candidate, best, device: str, mcts_cfg, num_games: int) -> float:
    """Return the candidate's score in [0,1] (win=1, draw=0.5), colors alternated."""
    cand_mcts = MCTS(game, candidate, device, mcts_cfg)
    best_mcts = MCTS(game, best, device, mcts_cfg)

    score = 0.0
    for g in range(num_games):
        if g % 2 == 0:  # candidate plays +1
            r = _play_match(game, cand_mcts, best_mcts)
            score += {1: 1.0, 0: 0.5, -1: 0.0}[r]
        else:           # candidate plays -1
            r = _play_match(game, best_mcts, cand_mcts)
            score += {1: 0.0, 0: 0.5, -1: 1.0}[r]
    return score / num_games
