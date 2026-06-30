"""Self-play: the agent plays itself with MCTS to generate training data.

Each move produces a training sample:
    (encoded_state, mcts_policy, z)
where `mcts_policy` is the visit-count distribution (the search's improved policy
target) and `z` is the final game outcome from that state's mover's perspective.
"""
from __future__ import annotations

import numpy as np

from alphazero.game import Game
from alphazero.mcts import MCTS


def play_game(game: Game, network, device: str, mcts_cfg):
    """Play one self-play game. Returns a list of (encoded, policy, value) samples."""
    mcts = MCTS(game, network, device, mcts_cfg)
    state = game.initial_state()

    history = []  # (encoded, policy, player_to_move)
    move_num = 0

    while not game.is_terminal(state):
        policy = mcts.run(state, add_noise=True)
        encoded = game.encode(state)
        history.append((encoded, policy, game.current_player(state)))

        # Temperature: explore early, exploit later.
        if move_num < mcts_cfg.temp_moves:
            action = np.random.choice(len(policy), p=policy)
        else:
            action = int(np.argmax(policy))

        state = game.next_state(state, action)
        move_num += 1

    outcome = game.result(state)  # from player +1's perspective

    samples = []
    for encoded, policy, player in history:
        z = outcome * player  # outcome from this state's mover's perspective
        for enc_sym, pol_sym in game.symmetries(encoded, policy):
            samples.append((enc_sym, pol_sym, float(z)))
    return samples
