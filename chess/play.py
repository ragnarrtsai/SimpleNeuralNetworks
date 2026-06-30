"""Play against a trained agent from the terminal.

    python play.py                      # play vs the latest best.pt (or random net)
    python play.py --checkpoint path.pt # play vs a specific checkpoint
    python play.py --sims 400           # stronger (slower) opponent
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from alphazero.games import make_game
from alphazero.mcts import MCTS
from alphazero.network import AlphaZeroNet
from config import Config, resolve_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="connect4")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--human-first", action="store_true", help="human plays player +1 (X)")
    args = ap.parse_args()

    cfg = Config()
    cfg.mcts.num_simulations = args.sims
    device = resolve_device(cfg.device)

    game = make_game(args.game)
    net = AlphaZeroNet(game, cfg.network.channels, cfg.network.num_res_blocks).to(device)
    if os.path.exists(args.checkpoint):
        net.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded {args.checkpoint}")
    else:
        print(f"No checkpoint at {args.checkpoint}; playing an untrained (random) net.")
    net.eval()

    mcts = MCTS(game, net, device, cfg.mcts)
    state = game.initial_state()
    human = 1 if args.human_first else -1

    while not game.is_terminal(state):
        print("\n" + game.render(state))
        if game.current_player(state) == human:
            legal = game.legal_actions(state)
            legal_cols = list(np.nonzero(legal)[0])
            while True:
                try:
                    col = int(input(f"Your move {legal_cols}: "))
                    if col in legal_cols:
                        break
                except ValueError:
                    pass
                print("Illegal column, try again.")
            state = game.next_state(state, col)
        else:
            policy = mcts.run(state, add_noise=False)
            action = int(np.argmax(policy))
            print(f"AI plays column {action}")
            state = game.next_state(state, action)

    print("\n" + game.render(state))
    result = game.result(state)  # from player +1's view
    if result == 0:
        print("Draw!")
    elif result == human:
        print("You win!")
    else:
        print("AI wins!")


if __name__ == "__main__":
    main()
