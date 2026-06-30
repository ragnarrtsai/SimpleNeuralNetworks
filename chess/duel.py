"""Pit two checkpoints against each other and report a win rate.

    python duel.py A.pt B.pt --games 30 --sims 120

Colors alternate every game for fairness. To avoid two fixed networks playing
the identical deterministic game every time, the first few plies are sampled
from the MCTS policy (temperature=1); after that both sides play greedily.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from alphazero.games import make_game
from alphazero.mcts import MCTS
from alphazero.network import AlphaZeroNet
from config import Config, resolve_device


def load(game, path, device, cfg):
    net = AlphaZeroNet(game, cfg.network.channels, cfg.network.num_res_blocks).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net


def play_game(game, mcts_p1, mcts_p2, opening_plies):
    state = game.initial_state()
    ply = 0
    while not game.is_terminal(state):
        mcts = mcts_p1 if game.current_player(state) == 1 else mcts_p2
        policy = mcts.run(state, add_noise=False)
        if ply < opening_plies:                       # diversify the opening
            action = int(np.random.choice(len(policy), p=policy))
        else:
            action = int(np.argmax(policy))
        state = game.next_state(state, action)
        ply += 1
    return game.result(state)  # +1 / -1 / 0 from player+1's view


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net_a")
    ap.add_argument("net_b")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--sims", type=int, default=120)
    ap.add_argument("--opening", type=int, default=4, help="random opening plies")
    args = ap.parse_args()

    cfg = Config()
    cfg.mcts.num_simulations = args.sims
    device = resolve_device(cfg.device)
    game = make_game(cfg.game)

    net_a = load(game, args.net_a, device, cfg)
    net_b = load(game, args.net_b, device, cfg)
    mcts_a = MCTS(game, net_a, device, cfg.mcts)
    mcts_b = MCTS(game, net_b, device, cfg.mcts)

    print(f"A = {args.net_a}\nB = {args.net_b}")
    print(f"{args.games} games, {args.sims} sims/move, {args.opening} random opening plies\n")

    a_wins = b_wins = draws = 0
    for g in range(args.games):
        if g % 2 == 0:  # A is player +1
            r = play_game(game, mcts_a, mcts_b, args.opening)
            outcome = {1: "A", -1: "B", 0: "draw"}[r]
        else:           # A is player -1
            r = play_game(game, mcts_b, mcts_a, args.opening)
            outcome = {1: "B", -1: "A", 0: "draw"}[r]
        if outcome == "A":
            a_wins += 1
        elif outcome == "B":
            b_wins += 1
        else:
            draws += 1
        print(f"  game {g+1:>2}: {outcome:>4}  (A {a_wins} - {b_wins} B, {draws} draws)")

    n = args.games
    score = (a_wins + 0.5 * draws) / n
    print("\n" + "=" * 40)
    print(f"A wins: {a_wins}  |  B wins: {b_wins}  |  draws: {draws}")
    print(f"A score: {score:.1%}   (win=1, draw=0.5)")
    print("=" * 40)


if __name__ == "__main__":
    main()
