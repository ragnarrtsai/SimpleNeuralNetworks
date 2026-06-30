"""Round-robin arena: pit several checkpoints against each other and rank them.

    python arena.py checkpoints/weak.pt checkpoints/best_iter50.pt checkpoints/iter_120.pt
    python arena.py checkpoints/*.pt --games 20 --sims 120

Every pair plays --games games with alternating colors (reusing duel.py's game
logic and the random-opening trick for variety). Prints each pairing's result
and a leaderboard sorted by score (win=1, draw=0.5) — handy for showing that
more training really does make the net stronger.
"""
from __future__ import annotations

import argparse
import itertools
import os

from alphazero.games import make_game
from alphazero.mcts import MCTS
from config import Config, resolve_device
from duel import load, play_game


def label(path):
    return os.path.splitext(os.path.basename(path))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nets", nargs="+", help="2 or more checkpoint paths")
    ap.add_argument("--games", type=int, default=20, help="games per pairing")
    ap.add_argument("--sims", type=int, default=120, help="MCTS simulations per move")
    ap.add_argument("--opening", type=int, default=4, help="random opening plies")
    args = ap.parse_args()

    if len(args.nets) < 2:
        ap.error("need at least 2 checkpoints")

    cfg = Config()
    cfg.mcts.num_simulations = args.sims
    device = resolve_device(cfg.device)
    game = make_game(cfg.game)

    names = [label(p) for p in args.nets]
    mctss = [MCTS(game, load(game, p, device, cfg), device, cfg.mcts) for p in args.nets]

    n = len(args.nets)
    wins = [0] * n
    draws = [0] * n
    losses = [0] * n
    score = [0.0] * n
    cell = [["    -    "] * n for _ in range(n)]  # head-to-head, row's W-L-D vs col

    print(f"Arena: {n} models, {args.games} games/pair, {args.sims} sims/move\n")
    for i, j in itertools.combinations(range(n), 2):
        a = b = d = 0
        for g in range(args.games):
            if g % 2 == 0:  # i plays player +1
                r = play_game(game, mctss[i], mctss[j], args.opening)
                res = {1: "i", -1: "j", 0: "d"}[r]
            else:           # i plays player -1
                r = play_game(game, mctss[j], mctss[i], args.opening)
                res = {1: "j", -1: "i", 0: "d"}[r]
            if res == "i":
                a += 1
            elif res == "j":
                b += 1
            else:
                d += 1
        wins[i] += a; losses[i] += b; draws[i] += d
        wins[j] += b; losses[j] += a; draws[j] += d
        score[i] += a + 0.5 * d
        score[j] += b + 0.5 * d
        cell[i][j] = f"{a}-{b}-{d}"
        cell[j][i] = f"{b}-{a}-{d}"
        print(f"  {names[i]:>16} vs {names[j]:<16}  {a}-{b}-{d}  (W-L-D for {names[i]})")

    w = max(11, *(len(nm) for nm in names))

    # Head-to-head matrix: each cell is the ROW model's wins-losses-draws vs the COLUMN model.
    print("\nHead-to-head (row vs col, W-L-D for the row):")
    print(" " * w + "  " + "  ".join(f"{nm:>{w}}" for nm in names))
    for i in range(n):
        print(f"{names[i]:>{w}}  " + "  ".join(f"{cell[i][j]:>{w}}" for j in range(n)))

    order = sorted(range(n), key=lambda k: score[k], reverse=True)
    print("\n" + "=" * 54)
    print(f"{'#':>2}  {'model':<18} {'score':>7} {'W':>4} {'D':>4} {'L':>4}")
    print("-" * 54)
    for rank, k in enumerate(order, 1):
        print(f"{rank:>2}  {names[k]:<18} {score[k]:>7.1f} {wins[k]:>4} {draws[k]:>4} {losses[k]:>4}")
    print("=" * 54)


if __name__ == "__main__":
    main()
