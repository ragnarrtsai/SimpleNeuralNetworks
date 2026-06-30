# AlphaZero (self-play RL) — Connect 4

A small, from-scratch [AlphaZero](https://www.nature.com/articles/nature24270)
implementation: a neural network learns to play purely by playing against
itself, guided by Monte Carlo Tree Search. No human games, no heuristics.

The first game is **Connect 4**, but the core (network, MCTS, self-play,
training) is game-agnostic — adding Othello or Gomoku just means writing a new
`Game` subclass.

> New to the project or non-technical? Read [OVERVIEW.md](OVERVIEW.md) first —
> it explains what this is and how it works in plain language.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
source venv/bin/activate
python -m alphazero.train
```

Each **iteration** runs three stages:

1. **Self-play** — the current best network plays itself with MCTS, recording
   `(board, search-policy, game-outcome)` samples.
2. **Train** — a candidate network learns to match the search policy (the move
   the search settled on) and predict the outcome.
3. **Arena** — the candidate plays the current best; it's promoted only if it
   wins ≥55% of games. This keeps the data quality going up.

Checkpoints land in `checkpoints/` (`best.pt` is the current champion);
`progress.json` records the last finished iteration.

**Resume is the default.** Re-running `python -m alphazero.train` auto-continues
from `best.pt` at the next iteration instead of starting over. To train from
scratch, pass `--fresh`. Other useful flags:

```bash
python -m alphazero.train --iterations 80         # train further (raises the cap)
python -m alphazero.train --games 50 --sims 100   # smaller/faster iterations
python -m alphazero.train --fresh                 # ignore checkpoints, start over
python -m alphazero.train --device cpu            # small nets are often faster on CPU
```

Tune the defaults in [`config.py`](config.py) — scale `num_res_blocks`,
`num_simulations`, and `games_per_iteration` up for stronger play.

## Play against it

**Web UI** (recommended):

```bash
python webplay.py          # then open http://127.0.0.1:8000
```

- Click a column to drop a disc; the most recent move glows.
- **First move**: random / you first / AI first.
- **Difficulty**: 簡單 50 / 普通 200 / 困難 600 / 超難 1200 MCTS simulations per move
  (more = stronger and slower; the network is identical, only search depth changes).
- **Show analysis**: on your turn, a win-% estimate and per-column score bars show
  what the network + search think about the position.

**Terminal**:

```bash
python play.py --sims 400          # you are O, AI moves first
python play.py --human-first       # you are X, you move first
```

## Arena — compare checkpoints

The repo bundles trained models so you can compare strength right away:

| file | what it is |
|------|------------|
| `checkpoints/best.pt`        | the champion used by `play.py` / `webplay.py` |
| `checkpoints/weak.pt`        | a deliberately weak net (baseline) |
| `checkpoints/best_iter50.pt` | the champion after ~50 training iterations |
| `checkpoints/iter_120.pt`    | the snapshot after 120 iterations (strongest) |

**Two checkpoints** — head-to-head win rate:

```bash
python duel.py checkpoints/best.pt checkpoints/weak.pt --games 30
```

**Several checkpoints (round-robin arena)** — every pair plays, then a
leaderboard ranks them by score (win=1, draw=0.5). Handy for showing that more
training really makes the net stronger:

```bash
python arena.py checkpoints/weak.pt checkpoints/best_iter50.pt checkpoints/iter_120.pt --games 50
```

Sample run (50 games per pairing, 120 sims/move) — strength is cleanly
monotonic in training time:

```
 #  model                score    W    D    L
 1  iter_120              83.0   80    6   14
 2  best_iter50           57.5   53    9   38
 3  weak                   9.5    6    7   87
```

`iter_120` beats `best_iter50` 32–14 and the weak net never beats it (0–48).

Both alternate colors and randomize the opening plies so the games differ.
(This is the user-facing version of the Arena step the training loop uses
internally.)

## Layout

```
config.py              # all hyperparameters
alphazero/
  game.py              # abstract Game interface
  games/connect4.py    # Connect 4 rules + board<->tensor encoding
  network.py           # ResNet with policy + value heads
  mcts.py              # PUCT Monte Carlo Tree Search
  selfplay.py          # generate training games
  replay_buffer.py     # FIFO sample store
  arena.py             # candidate-vs-best evaluation
  train.py             # the full self-play -> train -> arena loop (with resume)
webplay.py             # web UI server (play + analysis)
templates/connect4.html# web UI front-end
play.py                # human-vs-AI CLI
duel.py                # pit two checkpoints against each other
arena.py               # round-robin several checkpoints -> leaderboard
tests/                 # game-logic sanity tests  (python -m pytest)
```

## Adding a new game

Implement the `Game` interface in [`alphazero/game.py`](alphazero/game.py)
(state, legal moves, terminal/result, and a `(channels,H,W)` tensor encoding
from the mover's perspective), then register it in
[`alphazero/games/__init__.py`](alphazero/games/__init__.py). The network, MCTS,
self-play, training, and arena work unchanged — run with `--game <name>`.
