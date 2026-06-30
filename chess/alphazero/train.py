"""The AlphaZero training loop.

One iteration = self-play (generate data) -> train on the replay buffer ->
arena (promote the candidate only if it beats the current best). Repeat.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from alphazero.arena import evaluate
from alphazero.games import make_game
from alphazero.network import AlphaZeroNet
from alphazero.replay_buffer import ReplayBuffer
from alphazero.selfplay import play_game
from config import Config, resolve_device


def _loss(net, states, target_policy, target_value, device):
    states = torch.from_numpy(states).to(device)
    target_policy = torch.from_numpy(target_policy).to(device)
    target_value = torch.from_numpy(target_value).to(device)

    logits, value = net(states)
    # Policy loss: cross-entropy between MCTS visit distribution and net policy.
    log_probs = F.log_softmax(logits, dim=1)
    policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
    # Value loss: MSE against the game outcome.
    value_loss = F.mse_loss(value, target_value)
    return policy_loss + value_loss, policy_loss.item(), value_loss.item()


def train(cfg: Config, resume: str | None = None, start_iter: int = 1):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Game: {cfg.game} | Device: {device}")

    game = make_game(cfg.game)
    net = AlphaZeroNet(game, cfg.network.channels, cfg.network.num_res_blocks).to(device)
    # Resume: load a previous checkpoint into BOTH the candidate and the best
    # network, so training continues from where it left off instead of from a
    # fresh random net.
    if resume:
        if not os.path.exists(resume):
            raise FileNotFoundError(f"--resume checkpoint not found: {resume}")
        net.load_state_dict(torch.load(resume, map_location=device))
        print(f"Resuming from {resume} at iteration {start_iter}")
    best_net = copy.deepcopy(net)
    buffer = ReplayBuffer(cfg.train.replay_buffer_size)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    optimizer = torch.optim.Adam(
        net.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )

    if start_iter > cfg.train.iterations:
        print(f"Nothing to do: already at iteration {start_iter - 1} of "
              f"{cfg.train.iterations}. Raise --iterations to train further.")
        return

    for iteration in range(start_iter, cfg.train.iterations + 1):
        print(f"\n=== Iteration {iteration}/{cfg.train.iterations} ===")

        # 1. Self-play with the current best network.
        for _ in tqdm(range(cfg.selfplay.games_per_iteration), desc="self-play"):
            samples = play_game(game, best_net, device, cfg.mcts)
            buffer.add_game(samples)
        print(f"Buffer size: {len(buffer)}")

        # 2. Train the candidate network on sampled data.
        net.train()
        n_batches = max(1, len(buffer) // cfg.train.batch_size)
        for epoch in range(cfg.train.epochs_per_iteration):
            p_losses, v_losses = [], []
            for _ in range(n_batches):
                states, policies, values = buffer.sample(cfg.train.batch_size)
                loss, pl, vl = _loss(net, states, policies, values, device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                p_losses.append(pl)
                v_losses.append(vl)
            print(f"  epoch {epoch+1}: policy_loss={np.mean(p_losses):.4f} "
                  f"value_loss={np.mean(v_losses):.4f}")

        # 3. Arena: promote only if the candidate beats the current best.
        score = evaluate(game, net, best_net, device, cfg.mcts, cfg.train.eval_games)
        print(f"Candidate score vs best: {score:.3f} "
              f"(threshold {cfg.train.eval_win_threshold})")
        if score >= cfg.train.eval_win_threshold:
            print("  -> Promoting candidate to best.")
            best_net = copy.deepcopy(net)
            torch.save(best_net.state_dict(),
                       os.path.join(cfg.checkpoint_dir, "best.pt"))
        else:
            print("  -> Keeping current best; reverting candidate.")
            net = copy.deepcopy(best_net)
            optimizer = torch.optim.Adam(
                net.parameters(), lr=cfg.train.learning_rate,
                weight_decay=cfg.train.weight_decay,
            )

        if iteration % cfg.train.checkpoint_every == 0:
            path = os.path.join(cfg.checkpoint_dir, f"iter_{iteration:03d}.pt")
            torch.save(best_net.state_dict(), path)

        # Record true progress so re-runs resume from the right iteration,
        # regardless of stale iter_*.pt files left by earlier runs.
        with open(os.path.join(cfg.checkpoint_dir, "progress.json"), "w") as f:
            json.dump({"iteration": iteration, "game": cfg.game}, f)

    print("\nTraining complete. Best network saved to "
          f"{os.path.join(cfg.checkpoint_dir, 'best.pt')}")


def _latest_iter(checkpoint_dir: str) -> int:
    """Highest N among iter_NNN.pt files in the directory (0 if none)."""
    import re

    if not os.path.isdir(checkpoint_dir):
        return 0
    nums = [int(m.group(1)) for f in os.listdir(checkpoint_dir)
            if (m := re.fullmatch(r"iter_(\d+)\.pt", f))]
    return max(nums) if nums else 0


def _last_completed_iter(checkpoint_dir: str) -> int:
    """Last finished iteration. Trusts progress.json; falls back to file names.

    progress.json is authoritative because stale iter_*.pt files from earlier
    runs can otherwise inflate the count.
    """
    prog = os.path.join(checkpoint_dir, "progress.json")
    if os.path.exists(prog):
        try:
            with open(prog) as f:
                return int(json.load(f).get("iteration", 0))
        except (ValueError, OSError):
            pass
    return _latest_iter(checkpoint_dir)


def _parse_args() -> Config:
    import argparse

    cfg = Config()
    ap = argparse.ArgumentParser(description="AlphaZero self-play training")
    ap.add_argument("--game", default=cfg.game)
    ap.add_argument("--device", default=cfg.device, help="auto | cpu | mps | cuda")
    ap.add_argument("--iterations", type=int, default=cfg.train.iterations)
    ap.add_argument("--games", type=int, default=cfg.selfplay.games_per_iteration,
                    help="self-play games per iteration")
    ap.add_argument("--sims", type=int, default=cfg.mcts.num_simulations,
                    help="MCTS simulations per move")
    ap.add_argument("--res-blocks", type=int, default=cfg.network.num_res_blocks)
    ap.add_argument("--eval-games", type=int, default=cfg.train.eval_games)
    ap.add_argument("--resume", default=None,
                    help="checkpoint to continue from (default: auto-resume best.pt if present)")
    ap.add_argument("--start-iter", type=int, default=0,
                    help="iteration number to resume from (default: auto-detect)")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing checkpoints and train from scratch")
    args = ap.parse_args()

    cfg.game = args.game
    cfg.device = args.device
    cfg.train.iterations = args.iterations
    cfg.selfplay.games_per_iteration = args.games
    cfg.mcts.num_simulations = args.sims
    cfg.network.num_res_blocks = args.res_blocks
    cfg.train.eval_games = args.eval_games

    # Default behaviour: resume from the latest checkpoint so an accidental
    # re-run continues instead of wiping progress. Use --fresh to override.
    resume, start_iter = args.resume, args.start_iter
    if args.fresh:
        resume, start_iter = None, 1
    else:
        best = os.path.join(cfg.checkpoint_dir, "best.pt")
        if resume is None and os.path.exists(best):
            resume = best
            print(f"(auto-resume: found {best}; use --fresh to start over)")
        if start_iter == 0:  # not set by the user -> continue after the last iter
            last = _last_completed_iter(cfg.checkpoint_dir)
            start_iter = last + 1 if (resume and last > 0) else 1
    return cfg, resume, start_iter


if __name__ == "__main__":
    _cfg, _resume, _start = _parse_args()
    train(_cfg, resume=_resume, start_iter=_start)
