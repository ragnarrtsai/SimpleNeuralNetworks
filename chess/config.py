"""Central configuration for the AlphaZero training pipeline.

Defaults are sized to train a strong Connect-4 agent on a laptop (CPU/MPS) in a
few hours. Scale up `num_res_blocks`, `num_simulations`, and
`games_per_iteration` for stronger play or harder games.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NetworkConfig:
    channels: int = 64          # filters per conv layer
    num_res_blocks: int = 5     # residual tower depth


@dataclass
class MCTSConfig:
    num_simulations: int = 100   # rollouts per move
    c_puct: float = 1.5          # PUCT exploration constant
    dirichlet_alpha: float = 1.0 # root noise (higher for small action spaces)
    dirichlet_epsilon: float = 0.25
    temp_moves: int = 10         # plies of temperature=1 sampling before going greedy


@dataclass
class SelfPlayConfig:
    games_per_iteration: int = 100


@dataclass
class TrainConfig:
    iterations: int = 50
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs_per_iteration: int = 4
    replay_buffer_size: int = 50_000
    checkpoint_every: int = 1
    eval_games: int = 40          # arena games when judging a new net
    eval_win_threshold: float = 0.55


@dataclass
class Config:
    game: str = "connect4"
    seed: int = 0
    device: str = "auto"  # "auto" -> cuda > mps > cpu
    checkpoint_dir: str = "checkpoints"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
