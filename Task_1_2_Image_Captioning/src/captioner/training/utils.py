"""Small, reusable training utilities: early stopping and checkpointing."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def set_global_seed(seed: int) -> torch.Generator:
    """Seed Python's ``random``, NumPy, and PyTorch (CPU + all CUDA devices)
    for reproducible training runs, and return a seeded ``torch.Generator``
    for use with a shuffling ``DataLoader`` (the piece of run-to-run
    randomness that a bare ``torch.manual_seed`` call doesn't cover).

    Full bitwise determinism isn't guaranteed (some CUDA/cuDNN kernels are
    non-deterministic regardless of seeding), but this removes every
    *avoidable* source of run-to-run variance: split shuffling, weight
    initialization, dropout, and DataLoader batch ordering.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class EarlyStopping:
    """Stops training when a monitored metric stops improving.

    ``mode='min'`` for loss-like metrics, ``mode='max'`` for BLEU-like
    metrics. Tracks the best value seen and how many consecutive
    evaluations have passed without improvement.
    """

    def __init__(self, patience: int = 5, mode: str = "min", min_delta: float = 1e-4):
        assert mode in ("min", "max")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_value: Optional[float] = None
        self.num_bad_epochs = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Update state with the latest metric value. Returns True if this
        is the new best value (caller should checkpoint)."""
        is_better = (
            self.best_value is None
            or (self.mode == "min" and value < self.best_value - self.min_delta)
            or (self.mode == "max" and value > self.best_value + self.min_delta)
        )
        if is_better:
            self.best_value = value
            self.num_bad_epochs = 0
            return True

        self.num_bad_epochs += 1
        if self.num_bad_epochs >= self.patience:
            self.should_stop = True
        return False


@dataclass
class TrainingHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_bleu4: list = field(default_factory=list)
    learning_rates: list = field(default_factory=list)
    epochs_completed: int = 0


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    vocab_size: int,
    config: dict,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "vocab_size": vocab_size,
        "config": config,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)
