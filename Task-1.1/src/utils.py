"""Small shared utilities."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_training_curves(history_a: dict, history_b: dict, label_a: str, label_b: str, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for history, label in ((history_a, label_a), (history_b, label_b)):
        epochs = range(1, len(history["val_loss"]) + 1)
        axes[0].plot(epochs, history["val_loss"], marker="o", label=label)
        axes[1].plot(epochs, history["val_acc"], marker="o", label=label)

    axes[0].set_title("Validation loss (pad ignored)")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Validation token accuracy (pad ignored)")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
