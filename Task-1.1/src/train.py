"""Training loop shared by the baseline and FastText-initialised models."""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


def token_accuracy(logits, targets, pad_idx) -> float:
    """Accuracy computed on non-pad tokens only."""
    preds = logits.argmax(dim=-1)
    mask = targets != pad_idx
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == targets[mask]).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, pad_idx, grad_clip, train: bool, device: str = "cpu"):
    model.train() if train else model.eval()

    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            tgt_out = batch["tgt_out"].to(device)

            logits, _ = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item()
            total_acc += token_accuracy(logits, tgt_out, pad_idx)
            n_batches += 1

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def _save_checkpoint(path: Path, model, optimizer, epoch, history, best_val, best_state, stale):
    tmp_path = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "best_val": best_val,
            "best_state": best_state,
            "stale": stale,
        },
        tmp_path,
    )
    tmp_path.replace(path)  # atomic on the same filesystem


def train_model(
    model,
    train_loader,
    val_loader,
    pad_idx: int,
    config: Dict,
    device: str = "cpu",
    log_fn=print,
    checkpoint_path: Optional[Path] = None,
) -> Dict:
    """Train ``model`` with early stopping on validation loss.

    If ``checkpoint_path`` is given, progress is saved to disk after every
    epoch and automatically resumed from there if the process is restarted
    (e.g. after being interrupted mid-run). The checkpoint file is removed
    once training finishes normally.

    Returns a dict with the training history and the best validation loss,
    and restores the best-performing weights into ``model`` in place.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val = math.inf
    best_state = None
    stale = 0
    start_epoch = 1
    total_time_offset = 0.0

    if checkpoint_path is not None and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        history = ckpt["history"]
        best_val = ckpt["best_val"]
        best_state = ckpt["best_state"]
        stale = ckpt["stale"]
        start_epoch = ckpt["epoch"] + 1
        total_time_offset = history.get("training_seconds", 0.0)
        log_fn(f"resuming from checkpoint at epoch {start_epoch} (best val loss so far = {best_val:.4f})")

    t0 = time.time()

    for epoch in range(start_epoch, config["epochs"] + 1):
        epoch_start = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, pad_idx, config["grad_clip"], train=True, device=device
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, pad_idx, config["grad_clip"], train=False, device=device
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - epoch_start
        log_fn(
            f"epoch {epoch:2d}/{config['epochs']}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}  ({elapsed:.1f}s)"
        )

        stopped_early = False
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config["patience"]:
                log_fn(f"early stopping at epoch {epoch} (best val loss = {best_val:.4f})")
                stopped_early = True

        if checkpoint_path is not None:
            history["training_seconds"] = total_time_offset + (time.time() - t0)
            _save_checkpoint(checkpoint_path, model, optimizer, epoch, history, best_val, best_state, stale)

        if stopped_early:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    total_time = total_time_offset + (time.time() - t0)
    log_fn(f"training finished in {total_time / 60:.1f} min, best val loss = {best_val:.4f}")

    history["best_val_loss"] = best_val
    history["training_seconds"] = total_time
    history["epochs_run"] = len(history["train_loss"])

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint_path.unlink()

    return history
