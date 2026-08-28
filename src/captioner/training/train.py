"""Training loop for the caption model.

Implements:
  * teacher-forced cross-entropy training with padding masked out
  * gradient clipping
  * ReduceLROnPlateau learning-rate scheduling on validation loss
  * early stopping on validation loss
  * best/last checkpointing
  * a lightweight per-epoch validation BLEU-4 sample (cheap sanity signal;
    the authoritative, full evaluation happens in evaluation/evaluate.py)
"""

from __future__ import annotations

import json
import logging
import time
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from captioner.data.dataset import FeatureCaptionDataset, collate_captions
from captioner.data.prepare_splits import load_split_csv
from captioner.data.vocabulary import Vocabulary
from captioner.evaluation.metrics import corpus_bleu4
from captioner.models.caption_model import CaptionModel
from captioner.models.decoder import AttentionDecoder
from captioner.training.utils import EarlyStopping, TrainingHistory, save_checkpoint

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_model(vocab: Vocabulary, model_cfg, feature_dim: int) -> CaptionModel:
    decoder = AttentionDecoder(
        vocab_size=len(vocab),
        feature_dim=feature_dim,
        embed_dim=model_cfg.embed_dim,
        hidden_dim=model_cfg.hidden_dim,
        attention_dim=model_cfg.attention_dim,
        rnn_type=model_cfg.decoder_type,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
        pad_id=vocab.pad_id,
    )
    return CaptionModel(encoder=None, decoder=decoder, vocab=vocab)


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor, pad_id: int) -> torch.Tensor:
    """logits: (B, T, V); targets: (B, T) the *shifted* ground-truth tokens; lengths: valid lengths."""
    b, t, v = logits.shape
    mask = torch.arange(t, device=logits.device).unsqueeze(0) < lengths.unsqueeze(1)
    logits_flat = logits.reshape(-1, v)
    targets_flat = targets.reshape(-1)
    mask_flat = mask.reshape(-1)

    loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_id)
    losses = loss_fn(logits_flat, targets_flat)
    losses = losses * mask_flat.float()
    return losses.sum() / mask_flat.float().sum().clamp(min=1)


def run_epoch(model, loader, optimizer, device, pad_id, grad_clip, train: bool):
    model.decoder.train(train)
    total_loss, total_tokens = 0.0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        pbar = tqdm(loader, desc="train" if train else "val", leave=False)
        for batch in pbar:
            features = batch["features"].to(device)
            captions = batch["captions"].to(device)
            lengths = batch["lengths"].to(device)

            logits, _, decode_lengths = model(features, captions, lengths)
            targets = captions[:, 1:1 + logits.size(1)]
            loss = masked_cross_entropy(logits, targets, decode_lengths, pad_id)

            if train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), grad_clip)
                optimizer.step()

            n_tokens = decode_lengths.sum().item()
            total_loss += loss.item() * max(n_tokens, 1)
            total_tokens += max(n_tokens, 1)
            pbar.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def quick_val_bleu(model, val_records, feature_cache_dir, vocab, device, max_images: int = 200, max_len: int = 20) -> float:
    """Cheap BLEU-4 estimate on a random sample of validation images (greedy decoding).
    Used for per-epoch monitoring only; full beam-search evaluation happens later.
    """
    import random
    from collections import defaultdict

    by_image: dict[str, list[str]] = defaultdict(list)
    for r in val_records:
        by_image[r.image_id].append(r.caption)

    image_ids = list(by_image.keys())
    random.Random(0).shuffle(image_ids)
    image_ids = image_ids[:max_images]

    hypotheses, references = [], []
    model.decoder.eval()
    for image_id in image_ids:
        import numpy as np
        feat_path = Path(feature_cache_dir) / f"{Path(image_id).stem}.npy"
        if not feat_path.exists():
            continue
        feat = torch.from_numpy(np.load(feat_path)).float().unsqueeze(0).to(device)
        result = model.generate_greedy(feat, max_len=max_len)
        hypotheses.append(result.caption)
        references.append(by_image[image_id])

    return corpus_bleu4(references, hypotheses)


def train_model(
    cfg,
    vocab: Vocabulary,
    train_split_csv: str | Path,
    val_split_csv: str | Path,
    feature_cache_dir: str | Path,
    resume: bool = False,
) -> TrainingHistory:
    device = resolve_device(cfg.training.device)
    logger.info("Training on device: %s", device)

    train_records = load_split_csv(train_split_csv)
    val_records = load_split_csv(val_split_csv)
    logger.info("Train pairs: %d | Val pairs: %d", len(train_records), len(val_records))

    train_ds = FeatureCaptionDataset(train_records, feature_cache_dir, vocab, cfg.vocab.max_caption_len)
    val_ds = FeatureCaptionDataset(val_records, feature_cache_dir, vocab, cfg.vocab.max_caption_len)

    collate = partial(collate_captions, pad_id=vocab.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers, collate_fn=collate,
    )

    model = build_model(vocab, cfg.model, cfg.features.feature_dim).to(device)
    optimizer = torch.optim.Adam(
        model.decoder.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=cfg.training.lr_scheduler.factor,
        patience=cfg.training.lr_scheduler.patience,
        min_lr=cfg.training.lr_scheduler.min_lr,
    )
    early_stopper = EarlyStopping(patience=cfg.training.early_stopping_patience, mode="min")

    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / cfg.training.best_checkpoint_name
    last_path = ckpt_dir / cfg.training.last_checkpoint_name

    history = TrainingHistory()
    start_epoch = 0

    if resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.decoder.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info("Resumed from epoch %d", start_epoch)

    for epoch in range(start_epoch, cfg.training.num_epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimizer, device, vocab.pad_id, cfg.training.grad_clip, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, vocab.pad_id, cfg.training.grad_clip, train=False)

        val_bleu4 = quick_val_bleu(model, val_records, feature_cache_dir, vocab, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_bleu4.append(val_bleu4)
        history.learning_rates.append(current_lr)
        history.epochs_completed = epoch + 1

        dt = time.time() - t0
        logger.info(
            "epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_BLEU-4=%.4f | lr=%.2e | %.1fs",
            epoch + 1, cfg.training.num_epochs, train_loss, val_loss, val_bleu4, current_lr, dt,
        )

        is_best = early_stopper.step(val_loss)
        save_checkpoint(last_path, model.decoder, optimizer, epoch, len(vocab), dict(cfg))
        if is_best:
            save_checkpoint(best_path, model.decoder, optimizer, epoch, len(vocab), dict(cfg),
                             extra={"val_loss": val_loss, "val_bleu4": val_bleu4})
            logger.info("New best model (val_loss=%.4f) saved to %s", val_loss, best_path)

        log_dir = Path(cfg.training.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(vars(history) if hasattr(history, "__dict__") else history.__dict__, f, indent=2)

        if early_stopper.should_stop:
            logger.info("Early stopping triggered after %d epochs without improvement.", cfg.training.early_stopping_patience)
            break

    return history
