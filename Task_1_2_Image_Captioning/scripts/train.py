#!/usr/bin/env python3
"""Train the caption model on cached features + the training/validation splits.

Usage:
    python scripts/train.py
    python scripts/train.py --resume
    python scripts/train.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from captioner.config import load_config, resolve_path  # noqa: E402
from captioner.data.vocabulary import Vocabulary  # noqa: E402
from captioner.training.train import train_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to configs/config.yaml)")
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint if present")
    args = parser.parse_args()

    cfg = load_config(args.config)

    vocab_path = resolve_path(cfg.vocab.vocab_path)
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found at {vocab_path}. Run scripts/build_vocab.py first.")
    vocab = Vocabulary.load(vocab_path)

    processed_dir = resolve_path(cfg.dataset.processed_dir)
    feature_cache_dir = resolve_path(cfg.features.cache_dir)

    history = train_model(
        cfg=cfg,
        vocab=vocab,
        train_split_csv=processed_dir / "train.csv",
        val_split_csv=processed_dir / "val.csv",
        feature_cache_dir=feature_cache_dir,
        resume=args.resume,
    )
    logger.info("Training finished after %d epochs.", history.epochs_completed)
    logger.info("Best val loss: %.4f", min(history.val_loss) if history.val_loss else float("nan"))


if __name__ == "__main__":
    main()
