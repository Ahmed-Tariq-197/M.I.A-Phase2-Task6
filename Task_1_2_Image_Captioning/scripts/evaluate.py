#!/usr/bin/env python3
"""Evaluate the best checkpoint on the held-out test split: computes
BLEU-1..4 / ROUGE-L / METEOR and saves qualitative examples.

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from captioner.config import load_config, resolve_path  # noqa: E402
from captioner.data.vocabulary import Vocabulary  # noqa: E402
from captioner.evaluation.evaluate import evaluate_test_set  # noqa: E402
from captioner.training.train import resolve_device  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Path to a model checkpoint (defaults to the best checkpoint)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vocab = Vocabulary.load(resolve_path(cfg.vocab.vocab_path))
    processed_dir = resolve_path(cfg.dataset.processed_dir)
    feature_cache_dir = resolve_path(cfg.features.cache_dir)
    checkpoint = resolve_path(
        args.checkpoint or Path(cfg.training.checkpoint_dir) / cfg.training.best_checkpoint_name
    )

    summary = evaluate_test_set(
        cfg=cfg,
        vocab=vocab,
        test_split_csv=processed_dir / "test.csv",
        feature_cache_dir=feature_cache_dir,
        checkpoint_path=checkpoint,
        device=str(resolve_device(cfg.training.device)),
    )
    logger.info("Evaluated %d test images. Results saved to %s", summary["num_images"], summary["results_dir"])
    for name, value in summary["metrics"].items():
        logger.info("  %-8s: %.4f", name, value)


if __name__ == "__main__":
    main()
