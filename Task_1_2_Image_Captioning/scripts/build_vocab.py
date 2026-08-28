#!/usr/bin/env python3
"""Build the vocabulary from the *training* split only (no leakage from
val/test captions) and save it to artifacts/vocab.json.

Usage:
    python scripts/build_vocab.py
"""

from __future__ import annotations

import argparse

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from captioner.config import load_config, resolve_path  # noqa: E402
from captioner.data.prepare_splits import load_split_csv  # noqa: E402
from captioner.data.vocabulary import Vocabulary  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed_dir = resolve_path(cfg.dataset.processed_dir)
    train_records = load_split_csv(processed_dir / "train.csv")
    logger.info("Building vocabulary from %d training captions...", len(train_records))

    vocab = Vocabulary.build(
        captions=[r.caption for r in train_records],
        min_word_freq=cfg.vocab.min_word_freq,
        pad_token=cfg.vocab.pad_token,
        start_token=cfg.vocab.start_token,
        end_token=cfg.vocab.end_token,
        unk_token=cfg.vocab.unk_token,
    )

    vocab_path = resolve_path(cfg.vocab.vocab_path)
    vocab.save(vocab_path)
    logger.info("Vocabulary size (incl. special tokens): %d", len(vocab))
    logger.info("Saved to %s", vocab_path)


if __name__ == "__main__":
    main()
