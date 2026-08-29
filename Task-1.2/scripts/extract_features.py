#!/usr/bin/env python3
"""Extract and cache pretrained-CNN features for every unique image
across the train/val/test splits.

Usage:
    python scripts/extract_features.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from captioner.config import load_config, resolve_path  # noqa: E402
from captioner.data.prepare_splits import load_split_csv  # noqa: E402
from captioner.features.extract_features import extract_and_cache_features  # noqa: E402
from captioner.training.train import resolve_device  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Use a randomly initialized backbone instead of ImageNet weights "
             "(only for environments without weight-download access, e.g. CI smoke tests).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = resolve_path(cfg.dataset.processed_dir)

    from captioner.data.download import resolve_images_dir
    raw_dir = resolve_path(cfg.dataset.raw_dir)
    images_dir = resolve_images_dir(raw_dir, processed_dir, cfg.dataset.images_subdir)

    all_image_ids = set()
    for split_name in ("train", "val", "test"):
        records = load_split_csv(processed_dir / f"{split_name}.csv")
        all_image_ids.update(r.image_id for r in records)

    logger.info("Extracting features for %d unique images across all splits.", len(all_image_ids))
    if args.no_pretrained:
        logger.warning("Running with --no-pretrained: features will NOT reflect ImageNet transfer "
                        "learning. Use only for pipeline smoke-testing, never for a real model.")

    device = resolve_device(cfg.training.device)
    n_processed = extract_and_cache_features(
        image_ids=sorted(all_image_ids),
        images_dir=images_dir,
        cache_dir=resolve_path(cfg.features.cache_dir),
        backbone=cfg.features.backbone,
        image_size=cfg.image.size,
        batch_size=cfg.training.batch_size,
        device=str(device),
        mean=cfg.image.mean,
        std=cfg.image.std,
        pretrained=not args.no_pretrained,
    )
    logger.info("Feature extraction complete. %d new images processed.", n_processed)


if __name__ == "__main__":
    main()
