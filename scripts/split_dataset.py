#!/usr/bin/env python3
"""Build a leakage-free train/validation/test split from the downloaded
Flickr8k captions file and write one CSV per split under
data/processed/.

Usage:
    python scripts/split_dataset.py
"""

from __future__ import annotations

import argparse

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from captioner.config import load_config, resolve_path  # noqa: E402
from captioner.data.prepare_splits import load_captions, make_split, save_split_csv  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed_dir = resolve_path(cfg.dataset.processed_dir)
    captions_file = processed_dir / cfg.dataset.captions_file
    raw_dir = resolve_path(cfg.dataset.raw_dir)

    # images live under the raw download (or wherever --local-dir pointed);
    # resolve_images_dir checks the manifest download_data.py wrote first,
    # falling back to searching raw_dir for older runs.
    from captioner.data.download import resolve_images_dir
    images_dir = resolve_images_dir(raw_dir, processed_dir, cfg.dataset.images_subdir)

    logger.info("Loading captions from %s", captions_file)
    captions_by_image = load_captions(captions_file, images_dir)
    logger.info("Found %d unique images with captions.", len(captions_by_image))

    split = make_split(
        captions_by_image,
        images_dir,
        train_ratio=cfg.dataset.split.train,
        val_ratio=cfg.dataset.split.val,
        test_ratio=cfg.dataset.split.test,
        seed=cfg.project.seed,
    )

    n_train_img = len({r.image_id for r in split.train})
    n_val_img = len({r.image_id for r in split.val})
    n_test_img = len({r.image_id for r in split.test})
    logger.info(
        "Split -> train: %d images / %d captions | val: %d images / %d captions | test: %d images / %d captions",
        n_train_img, len(split.train), n_val_img, len(split.val), n_test_img, len(split.test),
    )

    save_split_csv(split.train, processed_dir / "train.csv")
    save_split_csv(split.val, processed_dir / "val.csv")
    save_split_csv(split.test, processed_dir / "test.csv")
    logger.info("Wrote train.csv / val.csv / test.csv to %s", processed_dir)


if __name__ == "__main__":
    main()
