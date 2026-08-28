#!/usr/bin/env python3
"""Download Flickr8k from Kaggle via kagglehub.

Usage:
    python scripts/download_data.py

Requires Kaggle API credentials to be configured (either a
``~/.kaggle/kaggle.json`` file or the ``KAGGLE_USERNAME`` /
``KAGGLE_KEY`` environment variables) -- see the README's "Dataset"
section for setup instructions.
"""

from __future__ import annotations

import argparse

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pathlib import Path

from captioner.config import load_config, resolve_path  # noqa: E402
from captioner.data.download import (  # noqa: E402
    DatasetPaths,
    _find_captions_file,
    copy_into_processed,
    download_flickr8k,
    find_images_dir,
    verify_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument(
        "--local-dir",
        default=None,
        help=(
            "Skip the Kaggle download entirely and use a dataset already "
            "extracted on disk (e.g. a manually downloaded/unzipped Kaggle "
            "archive folder). The folder is searched recursively for the "
            "images subfolder and captions file named in config.yaml."
        ),
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    raw_dir = resolve_path(cfg.dataset.raw_dir)
    processed_dir = resolve_path(cfg.dataset.processed_dir)

    if args.local_dir:
        root = Path(args.local_dir).expanduser().resolve()
        if not root.exists():
            raise SystemExit(f"--local-dir does not exist: {root}")
        logger.info("Using local dataset directory (no Kaggle download): %s", root)
        images_path = find_images_dir(root, cfg.dataset.images_subdir)
        captions_path = _find_captions_file(root, cfg.dataset.captions_file)
        logger.info("Resolved images directory: %s", images_path)
        logger.info("Resolved captions file: %s", captions_path)
        paths = DatasetPaths(root=root, images_dir=images_path, captions_file=captions_path)
    else:
        paths = download_flickr8k(
            kaggle_handle=cfg.dataset.kaggle_handle,
            output_dir=raw_dir,
            images_subdir=cfg.dataset.images_subdir,
            captions_file=cfg.dataset.captions_file,
        )
    verify_dataset(paths)
    stable_paths = copy_into_processed(paths, processed_dir)
    from captioner.data.download import write_dataset_manifest
    write_dataset_manifest(paths, processed_dir)

    logger.info("Images directory : %s", stable_paths.images_dir)
    logger.info("Captions file    : %s", stable_paths.captions_file)
    logger.info("Download complete.")


if __name__ == "__main__":
    main()
