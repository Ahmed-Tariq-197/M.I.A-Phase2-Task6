#!/usr/bin/env python3
"""Fetch the trained checkpoint + vocabulary from a Hugging Face Hub model
repository into the local artifacts/ layout expected by the serving apps.

This is what the Docker image runs at container startup: rather than
baking a multi-hundred-megabyte checkpoint into the image, the image
pulls it from the Hub the first time it starts (and caches it in a
volume for subsequent restarts).

Usage:
    python scripts/fetch_model_from_hub.py --repo-id your-username/flickr8k-caption-gen
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from captioner.config import load_config, resolve_path  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=None, help="e.g. your-username/flickr8k-caption-gen")
    parser.add_argument("--force", action="store_true", help="Re-download even if local artifacts already exist")
    args = parser.parse_args()

    cfg = load_config()
    repo_id = args.repo_id or cfg.hub.repo_id

    ckpt_dir = resolve_path(cfg.training.checkpoint_dir)
    ckpt_path = ckpt_dir / cfg.training.best_checkpoint_name
    vocab_path = resolve_path(cfg.vocab.vocab_path)

    if ckpt_path.exists() and vocab_path.exists() and not args.force:
        logger.info("Model artifacts already present locally (%s). Skipping download.", ckpt_path)
        return

    from huggingface_hub import hf_hub_download

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vocab_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading model artifacts from Hugging Face Hub repo '%s' ...", repo_id)
    downloaded_ckpt = hf_hub_download(repo_id=repo_id, filename="best_model.pt")
    downloaded_vocab = hf_hub_download(repo_id=repo_id, filename="vocab.json")

    shutil.copy2(downloaded_ckpt, ckpt_path)
    shutil.copy2(downloaded_vocab, vocab_path)
    logger.info("Model artifacts ready: %s, %s", ckpt_path, vocab_path)


if __name__ == "__main__":
    main()
