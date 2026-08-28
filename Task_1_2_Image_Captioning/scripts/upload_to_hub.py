#!/usr/bin/env python3
"""Upload the trained model checkpoint, vocabulary, and config to a
Hugging Face Hub model repository so it can be referenced with a
public/shareable link from the README.

Requires ``huggingface-cli login`` (or the ``HF_TOKEN`` environment
variable) to be configured beforehand.

Usage:
    python scripts/upload_to_hub.py --repo-id your-username/flickr8k-caption-gen
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from captioner.config import load_config, resolve_path  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=None, help="e.g. your-username/flickr8k-caption-gen")
    parser.add_argument("--private", action="store_true", help="Create/push to a private repo")
    args = parser.parse_args()

    from huggingface_hub import HfApi, create_repo

    cfg = load_config()
    repo_id = args.repo_id or cfg.hub.repo_id

    ckpt_dir = resolve_path(cfg.training.checkpoint_dir)
    best_ckpt = ckpt_dir / cfg.training.best_checkpoint_name
    vocab_path = resolve_path(cfg.vocab.vocab_path)
    config_path = resolve_path("configs/config.yaml")
    metrics_path = resolve_path(cfg.evaluation.results_dir) / "metrics.json"

    for required in (best_ckpt, vocab_path, config_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Required artifact missing: {required}. Run the full pipeline "
                f"(train + evaluate) before uploading."
            )

    api = HfApi()
    create_repo(repo_id, private=args.private, exist_ok=True, repo_type="model")

    files_to_upload = {
        best_ckpt: "best_model.pt",
        vocab_path: "vocab.json",
        config_path: "config.yaml",
    }
    if metrics_path.exists():
        files_to_upload[metrics_path] = "metrics.json"

    for local_path, remote_name in files_to_upload.items():
        logger.info("Uploading %s -> %s/%s", local_path, repo_id, remote_name)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_name,
            repo_id=repo_id,
            repo_type="model",
        )

    logger.info("Done. Model available at: https://huggingface.co/%s", repo_id)


if __name__ == "__main__":
    main()
