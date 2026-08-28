#!/usr/bin/env bash
# End-to-end pipeline: download -> split -> vocab -> features -> train -> evaluate.
# Intended to be run on a machine with real internet access (for the Kaggle
# download and, optionally, Hugging Face upload) and, ideally, a GPU.
#
# Usage:
#   bash scripts/run_pipeline.sh
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== 1/6 Downloading Flickr8k from Kaggle =="
python scripts/download_data.py

echo "== 2/6 Building leakage-free train/val/test split =="
python scripts/split_dataset.py

echo "== 3/6 Building vocabulary (training split only) =="
python scripts/build_vocab.py

echo "== 4/6 Extracting and caching CNN features =="
python scripts/extract_features.py

echo "== 5/6 Training the caption model =="
python scripts/train.py

echo "== 6/6 Evaluating on the held-out test set =="
python scripts/evaluate.py

echo "Pipeline complete. See artifacts/results/ for metrics and qualitative examples."
echo "Optional: python scripts/upload_to_hub.py --repo-id <your-username>/flickr8k-caption-gen"
