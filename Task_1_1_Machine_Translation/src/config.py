"""
Central configuration for the machine translation project.

All paths are resolved relative to the project root so the code can be run
from any working directory without editing hard-coded paths.

Hyperparameters below reuse the original notebook's own ``WORKSHOP_MODE``
configuration (see the source notebook, section "Workshop knobs") as the
baseline, since that mode is already the notebook's documented, intended way
of running the pipeline quickly. ``SUBSET_SIZE`` and ``epochs`` are the only
values scaled down further, and only because this project runs on a single
CPU core rather than the GPU the notebook was written for -- every other
hyperparameter (vocabulary size, sequence length, embedding/hidden sizes,
batch size, dropout, gradient clipping, optimiser, early-stopping patience)
is taken unchanged from the notebook.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

for _d in (RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CORPUS_FILE = RAW_DATA_DIR / "en_fr_pairs_cleaned.tsv"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.tsv"
VAL_FILE = PROCESSED_DATA_DIR / "val.tsv"
TEST_FILE = PROCESSED_DATA_DIR / "test.tsv"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
# The full cleaned corpus has 130k+ pairs (see the original notebook). This
# project trains on a fixed-size random subsample of it -- the same strategy
# the notebook itself uses in WORKSHOP_MODE (there: 40,000 pairs on a GPU).
# SUBSET_SIZE is reduced further here to keep total training time reasonable
# on a single CPU core; set it to ``None`` to train on the full corpus.
SUBSET_SIZE = 20_000
TEST_FRACTION = 0.10
VAL_FRACTION = 0.10  # fraction of the remaining (non-test) data

# ---------------------------------------------------------------------------
# Model / training hyperparameters (identical to the notebook's
# WORKSHOP_MODE=True configuration).
# ---------------------------------------------------------------------------
CONFIG = {
    "max_vocab_size": 8_000,
    "max_seq_len": 20,
    "embed_dim": 256,
    "hidden_dim": 256,        # encoder hidden size, per direction
    "decoder_hidden": 512,    # 2 * hidden_dim, after merging both directions
    "batch_size": 64,
    "epochs": 8,
    "learning_rate": 1e-3,
    "dropout": 0.2,
    "patience": 3,
    "grad_clip": 1.0,
    "num_workers": 0,
}

# FastText hyperparameters used to train the replacement word embeddings.
# `bucket` (the number of hash buckets used for character n-grams) is kept
# far below gensim's default of 2,000,000: with a vocabulary of a few
# thousand words that default would allocate roughly 1 GB per language for
# no practical benefit. 100,000 buckets keep hash collisions negligible at
# this corpus size while using a fraction of the memory.
FASTTEXT_CONFIG = {
    "vector_size": CONFIG["embed_dim"],
    "window": 5,
    "min_count": 1,
    "epochs": 15,
    "sg": 1,          # skip-gram
    "min_n": 3,
    "max_n": 6,
    "bucket": 100_000,
    "workers": max(1, os.cpu_count() or 1),
}

DEVICE = "cpu"
