"""
Ablation: FastText-initialised embeddings kept FROZEN during training
(instead of fine-tuned, as in the main modernised model).

Purpose
-------
The main experiment (see ``run_experiment.py``) showed the fine-tuned
FastText-initialised model trailing the randomly-initialised baseline on
every metric. A natural diagnostic question is whether fine-tuning on a
comparatively small, single-domain corpus is *distorting* the pretrained
vectors faster than the small amount of parallel data can put back useful
signal. Freezing the embeddings isolates that effect: if a frozen model
also under-performs the baseline, the gap is unlikely to be caused by
fine-tuning dynamics; if it under-performs the fine-tuned FastText model
too, that suggests the model needs to adapt the embeddings to the
translation objective and freezing is too restrictive at this data scale.

This script reuses the exact same data split, vocabulary and FastText
embedding matrices cached by ``run_experiment.py`` so the only variable
that changes is ``freeze_embeddings``.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch.nn as nn

from src import config
from src.data_prep import prepare_dataset
from src.dataset import make_loader
from src.evaluate import evaluate_translations
from src.inference import translate_sentence
from src.train import run_epoch, train_model
from src.utils import count_parameters, set_seed
from src.vocabulary import Vocabulary
from scripts.run_experiment import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("gensim").setLevel(logging.WARNING)
logger = logging.getLogger("run_frozen_ablation")

STAGE_NAME = "fasttext_frozen"


def main():
    set_seed(config.SEED)
    cfg = dict(config.CONFIG)

    train_df, val_df, test_df, source = prepare_dataset(force=False)
    logger.info("data source   : %s", source)
    logger.info("train/val/test: %d / %d / %d", len(train_df), len(val_df), len(test_df))

    src_vocab = Vocabulary(cfg["max_vocab_size"]).build(train_df["English"])
    tgt_vocab = Vocabulary(cfg["max_vocab_size"]).build(train_df["French"])

    emb_cache = config.RESULTS_DIR / "fasttext" / "embedding_matrices.npz"
    if not emb_cache.exists():
        raise FileNotFoundError(
            f"{emb_cache} not found -- run scripts/run_experiment.py first "
            "so the FastText embedding matrices are cached."
        )
    cached = np.load(emb_cache)
    src_emb, tgt_emb = cached["src_emb"], cached["tgt_emb"]

    train_loader = make_loader(train_df, src_vocab, tgt_vocab, cfg["max_seq_len"], cfg["batch_size"], shuffle=True)
    val_loader = make_loader(val_df, src_vocab, tgt_vocab, cfg["max_seq_len"], cfg["batch_size"], shuffle=False)
    test_loader = make_loader(test_df, src_vocab, tgt_vocab, cfg["max_seq_len"], cfg["batch_size"], shuffle=False)

    out_dir = config.RESULTS_DIR / STAGE_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint.pt"

    logger.info("\n=== Training FROZEN-embedding ablation model ===")
    set_seed(config.SEED)
    model = build_model(src_vocab, tgt_vocab, cfg, src_emb=src_emb, tgt_emb=tgt_emb, freeze=True)
    logger.info("frozen-embedding model trainable parameters: %d", count_parameters(model))
    history = train_model(
        model, train_loader, val_loader, src_vocab.pad_idx, cfg, log_fn=logger.info, checkpoint_path=ckpt_path
    )

    import torch
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("\n=== Evaluating frozen-embedding model on the test set ===")
    references = test_df["French"].tolist()
    source_sentences = test_df["English"].tolist()
    hyps = [translate_sentence(model, s, src_vocab, tgt_vocab, cfg["max_seq_len"]) for s in source_sentences]
    metrics = evaluate_translations(hyps, references)

    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_idx)
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, src_vocab.pad_idx, cfg["grad_clip"], train=False)
    metrics.update(test_loss=test_loss, token_accuracy=test_acc)

    logger.info("frozen    : %s", metrics)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(
        {"English": source_sentences, "Reference_FR": references, "FrozenFastText_FR": hyps}
    ).to_csv(out_dir / "test_set_translations.csv", index=False)

    logger.info("\nFrozen-embedding ablation results written to %s", out_dir)
    return metrics


if __name__ == "__main__":
    main()
