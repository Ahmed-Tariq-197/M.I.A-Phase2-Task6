"""
End-to-end experiment runner.

Prepares the dataset, trains the baseline (frequency-based / randomly
initialised embeddings) model and the modernised (FastText-initialised)
model under identical conditions, evaluates both with BLEU and ROUGE, and
writes every artifact needed by the README to ``results/``.

The two training stages are checkpointed independently (model weights +
history are written to disk as soon as each stage finishes), so re-running
the script after an interruption skips whatever already completed instead
of starting over.

Usage
-----
    python scripts/run_experiment.py                # full run (resumable)
    python scripts/run_experiment.py --force         # ignore checkpoints
    python scripts/run_experiment.py --quick         # fast smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from src import config
from src.data_prep import prepare_dataset
from src.dataset import make_loader
from src.embeddings import build_embedding_matrix, train_fasttext
from src.evaluate import evaluate_translations
from src.inference import translate_sentence
from src.model import Decoder, Encoder, Seq2Seq
from src.train import run_epoch, train_model
from src.utils import count_parameters, plot_training_curves, set_seed
from src.vocabulary import Vocabulary

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("gensim").setLevel(logging.WARNING)
logger = logging.getLogger("run_experiment")


def build_model(src_vocab, tgt_vocab, cfg, src_emb=None, tgt_emb=None, freeze=False):
    encoder = Encoder(
        vocab_size=len(src_vocab),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        pad_idx=src_vocab.pad_idx,
        dropout=cfg["dropout"],
        pretrained_embeddings=src_emb,
        freeze_embeddings=freeze,
    )
    decoder = Decoder(
        vocab_size=len(tgt_vocab),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["decoder_hidden"],
        pad_idx=tgt_vocab.pad_idx,
        dropout=cfg["dropout"],
        pretrained_embeddings=tgt_emb,
        freeze_embeddings=freeze,
    )
    return Seq2Seq(encoder, decoder, pad_idx=src_vocab.pad_idx)


def _stage_done(name: str) -> bool:
    out_dir = config.RESULTS_DIR / name
    return (out_dir / "model.pt").exists() and (out_dir / "history.json").exists()


def _save_stage(name: str, model, history: dict) -> None:
    out_dir = config.RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)


def _load_stage_model(name: str, src_vocab, tgt_vocab, cfg, src_emb=None, tgt_emb=None):
    model = build_model(src_vocab, tgt_vocab, cfg, src_emb=src_emb, tgt_emb=tgt_emb)
    state = torch.load(config.RESULTS_DIR / name / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    with open(config.RESULTS_DIR / name / "history.json") as f:
        history = json.load(f)
    return model, history


def run_pipeline(quick: bool = False, force: bool = False):
    set_seed(config.SEED)
    cfg = dict(config.CONFIG)
    if quick:
        cfg.update(epochs=2, patience=1)

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    train_df, val_df, test_df, source = prepare_dataset(force=force)
    if quick:
        train_df = train_df.sample(n=min(300, len(train_df)), random_state=config.SEED).reset_index(drop=True)
        val_df = val_df.sample(n=min(80, len(val_df)), random_state=config.SEED).reset_index(drop=True)
        test_df = test_df.sample(n=min(80, len(test_df)), random_state=config.SEED).reset_index(drop=True)

    logger.info("data source   : %s", source)
    logger.info("train/val/test: %d / %d / %d", len(train_df), len(val_df), len(test_df))

    # ------------------------------------------------------------------
    # 2. Vocabulary (shared by both models -- only the embedding init differs)
    # ------------------------------------------------------------------
    src_vocab = Vocabulary(cfg["max_vocab_size"]).build(train_df["English"])
    tgt_vocab = Vocabulary(cfg["max_vocab_size"]).build(train_df["French"])
    logger.info("src vocab size: %d, tgt vocab size: %d", len(src_vocab), len(tgt_vocab))

    train_loader = make_loader(train_df, src_vocab, tgt_vocab, cfg["max_seq_len"], cfg["batch_size"], shuffle=True)
    val_loader = make_loader(val_df, src_vocab, tgt_vocab, cfg["max_seq_len"], cfg["batch_size"], shuffle=False)
    test_loader = make_loader(test_df, src_vocab, tgt_vocab, cfg["max_seq_len"], cfg["batch_size"], shuffle=False)

    # ------------------------------------------------------------------
    # 3. Baseline model: frequency-based vocabulary, randomly initialised
    #    embeddings (exactly the original notebook's representation).
    # ------------------------------------------------------------------
    if not force and _stage_done("baseline") and not quick:
        logger.info("\n=== BASELINE already trained, loading checkpoint ===")
        baseline_model, baseline_history = _load_stage_model("baseline", src_vocab, tgt_vocab, cfg)
    else:
        logger.info("\n=== Training BASELINE model (random / frequency-based embeddings) ===")
        set_seed(config.SEED)
        baseline_model = build_model(src_vocab, tgt_vocab, cfg)
        logger.info("baseline trainable parameters: %d", count_parameters(baseline_model))
        baseline_ckpt = config.RESULTS_DIR / "baseline" / "checkpoint.pt"
        baseline_ckpt.parent.mkdir(parents=True, exist_ok=True)
        baseline_history = train_model(
            baseline_model, train_loader, val_loader, src_vocab.pad_idx, cfg,
            log_fn=logger.info, checkpoint_path=baseline_ckpt,
        )
        _save_stage("baseline", baseline_model, baseline_history)

    # ------------------------------------------------------------------
    # 4. FastText embeddings, trained on the training split only (no leakage).
    #    Cached to disk so a resumed run reuses the exact same vectors
    #    instead of retraining (gensim's multi-threaded training is not
    #    bit-for-bit deterministic across runs).
    # ------------------------------------------------------------------
    logger.info("\n=== Preparing FastText embeddings ===")
    emb_cache = config.RESULTS_DIR / "fasttext" / "embedding_matrices.npz"
    emb_cache.parent.mkdir(parents=True, exist_ok=True)
    if emb_cache.exists() and not force:
        cached = np.load(emb_cache)
        src_emb, tgt_emb = cached["src_emb"], cached["tgt_emb"]
        src_coverage, tgt_coverage = float(cached["src_coverage"]), float(cached["tgt_coverage"])
        ft_time = float(cached["ft_time"])
        logger.info("loaded cached FastText embedding matrices from %s", emb_cache)
    else:
        t0 = time.time()
        ft_cfg = dict(config.FASTTEXT_CONFIG)
        src_ft_model = train_fasttext(train_df["English"].tolist(), ft_cfg, seed=config.SEED)
        tgt_ft_model = train_fasttext(train_df["French"].tolist(), ft_cfg, seed=config.SEED)
        src_emb, src_coverage = build_embedding_matrix(src_vocab, src_ft_model, ft_cfg["vector_size"], seed=config.SEED)
        tgt_emb, tgt_coverage = build_embedding_matrix(tgt_vocab, tgt_ft_model, ft_cfg["vector_size"], seed=config.SEED)
        ft_time = time.time() - t0
        np.savez(
            emb_cache, src_emb=src_emb, tgt_emb=tgt_emb,
            src_coverage=src_coverage, tgt_coverage=tgt_coverage, ft_time=ft_time,
        )
    logger.info(
        "FastText embeddings ready in %.1fs | src coverage=%.1f%% tgt coverage=%.1f%%",
        ft_time, src_coverage * 100, tgt_coverage * 100,
    )

    if not force and _stage_done("fasttext") and not quick:
        logger.info("\n=== FASTTEXT model already trained, loading checkpoint ===")
        ft_model_seq2seq, ft_history = _load_stage_model("fasttext", src_vocab, tgt_vocab, cfg, src_emb=src_emb, tgt_emb=tgt_emb)
    else:
        logger.info("\n=== Training MODERNISED model (FastText-initialised embeddings) ===")
        set_seed(config.SEED)
        ft_model_seq2seq = build_model(src_vocab, tgt_vocab, cfg, src_emb=src_emb, tgt_emb=tgt_emb, freeze=False)
        logger.info("fasttext model trainable parameters: %d", count_parameters(ft_model_seq2seq))
        ft_ckpt = config.RESULTS_DIR / "fasttext" / "checkpoint.pt"
        ft_ckpt.parent.mkdir(parents=True, exist_ok=True)
        ft_history = train_model(
            ft_model_seq2seq, train_loader, val_loader, src_vocab.pad_idx, cfg,
            log_fn=logger.info, checkpoint_path=ft_ckpt,
        )
        _save_stage("fasttext", ft_model_seq2seq, ft_history)

    # ------------------------------------------------------------------
    # 5. Evaluation: BLEU + ROUGE on the held-out test set for both models
    # ------------------------------------------------------------------
    logger.info("\n=== Evaluating on the test set ===")
    references = test_df["French"].tolist()
    source_sentences = test_df["English"].tolist()

    def translate_all(model):
        return [
            translate_sentence(model, s, src_vocab, tgt_vocab, cfg["max_seq_len"])
            for s in source_sentences
        ]

    baseline_hyps = translate_all(baseline_model)
    ft_hyps = translate_all(ft_model_seq2seq)

    baseline_metrics = evaluate_translations(baseline_hyps, references)
    ft_metrics = evaluate_translations(ft_hyps, references)

    criterion = nn.CrossEntropyLoss(ignore_index=src_vocab.pad_idx)
    baseline_test_loss, baseline_test_acc = run_epoch(
        baseline_model, test_loader, criterion, None, src_vocab.pad_idx, cfg["grad_clip"], train=False
    )
    ft_test_loss, ft_test_acc = run_epoch(
        ft_model_seq2seq, test_loader, criterion, None, src_vocab.pad_idx, cfg["grad_clip"], train=False
    )
    baseline_metrics.update(test_loss=baseline_test_loss, token_accuracy=baseline_test_acc)
    ft_metrics.update(test_loss=ft_test_loss, token_accuracy=ft_test_acc)

    logger.info("baseline  : %s", baseline_metrics)
    logger.info("fasttext  : %s", ft_metrics)

    # ------------------------------------------------------------------
    # 6. Persist everything
    # ------------------------------------------------------------------
    for name, metrics in (("baseline", baseline_metrics), ("fasttext", ft_metrics)):
        out_dir = config.RESULTS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    sample_df = pd.DataFrame(
        {
            "English": source_sentences,
            "Reference_FR": references,
            "Baseline_FR": baseline_hyps,
            "FastText_FR": ft_hyps,
        }
    )
    sample_df.to_csv(config.RESULTS_DIR / "test_set_translations.csv", index=False)

    comparison = {
        "data_source": source,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "src_vocab_size": len(src_vocab),
        "tgt_vocab_size": len(tgt_vocab),
        "fasttext_training_seconds": ft_time,
        "fasttext_src_coverage": src_coverage,
        "fasttext_tgt_coverage": tgt_coverage,
        "baseline": baseline_metrics,
        "fasttext": ft_metrics,
        "baseline_epochs_run": baseline_history["epochs_run"],
        "fasttext_epochs_run": ft_history["epochs_run"],
        "baseline_training_seconds": baseline_history["training_seconds"],
        "fasttext_model_training_seconds": ft_history["training_seconds"],
    }
    with open(config.RESULTS_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    if not quick:
        plot_training_curves(
            baseline_history, ft_history, "baseline", "fasttext", config.FIGURES_DIR / "validation_curves.png"
        )

    logger.info("\nAll results written to %s", config.RESULTS_DIR)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a fast smoke test on a tiny subset.")
    parser.add_argument("--force", action="store_true", help="Ignore cached data/checkpoints and rerun everything.")
    args = parser.parse_args()
    run_pipeline(quick=args.quick, force=args.force)
