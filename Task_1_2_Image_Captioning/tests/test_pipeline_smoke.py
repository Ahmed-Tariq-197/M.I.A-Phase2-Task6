"""End-to-end smoke test: split -> vocab -> (fake) features -> a few
training steps -> checkpoint -> beam-search evaluation -> qualitative
example, all wired together exactly as the real scripts do, but on a
tiny synthetic dataset so it runs in seconds without network or GPU
access.

This test exists purely to catch integration bugs between modules
(shape mismatches, wrong keys, off-by-one padding, etc.). It is not a
substitute for training on the real Flickr8k dataset, and the loss
values here carry no meaning about model quality.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from captioner.data.dataset import FeatureCaptionDataset, collate_captions
from captioner.data.prepare_splits import make_split, save_split_csv, load_split_csv
from captioner.evaluation.evaluate import load_model_for_eval
from captioner.evaluation.metrics import evaluate_all_metrics
from captioner.models.decoder import AttentionDecoder
from captioner.training.train import build_model, masked_cross_entropy, run_epoch
from captioner.training.utils import save_checkpoint


class _ModelCfg:
    embed_dim = 32
    hidden_dim = 64
    attention_dim = 32
    decoder_type = "lstm"
    num_layers = 1
    dropout = 0.0


def test_full_pipeline_smoke(tmp_path, synthetic_records, synthetic_vocab, synthetic_feature_cache):
    # -- split -------------------------------------------------------
    captions_by_image = {}
    for r in synthetic_records:
        captions_by_image.setdefault(r.image_id, []).append(r.caption)
    split = make_split(captions_by_image, "unused", train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=3)

    train_csv, val_csv, test_csv = tmp_path / "train.csv", tmp_path / "val.csv", tmp_path / "test.csv"
    save_split_csv(split.train, train_csv)
    save_split_csv(split.val, val_csv)
    save_split_csv(split.test, test_csv)
    assert len(load_split_csv(train_csv)) > 0

    # -- datasets ------------------------------------------------------
    train_records = load_split_csv(train_csv)
    val_records = load_split_csv(val_csv)
    train_ds = FeatureCaptionDataset(train_records, synthetic_feature_cache, synthetic_vocab, max_caption_len=20)
    val_ds = FeatureCaptionDataset(val_records, synthetic_feature_cache, synthetic_vocab, max_caption_len=20)
    collate = partial(collate_captions, pad_id=synthetic_vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, collate_fn=collate)

    # -- model / a few optimization steps -------------------------------
    model = build_model(synthetic_vocab, _ModelCfg(), feature_dim=2048)
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-3)

    losses = []
    for _ in range(3):
        train_loss = run_epoch(model, train_loader, optimizer, torch.device("cpu"), synthetic_vocab.pad_id, grad_clip=5.0, train=True)
        losses.append(train_loss)
    val_loss = run_epoch(model, val_loader, optimizer, torch.device("cpu"), synthetic_vocab.pad_id, grad_clip=5.0, train=False)

    assert all(l == l for l in losses)  # no NaNs
    assert val_loss == val_loss

    # -- checkpointing ---------------------------------------------------
    ckpt_path = tmp_path / "checkpoints" / "best_model.pt"
    save_checkpoint(ckpt_path, model.decoder, optimizer, epoch=2, vocab_size=len(synthetic_vocab), config={})
    assert ckpt_path.exists()

    # -- reload + beam-search generation ----------------------------------
    reloaded = load_model_for_eval(ckpt_path, synthetic_vocab, _ModelCfg(), feature_dim=2048, device=torch.device("cpu"))
    feat = torch.randn(1, 49, 2048)
    result = reloaded.generate_beam(feat, beam_size=2, max_len=10)
    assert isinstance(result.caption, str)

    # -- metrics on a couple of synthetic test examples --------------------
    references = [["a red square on a plain background"], ["a green square on a plain background"]]
    hypotheses = ["a red square on a background", "a green block shape"]
    metrics = evaluate_all_metrics(references, hypotheses)
    assert all(0.0 <= v <= 1.0 for v in metrics.values())
