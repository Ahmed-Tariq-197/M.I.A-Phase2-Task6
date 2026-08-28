"""Full test-set evaluation: generates a caption for every unique test
image (beam search), computes BLEU/ROUGE/METEOR against all 5 human
references, and saves a handful of qualitative
image -> generated caption -> reference captions examples.
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import torch

from captioner.data.prepare_splits import CaptionRecord, load_split_csv
from captioner.data.vocabulary import Vocabulary
from captioner.evaluation.metrics import evaluate_all_metrics
from captioner.models.caption_model import CaptionModel
from captioner.models.decoder import AttentionDecoder

logger = logging.getLogger(__name__)


def _group_by_image(records: List[CaptionRecord]) -> dict:
    grouped = defaultdict(list)
    for r in records:
        grouped[r.image_id].append(r.caption)
    return grouped


def load_model_for_eval(checkpoint_path: str | Path, vocab: Vocabulary, model_cfg, feature_dim: int, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    decoder = AttentionDecoder(
        vocab_size=len(vocab),
        feature_dim=feature_dim,
        embed_dim=model_cfg.embed_dim,
        hidden_dim=model_cfg.hidden_dim,
        attention_dim=model_cfg.attention_dim,
        rnn_type=model_cfg.decoder_type,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
        pad_id=vocab.pad_id,
    )
    decoder.load_state_dict(ckpt["model_state_dict"])
    decoder.to(device).eval()
    return CaptionModel(encoder=None, decoder=decoder, vocab=vocab)


def evaluate_test_set(
    cfg,
    vocab: Vocabulary,
    test_split_csv: str | Path,
    feature_cache_dir: str | Path,
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> dict:
    records = load_split_csv(test_split_csv)
    grouped = _group_by_image(records)
    image_ids = sorted(grouped.keys())
    logger.info("Evaluating on %d unique test images (%d reference captions total).", len(image_ids), len(records))

    model = load_model_for_eval(checkpoint_path, vocab, cfg.model, cfg.features.feature_dim, device)

    references, hypotheses, per_image = [], [], []
    for image_id in image_ids:
        feat_path = Path(feature_cache_dir) / f"{Path(image_id).stem}.npy"
        feat = torch.from_numpy(np.load(feat_path)).float().unsqueeze(0).to(device)
        result = model.generate_beam(feat, beam_size=cfg.evaluation.beam_size, max_len=cfg.evaluation.max_gen_len)

        references.append(grouped[image_id])
        hypotheses.append(result.caption)
        per_image.append({
            "image_id": image_id,
            "generated_caption": result.caption,
            "reference_captions": grouped[image_id],
            "beam_score": result.score,
        })

    metrics = evaluate_all_metrics(references, hypotheses)
    logger.info("Test metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    results_dir = Path(cfg.evaluation.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(results_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(per_image, f, indent=2, ensure_ascii=False)

    rng = random.Random(cfg.project.seed)
    sample = rng.sample(per_image, min(cfg.evaluation.num_qualitative_examples, len(per_image)))
    with open(results_dir / "qualitative_examples.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)

    images_dir = Path(cfg.dataset.raw_dir)
    try:
        from captioner.data.download import resolve_images_dir
        processed_dir = resolve_path_safe(cfg.dataset.processed_dir)
        images_dir = resolve_images_dir(
            resolve_path_safe(cfg.dataset.raw_dir), processed_dir, cfg.dataset.images_subdir
        )
    except Exception:  # noqa: BLE001 - visualization is best-effort, never fatal
        logger.warning("Could not locate the images directory for qualitative rendering; "
                        "qualitative_examples.json was still written.")
    else:
        render_qualitative_examples(sample, images_dir, results_dir / "qualitative_examples.png")

    return {"metrics": metrics, "num_images": len(image_ids), "results_dir": str(results_dir)}


def resolve_path_safe(p):
    from captioner.config import resolve_path
    return resolve_path(p)


def render_qualitative_examples(examples: list, images_dir: Path, out_path: Path) -> None:
    """Save a single contact-sheet image: each row is
    [input photo | generated caption | all 5 reference captions],
    exactly the 'Input Image -> Generated Caption -> Reference Captions'
    layout the qualitative evaluation calls for.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n = len(examples)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(11, 3.1 * n), gridspec_kw={"width_ratios": [1, 2]})
    if n == 1:
        axes = axes.reshape(1, 2)

    for row, example in enumerate(examples):
        img_path = Path(images_dir) / example["image_id"]
        ax_img, ax_text = axes[row, 0], axes[row, 1]
        try:
            with Image.open(img_path) as im:
                ax_img.imshow(im.convert("RGB"))
        except Exception:  # noqa: BLE001
            ax_img.text(0.5, 0.5, "image unavailable", ha="center", va="center")
        ax_img.axis("off")
        ax_img.set_title(example["image_id"], fontsize=8)

        refs = "\n".join(f"  - {r}" for r in example["reference_captions"])
        text = f"Generated: {example['generated_caption']}\n\nReferences:\n{refs}"
        ax_text.text(0.0, 0.5, text, fontsize=9, va="center", wrap=True, family="monospace")
        ax_text.axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved qualitative examples contact sheet to %s", out_path)
