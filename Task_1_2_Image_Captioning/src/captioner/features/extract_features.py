"""Run every unique image through the frozen pretrained CNN once and
cache the resulting spatial feature map to disk as a ``.npy`` file.

Caching is what makes training tractable on modest hardware: without
it, every training epoch would repeat a full ResNet forward pass over
all 8,000 images. With it, that cost is paid exactly once, and every
subsequent epoch just reads small float16 arrays from disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from captioner.data.dataset import ImageOnlyDataset
from captioner.data.transforms import build_image_transform
from captioner.models.encoder import get_encoder

logger = logging.getLogger(__name__)


def extract_and_cache_features(
    image_ids: Iterable[str],
    images_dir: str | Path,
    cache_dir: str | Path,
    backbone: str = "resnet50",
    image_size: int = 224,
    batch_size: int = 32,
    device: str = "cpu",
    mean=None,
    std=None,
    skip_existing: bool = True,
    pretrained: bool = True,
) -> int:
    """Extract and cache CNN features for a collection of images.

    skip_existing:
        Existing cached files are skipped by default, so repeated runs are
        cheap and resumable.
    pretrained:
        Whether to load ImageNet-pretrained weights (the correct setting
        for any real run). Exposed as a parameter so integration tests /
        environments without weight-download access can still exercise
        the full extraction pipeline mechanically with a randomly
        initialized backbone.

    Returns the number of images actually processed.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    image_ids = list(image_ids)
    if skip_existing:
        pending = [i for i in image_ids if not (cache_dir / f"{Path(i).stem}.npy").exists()]
    else:
        pending = image_ids

    if not pending:
        logger.info("All %d images already cached in %s -- nothing to do.", len(image_ids), cache_dir)
        return 0

    logger.info("Extracting features for %d/%d images using %s ...", len(pending), len(image_ids), backbone)

    transform = build_image_transform(image_size=image_size, mean=mean, std=std, augment=False)
    dataset = ImageOnlyDataset(pending, images_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    encoder = get_encoder(backbone=backbone, pretrained=pretrained, fine_tune=False)
    encoder.to(device)
    encoder.eval()

    processed = 0
    with torch.no_grad():
        for image_ids_batch, images in tqdm(loader, desc="extract_features"):
            images = images.to(device)
            features = encoder(images)  # (B, num_pixels, feature_dim)
            features = features.cpu().numpy().astype(np.float16)
            for img_id, feat in zip(image_ids_batch, features):
                out_path = cache_dir / f"{Path(img_id).stem}.npy"
                np.save(out_path, feat)
                processed += 1

    logger.info("Cached %d feature files in %s", processed, cache_dir)
    return processed
