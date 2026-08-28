"""Shared pytest fixtures.

The full Flickr8k dataset is not available in CI/test environments, so
these fixtures generate a small, self-contained synthetic dataset
(a handful of solid-color images with simple template captions) that
exercises every stage of the pipeline -- tokenization, splitting,
feature extraction, batching, a forward/backward pass, generation, and
the API -- without requiring network access or GPU time.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch
from PIL import Image

from captioner.data.prepare_splits import CaptionRecord
from captioner.data.vocabulary import Vocabulary

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
TEMPLATES = [
    "a {color} square on a plain background",
    "this image shows a solid {color} shape",
    "a bright {color} colored square",
    "a small {color} square centered in the frame",
    "an image of a {color} block",
]
COLOR_NAMES = ["red", "green", "blue", "yellow", "cyan", "magenta"]


@pytest.fixture()
def synthetic_images_dir(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    image_ids = []
    for i, color in enumerate(COLORS * 2):  # 12 images total
        img = Image.new("RGB", (64, 64), color=color)
        name = f"img_{i:03d}.jpg"
        img.save(images_dir / name)
        image_ids.append(name)
    return images_dir, image_ids


@pytest.fixture()
def synthetic_records(synthetic_images_dir):
    images_dir, image_ids = synthetic_images_dir
    records = []
    for i, image_id in enumerate(image_ids):
        color_name = COLOR_NAMES[i % len(COLOR_NAMES)]
        for template in TEMPLATES:
            caption = template.format(color=color_name)
            records.append(CaptionRecord(image_id=image_id, image_path=str(images_dir / image_id), caption=caption))
    return records


@pytest.fixture()
def synthetic_vocab(synthetic_records):
    return Vocabulary.build([r.caption for r in synthetic_records], min_word_freq=1)


@pytest.fixture()
def synthetic_feature_cache(tmp_path, synthetic_images_dir):
    """Fake cached CNN features (random but deterministic), same shape as
    a real ResNet-50 spatial feature map (49 x 2048), so model code paths
    are exercised without needing pretrained ImageNet weights."""
    images_dir, image_ids = synthetic_images_dir
    cache_dir = tmp_path / "features"
    cache_dir.mkdir()
    rng = np.random.RandomState(0)
    for image_id in image_ids:
        feat = rng.randn(49, 2048).astype(np.float16)
        np.save(cache_dir / f"{Path(image_id).stem}.npy", feat)
    return cache_dir


@pytest.fixture(autouse=True)
def _deterministic_seed():
    torch.manual_seed(0)
    np.random.seed(0)
