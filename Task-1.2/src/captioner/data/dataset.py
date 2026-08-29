"""PyTorch ``Dataset`` implementations.

Two datasets are provided:

``ImageOnlyDataset``
    Yields ``(image_id, image_tensor)``. Used exactly once, by
    ``extract_features.py``, to run every image through the frozen CNN
    backbone and cache the resulting features to disk.

``FeatureCaptionDataset``
    Yields ``(image_id, feature_tensor, caption_ids)``. Used for
    training/validation/testing: it reads the *cached* CNN features
    (fast, no repeated CNN forward passes) and pairs them with a
    numericalized caption. This is what "cache extracted features to
    improve training efficiency" buys us -- each epoch no longer pays
    the cost of a ResNet forward pass per image.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from captioner.data.prepare_splits import CaptionRecord
from captioner.data.vocabulary import Vocabulary


class ImageOnlyDataset(Dataset):
    """One entry per *unique* image (not per caption) -- used for feature extraction."""

    def __init__(self, image_ids: Sequence[str], images_dir: str | Path, transform):
        self.image_ids = list(image_ids)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_path = self.images_dir / image_id
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return image_id, tensor


class FeatureCaptionDataset(Dataset):
    """One entry per (image, caption) pair, backed by cached CNN features."""

    def __init__(
        self,
        records: List[CaptionRecord],
        feature_cache_dir: str | Path,
        vocab: Vocabulary,
        max_caption_len: int = 35,
    ):
        self.records = records
        self.feature_cache_dir = Path(feature_cache_dir)
        self.vocab = vocab
        self.max_caption_len = max_caption_len
        self._feature_cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load_feature(self, image_id: str) -> torch.Tensor:
        cached = self._feature_cache.get(image_id)
        if cached is not None:
            return cached
        feat_path = self.feature_cache_dir / f"{Path(image_id).stem}.npy"
        arr = np.load(feat_path)
        tensor = torch.from_numpy(arr).float()
        # Small in-memory LRU-ish cache to avoid re-reading disk for images
        # that repeat across the 5-captions-per-image structure.
        if len(self._feature_cache) < 4096:
            self._feature_cache[image_id] = tensor
        return tensor

    def __getitem__(self, idx: int):
        record = self.records[idx]
        feature = self._load_feature(record.image_id)
        caption_ids = self.vocab.encode(record.caption, max_len=self.max_caption_len)
        return {
            "image_id": record.image_id,
            "feature": feature,
            "caption_ids": torch.tensor(caption_ids, dtype=torch.long),
            "raw_caption": record.caption,
        }


def collate_captions(batch: List[dict], pad_id: int) -> dict:
    """Pad a batch of variable-length caption id sequences to the batch max
    length and stack image features. Also returns true lengths, needed for
    ``pack_padded_sequence``-style efficient LSTM training and for masking
    the loss on padding tokens.
    """
    batch = sorted(batch, key=lambda x: len(x["caption_ids"]), reverse=True)
    features = torch.stack([b["feature"] for b in batch], dim=0)
    lengths = torch.tensor([len(b["caption_ids"]) for b in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, b in enumerate(batch):
        cap = b["caption_ids"]
        padded[i, : len(cap)] = cap

    return {
        "image_ids": [b["image_id"] for b in batch],
        "features": features,
        "captions": padded,
        "lengths": lengths,
        "raw_captions": [b["raw_caption"] for b in batch],
    }
