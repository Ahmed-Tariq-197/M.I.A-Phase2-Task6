"""Leakage-free train/validation/test splitting for Flickr8k.

Flickr8k has 5 captions per image. Splitting at the *caption* level
would let different captions of the *same* image land in both the
training and test sets -- the model would then be evaluated on an
image it (via a sibling caption) already saw during training. This
module always splits on the **unique image id**, so every caption of a
given image is guaranteed to fall in exactly one split.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class CaptionRecord:
    image_id: str
    image_path: str
    caption: str


@dataclass
class SplitResult:
    train: List[CaptionRecord] = field(default_factory=list)
    val: List[CaptionRecord] = field(default_factory=list)
    test: List[CaptionRecord] = field(default_factory=list)


def load_captions(captions_file: str | Path, images_dir: str | Path) -> Dict[str, List[str]]:
    """Parse the Kaggle ``captions.txt`` (columns: image,caption) into
    {image_id: [caption, caption, ...]} preserving all 5 captions per image.
    """
    captions_file = Path(captions_file)
    images_dir = Path(images_dir)
    by_image: Dict[str, List[str]] = defaultdict(list)

    with open(captions_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Some dumps of this dataset omit the header; detect and rewind if so.
        if header and header[0].strip().lower() not in ("image", "image_name", "filename"):
            f.seek(0)
            reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            image_name, caption = row[0].strip(), ",".join(row[1:]).strip()
            if not image_name or not caption:
                continue
            image_id = image_name.split("#")[0]  # some variants use "img.jpg#0"
            by_image[image_id].append(caption)

    # Drop any entries whose image file doesn't actually exist on disk.
    missing = [img for img in by_image if not (images_dir / img).exists()]
    for img in missing:
        del by_image[img]

    return dict(by_image)


def make_split(
    captions_by_image: Dict[str, List[str]],
    images_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> SplitResult:
    """Shuffle unique image ids with a fixed seed and cut into three splits."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "split ratios must sum to 1"

    images_dir = Path(images_dir)
    image_ids = sorted(captions_by_image.keys())  # sort first for determinism, then shuffle
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    n = len(image_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train:n_train + n_val])
    test_ids = set(image_ids[n_train + n_val:])

    # Guard against any accidental overlap (should be impossible by construction).
    assert not (train_ids & val_ids), "train/val overlap detected"
    assert not (train_ids & test_ids), "train/test overlap detected"
    assert not (val_ids & test_ids), "val/test overlap detected"

    result = SplitResult()
    for image_id, captions in captions_by_image.items():
        image_path = str(images_dir / image_id)
        target = (
            result.train if image_id in train_ids
            else result.val if image_id in val_ids
            else result.test
        )
        for cap in captions:
            target.append(CaptionRecord(image_id=image_id, image_path=image_path, caption=cap))

    return result


def save_split_csv(records: List[CaptionRecord], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "image_path", "caption"])
        for r in records:
            writer.writerow([r.image_id, r.image_path, r.caption])


def load_split_csv(path: str | Path) -> List[CaptionRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(CaptionRecord(**row))
    return records
