from functools import partial

import torch
from torch.utils.data import DataLoader

from captioner.data.dataset import FeatureCaptionDataset, collate_captions
from captioner.data.prepare_splits import make_split


def test_split_has_no_image_leakage(synthetic_records):
    captions_by_image = {}
    for r in synthetic_records:
        captions_by_image.setdefault(r.image_id, []).append(r.caption)

    split = make_split(
        captions_by_image, images_dir="unused", train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=1
    )
    train_ids = {r.image_id for r in split.train}
    val_ids = {r.image_id for r in split.val}
    test_ids = {r.image_id for r in split.test}

    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)
    # every caption of a given image lands in exactly one split
    all_ids = train_ids | val_ids | test_ids
    assert all_ids == set(captions_by_image.keys())


def test_split_is_deterministic_given_seed(synthetic_records):
    captions_by_image = {}
    for r in synthetic_records:
        captions_by_image.setdefault(r.image_id, []).append(r.caption)

    split_a = make_split(captions_by_image, "unused", 0.5, 0.25, 0.25, seed=7)
    split_b = make_split(captions_by_image, "unused", 0.5, 0.25, 0.25, seed=7)
    assert {r.image_id for r in split_a.train} == {r.image_id for r in split_b.train}


def test_feature_caption_dataset_returns_expected_shapes(synthetic_records, synthetic_vocab, synthetic_feature_cache):
    ds = FeatureCaptionDataset(synthetic_records, synthetic_feature_cache, synthetic_vocab, max_caption_len=20)
    item = ds[0]
    assert item["feature"].shape == (49, 2048)
    assert item["caption_ids"].dim() == 1
    assert item["caption_ids"][0].item() == synthetic_vocab.start_id


def test_collate_pads_batch_correctly(synthetic_records, synthetic_vocab, synthetic_feature_cache):
    ds = FeatureCaptionDataset(synthetic_records, synthetic_feature_cache, synthetic_vocab, max_caption_len=20)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=partial(collate_captions, pad_id=synthetic_vocab.pad_id))
    batch = next(iter(loader))

    assert batch["features"].shape[0] == 8
    assert batch["captions"].shape[0] == 8
    assert batch["features"].shape[1:] == (49, 2048)
    # every row's length should match the recorded true length
    for i, length in enumerate(batch["lengths"]):
        row = batch["captions"][i]
        assert (row != synthetic_vocab.pad_id).sum().item() == length.item()
