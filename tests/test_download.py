from pathlib import Path

import pytest

from captioner.data.download import DatasetDownloadError, find_images_dir, _find_captions_file, verify_dataset, DatasetPaths


def _make_fake_kaggle_layout(tmp_path: Path, nested: bool = False) -> Path:
    """Recreate the two directory shapes kagglehub is known to produce."""
    root = tmp_path / "flickr8k"
    base = (root / "versions" / "1") if nested else root
    images_dir = base / "Images"
    images_dir.mkdir(parents=True)
    for i in range(5):
        (images_dir / f"img_{i}.jpg").write_bytes(b"\xff\xd8\xff")  # minimal jpeg header stub
    (base / "captions.txt").write_text("image,caption\nimg_0.jpg,a caption\n")
    return root


def test_find_images_dir_flat_layout(tmp_path):
    root = _make_fake_kaggle_layout(tmp_path, nested=False)
    found = find_images_dir(root, "Images")
    assert found.name == "Images"
    assert sum(1 for _ in found.glob("*.jpg")) == 5


def test_find_images_dir_nested_layout(tmp_path):
    root = _make_fake_kaggle_layout(tmp_path, nested=True)
    found = find_images_dir(root, "Images")
    assert found.name == "Images"
    assert (found / "img_0.jpg").exists()


def test_find_captions_file_locates_nested_csv(tmp_path):
    root = _make_fake_kaggle_layout(tmp_path, nested=True)
    found = _find_captions_file(root, "captions.txt")
    assert found.name == "captions.txt"


def test_find_images_dir_raises_when_missing(tmp_path):
    empty_root = tmp_path / "empty"
    empty_root.mkdir()
    with pytest.raises(FileNotFoundError):
        find_images_dir(empty_root, "Images")


def test_verify_dataset_raises_on_too_few_images(tmp_path):
    root = _make_fake_kaggle_layout(tmp_path, nested=False)
    paths = DatasetPaths(root=root, images_dir=root / "Images", captions_file=root / "captions.txt")
    with pytest.raises(RuntimeError):
        verify_dataset(paths, expected_min_images=8000)


def test_verify_dataset_passes_with_matching_count(tmp_path):
    root = _make_fake_kaggle_layout(tmp_path, nested=False)
    paths = DatasetPaths(root=root, images_dir=root / "Images", captions_file=root / "captions.txt")
    verify_dataset(paths, expected_min_images=5)  # no exception
