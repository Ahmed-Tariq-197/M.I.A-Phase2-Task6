"""Download the Flickr8k dataset from Kaggle.

The Kaggle mirror ``adityajn105/flickr8k`` ships a flat, easy-to-parse
layout (an ``Images/`` folder of JPEGs plus a single ``captions.txt``
CSV with five rows per image). Because different kagglehub versions /
cache states can nest that content one directory level deeper, this
module never hard-codes the final path -- it downloads, then *searches*
the resulting tree for ``captions.txt`` and an images folder, and
returns fully-resolved paths. That keeps every downstream script
reproducible regardless of where kagglehub happens to place its cache.

Two download backends are tried, in order:

1. ``kagglehub.dataset_download`` (the modern, recommended client).
2. The classic ``kaggle`` CLI (``kaggle datasets download -d ... --unzip``),
   used automatically if kagglehub raises (missing package, auth failure,
   network restriction, API change, etc.).

Both backends read Kaggle credentials the standard way -- a
``~/.kaggle/kaggle.json`` file, or the ``KAGGLE_USERNAME`` /
``KAGGLE_KEY`` environment variables -- so no credentials are ever
hard-coded here.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DatasetDownloadError(RuntimeError):
    """Raised when neither the kagglehub nor the Kaggle-CLI backend can
    fetch the dataset (e.g. no credentials configured, or the environment
    has no route to kaggle.com)."""


def _has_kaggle_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


@dataclass
class DatasetPaths:
    root: Path
    images_dir: Path
    captions_file: Path


def _find_captions_file(root: Path, expected_name: str) -> Path:
    # Exact name match first (fast path).
    direct = root / expected_name
    if direct.exists():
        return direct
    candidates = list(root.rglob(expected_name))
    if candidates:
        return candidates[0]
    # Fall back to any .txt file that looks like the captions table.
    for txt in root.rglob("*.txt"):
        if "caption" in txt.name.lower():
            return txt
    raise FileNotFoundError(
        f"Could not locate a captions file named '{expected_name}' under {root}. "
        f"Contents found: {[p.name for p in root.iterdir()]}"
    )


def find_images_dir(root: Path, expected_name: str) -> Path:
    direct = root / expected_name
    if direct.is_dir():
        return direct
    for d in root.rglob(expected_name):
        if d.is_dir():
            return d
    # Fall back: any directory containing a large number of .jpg files.
    best_dir, best_count = None, 0
    for d in root.rglob("*"):
        if d.is_dir():
            count = sum(1 for _ in d.glob("*.jpg"))
            if count > best_count:
                best_dir, best_count = d, count
    if best_dir is None or best_count == 0:
        raise FileNotFoundError(f"Could not locate an images folder under {root}.")
    return best_dir


def _download_via_kagglehub(kaggle_handle: str, output_dir: Path, force_download: bool) -> Path:
    import kagglehub

    logger.info("Attempting download via kagglehub: '%s' -> %s ...", kaggle_handle, output_dir)
    downloaded_path = kagglehub.dataset_download(
        kaggle_handle,
        output_dir=str(output_dir),
        force_download=force_download,
    )
    return Path(downloaded_path)


def _download_via_kaggle_cli(kaggle_handle: str, output_dir: Path) -> Path:
    """Fallback path: shell out to the official ``kaggle`` CLI.

    Equivalent to:
        kaggle datasets download -d <handle> -p <output_dir> --unzip
    """
    if shutil.which("kaggle") is None:
        # The CLI ships as a Python console-script from the `kaggle` pip
        # package; if it's not on PATH, try invoking it as a module instead.
        cmd = [sys.executable, "-m", "kaggle", "datasets", "download",
               "-d", kaggle_handle, "-p", str(output_dir), "--unzip"]
    else:
        cmd = ["kaggle", "datasets", "download", "-d", kaggle_handle,
               "-p", str(output_dir), "--unzip"]

    logger.info("Attempting download via Kaggle CLI: %s", " ".join(cmd))
    output_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        raise DatasetDownloadError(
            f"Kaggle CLI download failed (exit {result.returncode}).\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
        )
    return output_dir


def download_flickr8k(
    kaggle_handle: str,
    output_dir: str | Path,
    images_subdir: str = "Images",
    captions_file: str = "captions.txt",
    force_download: bool = False,
) -> DatasetPaths:
    """Download Flickr8k from Kaggle and return resolved dataset paths.

    Tries ``kagglehub`` first; if that raises for any reason (package
    missing, auth failure, network restriction, API error), automatically
    falls back to the ``kaggle`` CLI. Both backends pick up credentials
    from the standard ``~/.kaggle/kaggle.json`` file or
    ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` environment variables.

    Parameters
    ----------
    kaggle_handle:
        Kaggle dataset identifier, e.g. ``"adityajn105/flickr8k"``.
    output_dir:
        Local directory the dataset should be placed into.
    images_subdir / captions_file:
        Expected names inside the archive; used as hints for path discovery
        (the actual search is robust to nesting).
    force_download:
        Re-download even if a local copy already exists.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not _has_kaggle_credentials():
        logger.warning(
            "No Kaggle credentials detected (~/.kaggle/kaggle.json or "
            "KAGGLE_USERNAME/KAGGLE_KEY). Download attempts will likely fail "
            "with an authentication error until credentials are configured."
        )

    root: Optional[Path] = None
    errors = []

    try:
        root = _download_via_kagglehub(kaggle_handle, output_dir, force_download)
        logger.info("kagglehub reports dataset root: %s", root)
    except Exception as exc:  # noqa: BLE001 - any failure triggers the CLI fallback
        logger.warning("kagglehub download failed (%s: %s); falling back to the Kaggle CLI.",
                        type(exc).__name__, exc)
        errors.append(f"kagglehub: {type(exc).__name__}: {exc}")

    if root is None:
        try:
            root = _download_via_kaggle_cli(kaggle_handle, output_dir)
            logger.info("Kaggle CLI reports dataset root: %s", root)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"kaggle CLI: {type(exc).__name__}: {exc}")
            raise DatasetDownloadError(
                "Could not download '" + kaggle_handle + "' via kagglehub or the Kaggle CLI.\n"
                + "\n".join(errors)
                + "\n\nCheck: (1) Kaggle credentials are configured "
                  "(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY), "
                  "(2) this machine has outbound network access to kaggle.com / "
                  "api.kaggle.com, (3) the dataset handle is correct."
            ) from exc

    captions_path = _find_captions_file(root, captions_file)
    images_path = find_images_dir(root, images_subdir)
    logger.info("Resolved captions file: %s", captions_path)
    logger.info("Resolved images directory: %s (%d files)", images_path,
                sum(1 for _ in images_path.glob("*.jpg")))

    return DatasetPaths(root=root, images_dir=images_path, captions_file=captions_path)


def verify_dataset(paths: DatasetPaths, expected_min_images: int = 8000) -> None:
    """Sanity-check the downloaded dataset before the pipeline touches it."""
    n_images = sum(1 for _ in paths.images_dir.glob("*.jpg"))
    if n_images < expected_min_images * 0.95:
        raise RuntimeError(
            f"Expected roughly {expected_min_images} images, found {n_images} in "
            f"{paths.images_dir}. The download may be incomplete."
        )
    if not paths.captions_file.exists():
        raise RuntimeError(f"Captions file missing: {paths.captions_file}")
    logger.info("Dataset verified: %d images, captions file OK.", n_images)


def copy_into_processed(paths: DatasetPaths, processed_dir: str | Path) -> DatasetPaths:
    """Optionally materialize a stable local copy under data/processed.

    Kaggle cache locations can be evicted/cleared between machines; copying
    the captions file (small) locally makes downstream steps independent of
    where kagglehub decided to cache things. Images are left in place and
    referenced by absolute path (copying 8k JPEGs is unnecessary I/O).
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    dest_captions = processed_dir / paths.captions_file.name
    if not dest_captions.exists():
        shutil.copy2(paths.captions_file, dest_captions)
    return DatasetPaths(root=paths.root, images_dir=paths.images_dir, captions_file=dest_captions)


_MANIFEST_NAME = "dataset_paths.json"


def write_dataset_manifest(paths: DatasetPaths, processed_dir: str | Path) -> Path:
    """Record the resolved dataset paths (esp. the images directory) so
    every downstream script (split/features/train/evaluate) can find the
    images regardless of whether they came from a Kaggle download or a
    ``--local-dir`` on disk. Without this, scripts that independently
    re-search ``data/raw`` would fail whenever images live elsewhere."""
    import json

    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = processed_dir / _MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(
            {
                "root": str(paths.root),
                "images_dir": str(paths.images_dir),
                "captions_file": str(paths.captions_file),
            },
            indent=2,
        )
    )
    return manifest_path


def read_dataset_manifest(processed_dir: str | Path) -> Optional[DatasetPaths]:
    """Read back the manifest written by ``write_dataset_manifest``, or
    return ``None`` if it doesn't exist (e.g. an older run)."""
    import json

    manifest_path = Path(processed_dir) / _MANIFEST_NAME
    if not manifest_path.exists():
        return None
    data = json.loads(manifest_path.read_text())
    return DatasetPaths(
        root=Path(data["root"]),
        images_dir=Path(data["images_dir"]),
        captions_file=Path(data["captions_file"]),
    )


def resolve_images_dir(raw_dir: str | Path, processed_dir: str | Path, images_subdir: str) -> Path:
    """Preferred way for downstream scripts to locate the images folder:
    use the manifest written by download_data.py if present (works for
    both Kaggle and --local-dir runs), otherwise fall back to searching
    ``raw_dir`` directly (legacy behaviour)."""
    manifest = read_dataset_manifest(processed_dir)
    if manifest is not None and manifest.images_dir.is_dir():
        return manifest.images_dir
    return find_images_dir(Path(raw_dir), images_subdir)
