"""
Dataset acquisition, cleaning, filtering and train/val/test splitting.

Dataset
-------
This project uses the same parallel corpus as the original notebook: the
tab-separated "eng-fra.txt" file (English, French) built from the Tatoeba
project and distributed as ``download.pytorch.org/tutorial/data.zip`` for
the classic PyTorch "NLP From Scratch" seq2seq tutorial that the workshop
notebook is based on.

The file must be present at ``data/raw/eng-fra.txt`` before running the
pipeline (135,842 tab-separated English/French sentence pairs, identical to
the file the original notebook downloads). See ``README.md`` for exact
download instructions. The loader intentionally does not fall back to a
different corpus: substituting the dataset would break the fair, apples-to-
apples comparison between the frequency-based baseline and the modernised
embedding model that Task 1.1 asks for.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .text_cleaning import clean_english, clean_french

logger = logging.getLogger(__name__)

ANKI_FILENAME = "eng-fra.txt"


class DatasetNotFoundError(FileNotFoundError):
    pass


def _load_from_local_anki_file(path: Path) -> List[Tuple[str, str]]:
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def load_raw_pairs() -> Tuple[List[Tuple[str, str]], str]:
    """Load the original English-French Anki/Tatoeba corpus.

    Raises ``DatasetNotFoundError`` with setup instructions if the file has
    not been placed in ``data/raw/`` yet.
    """
    anki_path = config.RAW_DATA_DIR / ANKI_FILENAME
    if not anki_path.exists():
        raise DatasetNotFoundError(
            f"Required dataset file not found at {anki_path}.\n"
            "Download the original English-French corpus used by the "
            "workshop notebook (the 'eng-fra.txt' file from "
            "download.pytorch.org/tutorial/data.zip) and place it at "
            f"{anki_path} before running the pipeline. See README.md, "
            "section 'Reproducing the results', for the exact steps."
        )
    logger.info("Loading original Anki/Tatoeba corpus from %s", anki_path)
    return _load_from_local_anki_file(anki_path), "anki_tatoeba_eng_fra"


def build_clean_dataframe() -> Tuple[pd.DataFrame, str]:
    """Load, clean and deduplicate the parallel corpus.

    This mirrors the original notebook's cleaning cell exactly: lower-case
    + contraction expansion + character whitelisting, then drop empty /
    duplicate rows. No sentence-length filtering is applied here (the
    original notebook does not filter by length either -- it relies on
    ``max_seq_len`` truncation inside vocabulary encoding for the rare long
    outliers). A fixed-size random subsample is then drawn, exactly as the
    original notebook's own ``WORKSHOP_MODE`` does for faster iteration.
    """
    raw_pairs, source_name = load_raw_pairs()

    df = pd.DataFrame(raw_pairs, columns=["English", "French"])
    df = df[df["English"].astype(str).str.strip().ne("") & df["French"].astype(str).str.strip().ne("")]
    df = df.drop_duplicates(subset=["English", "French"]).reset_index(drop=True)

    df["English"] = df["English"].map(clean_english)
    df["French"] = df["French"].map(clean_french)

    df = df[(df["English"].str.len() > 0) & (df["French"].str.len() > 0)]
    df = df.drop_duplicates(subset=["English", "French"]).reset_index(drop=True)

    if config.SUBSET_SIZE is not None and len(df) > config.SUBSET_SIZE:
        df = df.sample(n=config.SUBSET_SIZE, random_state=config.SEED).reset_index(drop=True)

    df.to_csv(config.RAW_CORPUS_FILE, sep="\t", index=False)
    return df, source_name


def split_and_save(df: pd.DataFrame, source_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df, test_size=config.TEST_FRACTION, random_state=config.SEED
    )
    train_df, val_df = train_test_split(
        train_df, test_size=config.VAL_FRACTION, random_state=config.SEED
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(config.TRAIN_FILE, sep="\t", index=False)
    val_df.to_csv(config.VAL_FILE, sep="\t", index=False)
    test_df.to_csv(config.TEST_FILE, sep="\t", index=False)
    with open(config.PROCESSED_DATA_DIR / "source.txt", "w") as f:
        f.write(source_name)
    return train_df, val_df, test_df


def prepare_dataset(force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """End-to-end dataset preparation, cached on disk after the first run."""
    source_file = config.PROCESSED_DATA_DIR / "source.txt"
    if (
        not force
        and config.TRAIN_FILE.exists()
        and config.VAL_FILE.exists()
        and config.TEST_FILE.exists()
        and source_file.exists()
    ):
        train_df = pd.read_csv(config.TRAIN_FILE, sep="\t", keep_default_na=False)
        val_df = pd.read_csv(config.VAL_FILE, sep="\t", keep_default_na=False)
        test_df = pd.read_csv(config.TEST_FILE, sep="\t", keep_default_na=False)
        source_name = source_file.read_text().strip()
        return train_df, val_df, test_df, source_name

    df, source_name = build_clean_dataframe()
    train_df, val_df, test_df = split_and_save(df, source_name)
    return train_df, val_df, test_df, source_name


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_df, val_df, test_df, source = prepare_dataset(force=True)
    print(f"source        : {source}")
    print(f"train pairs   : {len(train_df):,}")
    print(f"val pairs     : {len(val_df):,}")
    print(f"test pairs    : {len(test_df):,}")
    print(train_df.head())
