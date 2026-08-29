"""
Modern word-embedding layer used to replace the frequency-based
representation of the original notebook.

Why FastText
------------
The baseline notebook represents every word purely by its index in a
frequency-ranked vocabulary: the ``nn.Embedding`` table behind that index is
initialised at random and receives *all* of its semantic content from
back-propagation through the (fairly small) parallel corpus. Three modern
alternatives were considered for replacing that random initialisation:

* **GloVe** is trained from a global co-occurrence matrix. High-quality
  pretrained vectors exist for English, but not for French through the same
  distribution channel, and training GloVe from scratch requires building
  and factorising a co-occurrence matrix, which is a separate, heavier
  training procedure than the other two options.
* **Word2Vec** learns dense vectors from local context windows and is a
  natural fit for a corpus we control, but it represents each word as an
  atomic unit: it cannot produce a vector for a word it did not see during
  training, and it has no way to share statistics between related word
  forms (e.g. French verb conjugations, plurals, accented variants).
* **FastText** extends Word2Vec by representing each word as a bag of
  character n-grams. It is trained the same way (cheap, self-contained, no
  external download required) but shares sub-word statistics across
  morphologically related words -- which matters here because French is
  morphologically rich (accents, gender/number agreement, verb
  conjugations) and the training vocabulary is small.

FastText was selected because it keeps the project fully self-contained
(the vectors are trained on our own English and French training sentences,
so no multi-hundred-megabyte pretrained file needs to be fetched from a
third-party server) while still giving every word a distributional,
sub-word-aware representation instead of a purely random one -- directly
addressing the "frequency-based representation" the task asks to replace.
"""
from __future__ import annotations

from typing import List

import numpy as np
from gensim.models import FastText

from .vocabulary import Vocabulary


def train_fasttext(sentences: List[str], config: dict, seed: int = 42) -> FastText:
    """Train a FastText model on a list of whitespace-tokenised sentences."""
    tokenized = [s.split() for s in sentences if s.strip()]
    model = FastText(
        sentences=tokenized,
        vector_size=config["vector_size"],
        window=config["window"],
        min_count=config["min_count"],
        sg=config["sg"],
        min_n=config["min_n"],
        max_n=config["max_n"],
        bucket=config.get("bucket", 100_000),
        epochs=config["epochs"],
        workers=config["workers"],
        seed=seed,
    )
    return model


def build_embedding_matrix(vocab: Vocabulary, ft_model: FastText, vector_size: int, seed: int = 42) -> np.ndarray:
    """Build a (vocab_size, vector_size) matrix aligned with ``vocab``.

    Every vocabulary word that FastText can represent (either because it was
    seen during training, or via its character n-grams) is initialised with
    its learned vector. Reserved special tokens have no linguistic content
    of their own, so they get a small random vector (``<pad>`` is kept at
    zero so it never contributes to the embedding average).
    """
    rng = np.random.default_rng(seed)
    matrix = rng.normal(scale=0.1, size=(len(vocab), vector_size)).astype(np.float32)
    matrix[vocab.pad_idx] = 0.0

    special_tokens = {vocab.PAD, vocab.UNK, vocab.SOS, vocab.EOS}
    hits = 0
    for word, idx in vocab.word2idx.items():
        if word in special_tokens:
            continue
        try:
            matrix[idx] = ft_model.wv[word]
            hits += 1
        except KeyError:
            # Should essentially never happen for FastText (sub-word
            # fallback covers unseen character combinations too), but the
            # random initialisation above is kept as a safety net.
            continue

    coverage = hits / max(1, (len(vocab) - len(special_tokens)))
    return matrix, coverage
