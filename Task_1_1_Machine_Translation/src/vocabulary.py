"""
Word-level vocabulary with reserved special tokens.

This is the same frequency-ranked vocabulary construction used in the
original notebook: the ``max_size`` most frequent training-set words are
kept, plus four reserved tokens. What changes in the modernised pipeline is
*not* this class -- it is how the embedding matrix that sits behind these
indices is initialised (see ``src/embeddings.py``).
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Tuple


class Vocabulary:
    PAD = "<pad>"
    UNK = "<unk>"
    SOS = "<start>"
    EOS = "<end>"

    def __init__(self, max_size: int = 10_000):
        self.max_size = max_size
        self.word2idx = {self.PAD: 0, self.UNK: 1, self.SOS: 2, self.EOS: 3}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def build(self, sentences: Iterable[str]) -> "Vocabulary":
        counts: Counter = Counter()
        for sentence in sentences:
            counts.update(sentence.split())

        for word, _ in counts.most_common(self.max_size - len(self.word2idx)):
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        return self

    def encode(self, sentence: str, max_len: int, add_special: bool = False) -> Tuple[List[int], int]:
        tokens = sentence.split()
        if add_special:
            tokens = [self.SOS] + tokens + [self.EOS]
        ids = [self.word2idx.get(tok, self.word2idx[self.UNK]) for tok in tokens]
        ids = ids[:max_len]
        length = len(ids)
        ids = ids + [self.word2idx[self.PAD]] * (max_len - length)
        return ids, length

    def decode(self, ids, skip_special: bool = True) -> str:
        special = {self.PAD, self.UNK, self.SOS, self.EOS} if skip_special else set()
        words = []
        for idx in ids:
            word = self.idx2word.get(int(idx), self.UNK)
            if word == self.EOS:
                break
            if word in special:
                continue
            words.append(word)
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.word2idx)

    @property
    def pad_idx(self) -> int:
        return self.word2idx[self.PAD]

    @property
    def sos_idx(self) -> int:
        return self.word2idx[self.SOS]

    @property
    def eos_idx(self) -> int:
        return self.word2idx[self.EOS]

    @property
    def unk_idx(self) -> int:
        return self.word2idx[self.UNK]
