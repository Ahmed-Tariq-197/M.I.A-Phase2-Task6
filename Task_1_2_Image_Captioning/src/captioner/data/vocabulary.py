"""Caption tokenization, vocabulary construction, and numericalization.

The vocabulary is built **only from the training split** (never from
validation/test captions) to avoid leaking test-time words into the
model's known vocabulary -- a common and easy-to-miss source of leakage
in captioning pipelines.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(caption: str) -> List[str]:
    """Lowercase + regex word tokenizer.

    Kept dependency-free (no nltk punkt download required) and robust to
    the punctuation noise typical of Flickr8k captions (stray periods,
    commas, quotation marks).
    """
    caption = caption.strip().lower()
    return _TOKEN_RE.findall(caption)


class Vocabulary:
    """Bidirectional word<->index mapping with special tokens and an
    unknown-word bucket for rare/out-of-vocabulary words.
    """

    def __init__(
        self,
        pad_token: str = "<pad>",
        start_token: str = "<start>",
        end_token: str = "<end>",
        unk_token: str = "<unk>",
    ):
        self.pad_token = pad_token
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = unk_token

        self._word2idx: dict[str, int] = {}
        self._idx2word: dict[int, str] = {}
        self._freq: Counter[str] = Counter()

        for tok in (pad_token, start_token, end_token, unk_token):
            self._add_word(tok)

    # -- construction -----------------------------------------------------
    def _add_word(self, word: str) -> int:
        if word not in self._word2idx:
            idx = len(self._word2idx)
            self._word2idx[word] = idx
            self._idx2word[idx] = word
        return self._word2idx[word]

    @classmethod
    def build(
        cls,
        captions: Iterable[str],
        min_word_freq: int = 5,
        pad_token: str = "<pad>",
        start_token: str = "<start>",
        end_token: str = "<end>",
        unk_token: str = "<unk>",
    ) -> "Vocabulary":
        vocab = cls(pad_token, start_token, end_token, unk_token)
        counter: Counter[str] = Counter()
        for cap in captions:
            counter.update(tokenize(cap))
        vocab._freq = counter
        for word, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
            if count >= min_word_freq:
                vocab._add_word(word)
        return vocab

    # -- lookups ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._word2idx)

    def __contains__(self, word: str) -> bool:
        return word in self._word2idx

    @property
    def pad_id(self) -> int:
        return self._word2idx[self.pad_token]

    @property
    def start_id(self) -> int:
        return self._word2idx[self.start_token]

    @property
    def end_id(self) -> int:
        return self._word2idx[self.end_token]

    @property
    def unk_id(self) -> int:
        return self._word2idx[self.unk_token]

    def word_to_id(self, word: str) -> int:
        return self._word2idx.get(word, self.unk_id)

    def id_to_word(self, idx: int) -> str:
        return self._idx2word.get(idx, self.unk_token)

    def word_frequency(self, word: str) -> int:
        return self._freq.get(word, 0)

    # -- encode/decode --------------------------------------------------
    def encode(self, caption: str, max_len: int | None = None, add_special_tokens: bool = True) -> List[int]:
        """Tokenize + numericalize a raw caption string.

        Applies rare/unknown-word handling via ``word_to_id`` (OOV and
        sub-frequency-threshold words collapse to ``<unk>``), wraps with
        ``<start>``/``<end>``, and truncates to ``max_len`` if given.
        """
        tokens = tokenize(caption)
        ids = [self.word_to_id(t) for t in tokens]
        if add_special_tokens:
            ids = [self.start_id] + ids + [self.end_id]
        if max_len is not None:
            if add_special_tokens:
                ids = ids[: max_len - 1] + [self.end_id] if len(ids) > max_len else ids
                ids = ids[:max_len]
            else:
                ids = ids[:max_len]
        return ids

    def decode(self, ids: Sequence[int], strip_special: bool = True) -> str:
        words = []
        for i in ids:
            word = self.id_to_word(int(i))
            if strip_special and word in (self.pad_token, self.start_token, self.end_token):
                if word == self.end_token:
                    break
                continue
            words.append(word)
        return " ".join(words)

    # -- persistence ------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "word2idx": self._word2idx,
            "special_tokens": {
                "pad": self.pad_token,
                "start": self.start_token,
                "end": self.end_token,
                "unk": self.unk_token,
            },
            "freq": dict(self._freq),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        st = data["special_tokens"]
        vocab = cls(st["pad"], st["start"], st["end"], st["unk"])
        vocab._word2idx = {w: int(i) for w, i in data["word2idx"].items()}
        vocab._idx2word = {int(i): w for w, i in vocab._word2idx.items()}
        vocab._freq = Counter(data.get("freq", {}))
        return vocab
