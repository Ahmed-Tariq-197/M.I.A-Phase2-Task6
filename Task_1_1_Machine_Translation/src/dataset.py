"""PyTorch Dataset / DataLoader construction for the translation pairs."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from .vocabulary import Vocabulary


class TranslationDataset(Dataset):
    def __init__(self, frame, src_vocab: Vocabulary, tgt_vocab: Vocabulary, max_len: int):
        self.src_texts = frame["English"].tolist()
        self.tgt_texts = frame["French"].tolist()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_ids, src_len = self.src_vocab.encode(self.src_texts[idx], self.max_len, add_special=False)
        tgt_ids, _ = self.tgt_vocab.encode(self.tgt_texts[idx], self.max_len, add_special=True)

        # Decoder input = everything except the last token.
        # Decoder target = everything except <start> (teacher forcing).
        tgt_in = tgt_ids[:-1]
        tgt_out = tgt_ids[1:]
        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "src_len": torch.tensor(src_len, dtype=torch.long),
            "tgt_in": torch.tensor(tgt_in, dtype=torch.long),
            "tgt_out": torch.tensor(tgt_out, dtype=torch.long),
            "src_text": self.src_texts[idx],
            "tgt_text": self.tgt_texts[idx],
        }


def make_loader(frame, src_vocab, tgt_vocab, max_len, batch_size, shuffle, num_workers=0):
    dataset = TranslationDataset(frame, src_vocab, tgt_vocab, max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
