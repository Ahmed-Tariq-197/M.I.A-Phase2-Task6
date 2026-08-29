"""Greedy-decoding inference for the trained Seq2Seq model."""
from __future__ import annotations

from typing import List

import torch

from .text_cleaning import clean_english
from .vocabulary import Vocabulary


@torch.no_grad()
def translate_sentence(
    model,
    sentence: str,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    max_seq_len: int,
    max_len: int | None = None,
    device: str = "cpu",
) -> str:
    """Greedy-decode one English sentence into French."""
    model.eval()
    max_len = max_len or max_seq_len

    cleaned = clean_english(sentence)
    src_ids, src_len = src_vocab.encode(cleaned, max_seq_len, add_special=False)
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    encoder_outputs, hidden, cell = model.encoder(src)
    src_mask = model.create_src_mask(src)

    token = torch.tensor([[tgt_vocab.sos_idx]], device=device)
    output_ids: List[int] = []

    for _ in range(max_len):
        logits, hidden, cell, _ = model.decoder(token, hidden, cell, encoder_outputs, src_mask)
        next_id = int(logits[0, -1].argmax())
        if next_id == tgt_vocab.eos_idx:
            break
        output_ids.append(next_id)
        token = torch.tensor([[next_id]], device=device)

    return tgt_vocab.decode(output_ids)


def translate_batch(
    model, sentences: List[str], src_vocab, tgt_vocab, max_seq_len: int, device: str = "cpu"
) -> List[str]:
    return [translate_sentence(model, s, src_vocab, tgt_vocab, max_seq_len, device=device) for s in sentences]
