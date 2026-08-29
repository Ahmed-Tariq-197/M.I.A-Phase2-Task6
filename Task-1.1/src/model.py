"""
Sequence-to-sequence architecture: bidirectional LSTM encoder, Luong
(dot-product) attention, LSTM decoder.

This is the same architecture as the original workshop notebook. The only
addition is that ``Encoder`` and ``Decoder`` can optionally be initialised
from a pretrained embedding matrix (produced by ``src/embeddings.py``)
instead of the default random initialisation -- everything downstream
(attention, decoding, training loop) is unchanged.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_embedding(
    vocab_size: int,
    embed_dim: int,
    pad_idx: int,
    pretrained: Optional[np.ndarray] = None,
    freeze: bool = False,
) -> nn.Embedding:
    if pretrained is None:
        return nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
    weight = torch.tensor(pretrained, dtype=torch.float32)
    return nn.Embedding.from_pretrained(weight, freeze=freeze, padding_idx=pad_idx)


class Encoder(nn.Module):
    """Bidirectional LSTM encoder.

    Input : (batch, src_len) token ids
    Output: encoder_outputs (batch, src_len, 2*hidden)
            hidden, cell    (1, batch, 2*hidden)  -- merged directions
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        pad_idx: int,
        dropout: float = 0.2,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding = _make_embedding(vocab_size, embed_dim, pad_idx, pretrained_embeddings, freeze_embeddings)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat([hidden[0], hidden[1]], dim=-1).unsqueeze(0)
        cell = torch.cat([cell[0], cell[1]], dim=-1).unsqueeze(0)
        return outputs, hidden, cell


class LuongAttention(nn.Module):
    """Dot-product attention with optional padding mask."""

    def forward(self, decoder_outputs, encoder_outputs, src_mask=None):
        scores = torch.bmm(decoder_outputs, encoder_outputs.transpose(1, 2))
        if src_mask is not None:
            scores = scores.masked_fill(~src_mask.unsqueeze(1), -1e9)
        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, encoder_outputs)
        return context, weights


class Decoder(nn.Module):
    """LSTM decoder + attention + vocabulary projection."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        pad_idx: int,
        dropout: float = 0.2,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze_embeddings: bool = False,
    ):
        super().__init__()
        self.embedding = _make_embedding(vocab_size, embed_dim, pad_idx, pretrained_embeddings, freeze_embeddings)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True)
        self.attention = LuongAttention()
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_in, hidden, cell, encoder_outputs, src_mask=None):
        embedded = self.dropout(self.embedding(tgt_in))
        dec_out, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        context, attn = self.attention(dec_out, encoder_outputs, src_mask)
        combined = torch.cat([dec_out, context], dim=-1)
        logits = self.fc(combined)
        return logits, hidden, cell, attn


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, pad_idx: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def create_src_mask(self, src):
        return src != self.pad_idx

    def forward(self, src, tgt_in):
        encoder_outputs, hidden, cell = self.encoder(src)
        src_mask = self.create_src_mask(src)
        logits, _, _, attn = self.decoder(tgt_in, hidden, cell, encoder_outputs, src_mask)
        return logits, attn
