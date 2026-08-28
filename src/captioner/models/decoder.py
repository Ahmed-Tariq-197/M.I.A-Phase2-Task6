"""Attention-based recurrent decoder (Show, Attend and Tell style).

At every decoding step the decoder computes a soft (Bahdanau-style)
attention distribution over the encoder's spatial feature map,
conditioned on the current hidden state, then feeds the resulting
context vector -- concatenated with the previous word embedding --
into an LSTM or GRU cell to predict the next word.

This is a strictly stronger design than pooling the CNN output into a
single global vector: the model can ground each generated word in a
different image region (e.g. "dog" -> left half, "frisbee" -> upper
right), which is both more accurate and considerably more
interpretable (attention maps can be visualized).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, attention_dim)
        self.hidden_proj = nn.Linear(hidden_dim, attention_dim)
        self.energy_proj = nn.Linear(attention_dim, 1)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor):
        """features: (B, num_pixels, feature_dim); hidden: (B, hidden_dim)
        Returns: context (B, feature_dim), attention_weights (B, num_pixels)
        """
        proj_features = self.feature_proj(features)          # (B, P, A)
        proj_hidden = self.hidden_proj(hidden).unsqueeze(1)   # (B, 1, A)
        energy = self.energy_proj(torch.tanh(proj_features + proj_hidden)).squeeze(-1)  # (B, P)
        alpha = F.softmax(energy, dim=1)                      # (B, P)
        context = (features * alpha.unsqueeze(-1)).sum(dim=1)  # (B, feature_dim)
        return context, alpha


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feature_dim: int = 2048,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        attention_dim: int = 256,
        rnn_type: str = "lstm",
        num_layers: int = 1,
        dropout: float = 0.5,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.rnn_type = rnn_type.lower()
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(feature_dim, hidden_dim, attention_dim)
        self.dropout = nn.Dropout(dropout)

        # Learned linear maps to initialize the RNN hidden (and cell, for
        # LSTM) state from the mean-pooled image features -- the standard
        # "show and tell" style initialization.
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)

        # A learned gate on the context vector, as in Show/Attend/Tell,
        # which lets the model decide how much visual signal to trust
        # at each step (helpful for function words like "the"/"a").
        self.f_beta = nn.Linear(hidden_dim, feature_dim)

        rnn_cell = nn.LSTMCell if self.rnn_type == "lstm" else nn.GRUCell
        self.rnn_cell = rnn_cell(embed_dim + feature_dim, hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def init_hidden_state(self, features: torch.Tensor):
        mean_features = features.mean(dim=1)  # (B, feature_dim)
        h = torch.tanh(self.init_h(mean_features))
        if self.rnn_type == "lstm":
            c = torch.tanh(self.init_c(mean_features))
            return h, c
        return h

    def step(self, embedded_word: torch.Tensor, features: torch.Tensor, state):
        """Single decoding step. Returns (logits, new_state, attention_weights)."""
        if self.rnn_type == "lstm":
            h, c = state
        else:
            h = state

        context, alpha = self.attention(features, h)
        gate = torch.sigmoid(self.f_beta(h))
        context = gate * context

        rnn_input = torch.cat([embedded_word, context], dim=1)
        if self.rnn_type == "lstm":
            h, c = self.rnn_cell(rnn_input, (h, c))
            new_state = (h, c)
        else:
            h = self.rnn_cell(rnn_input, h)
            new_state = h

        logits = self.output_layer(self.dropout(h))
        return logits, new_state, alpha

    def forward(self, features: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor):
        """Teacher-forced forward pass over a whole batch of captions.

        features: (B, num_pixels, feature_dim)
        captions: (B, T) token ids, including <start> ... <end>
        lengths:  (B,) true (non-padded) lengths of each caption

        Returns logits (B, T-1, vocab_size) predicting tokens[1:] from
        tokens[:-1], plus the stacked attention weights (for optional
        regularization / visualization).
        """
        batch_size, max_len = captions.shape
        device = captions.device

        state = self.init_hidden_state(features)
        embeddings = self.embedding(captions)  # (B, T, E)

        decode_lengths = (lengths - 1).clamp(min=0)  # predicting T-1 steps
        max_decode_len = int(decode_lengths.max().item())

        all_logits = features.new_zeros(batch_size, max_decode_len, self.vocab_size)
        all_alphas = features.new_zeros(batch_size, max_decode_len, features.size(1))

        for t in range(max_decode_len):
            active = decode_lengths > t
            if active.sum() == 0:
                break
            logits, state, alpha = self.step(embeddings[:, t, :], features, state)
            all_logits[:, t, :] = logits
            all_alphas[:, t, :] = alpha

        return all_logits, all_alphas, decode_lengths
