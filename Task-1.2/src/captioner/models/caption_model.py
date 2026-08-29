"""Combines the (optionally frozen) CNN encoder with the attention
decoder, and implements inference-time caption generation.

Two generation modes are supported:

* ``generate_greedy`` -- fast, argmax at every step.
* ``generate_beam`` -- slower but higher quality, keeps the top-k
  partial hypotheses at each step (standard beam search with length
  normalization).

During training, features are precomputed/cached (see
``features/extract_features.py``) and the encoder is not invoked at
all -- ``CaptionModel.forward`` accepts features directly. At
inference time on a brand-new user image, ``CaptionModel.encode_image``
runs the CNN once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from captioner.models.decoder import AttentionDecoder
from captioner.models.encoder import EncoderCNN


@dataclass
class GenerationResult:
    token_ids: List[int]
    caption: str
    score: float
    attention_weights: Optional[torch.Tensor] = None  # (T, num_pixels)


class CaptionModel(nn.Module):
    def __init__(self, encoder: Optional[EncoderCNN], decoder: AttentionDecoder, vocab):
        super().__init__()
        self.encoder = encoder  # may be None if features are always precomputed
        self.decoder = decoder
        self.vocab = vocab

    def forward(self, features: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor):
        return self.decoder(features, captions, lengths)

    @torch.no_grad()
    def encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("No encoder attached to this model; pass precomputed features instead.")
        self.encoder.eval()
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        return self.encoder(image_tensor)

    @torch.no_grad()
    def generate_greedy(self, features: torch.Tensor, max_len: int = 35) -> GenerationResult:
        self.decoder.eval()
        device = features.device
        state = self.decoder.init_hidden_state(features)
        word_id = torch.tensor([self.vocab.start_id], device=device)

        token_ids: List[int] = []
        attn_maps = []
        log_prob_sum = 0.0

        for _ in range(max_len):
            embedded = self.decoder.embedding(word_id)
            logits, state, alpha = self.decoder.step(embedded, features, state)
            log_probs = F.log_softmax(logits, dim=-1)
            next_id = int(torch.argmax(log_probs, dim=-1).item())
            log_prob_sum += log_probs[0, next_id].item()
            attn_maps.append(alpha.squeeze(0).cpu())

            if next_id == self.vocab.end_id:
                break
            token_ids.append(next_id)
            word_id = torch.tensor([next_id], device=device)

        caption = self.vocab.decode(token_ids, strip_special=True)
        attn_stack = torch.stack(attn_maps, dim=0) if attn_maps else None
        return GenerationResult(token_ids=token_ids, caption=caption, score=log_prob_sum, attention_weights=attn_stack)

    @torch.no_grad()
    def generate_beam(self, features: torch.Tensor, beam_size: int = 3, max_len: int = 35,
                       length_penalty_alpha: float = 0.7) -> GenerationResult:
        """Beam search with length normalization.

        ``features`` must be a single-image batch, shape (1, num_pixels, feature_dim).
        """
        self.decoder.eval()
        device = features.device

        init_state = self.decoder.init_hidden_state(features)
        beams = [{
            "tokens": [self.vocab.start_id],
            "log_prob": 0.0,
            "state": init_state,
            "done": False,
        }]
        completed = []

        for _ in range(max_len):
            candidates = []
            for beam in beams:
                if beam["done"]:
                    completed.append(beam)
                    continue
                last_token = torch.tensor([beam["tokens"][-1]], device=device)
                embedded = self.decoder.embedding(last_token)
                logits, new_state, _ = self.decoder.step(embedded, features, beam["state"])
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (vocab,)

                topk_log_probs, topk_ids = torch.topk(log_probs, beam_size)
                for lp, idx in zip(topk_log_probs.tolist(), topk_ids.tolist()):
                    candidates.append({
                        "tokens": beam["tokens"] + [idx],
                        "log_prob": beam["log_prob"] + lp,
                        "state": new_state,
                        "done": idx == self.vocab.end_id,
                    })

            if not candidates:
                break

            candidates.sort(key=lambda b: b["log_prob"] / (len(b["tokens"]) ** length_penalty_alpha), reverse=True)
            beams = candidates[:beam_size]

            if all(b["done"] for b in beams):
                completed.extend(beams)
                beams = []
                break

        completed.extend(b for b in beams if not b["done"])
        if not completed:
            completed = beams

        def normalized_score(b):
            return b["log_prob"] / (len(b["tokens"]) ** length_penalty_alpha)

        best = max(completed, key=normalized_score)
        token_ids = [t for t in best["tokens"][1:] if t != self.vocab.end_id]
        caption = self.vocab.decode(token_ids, strip_special=True)
        return GenerationResult(token_ids=token_ids, caption=caption, score=normalized_score(best))
