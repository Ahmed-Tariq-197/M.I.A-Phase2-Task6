"""Single-image caption generation for a brand-new (uncached) image --
what the FastAPI and Gradio front ends call when a user uploads a
photo. Unlike training/evaluation, this path *does* run the CNN
encoder live (there is no cached feature for an image nobody has seen
before).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from captioner.config import ConfigDict, load_config, resolve_path
from captioner.data.transforms import build_image_transform
from captioner.data.vocabulary import Vocabulary
from captioner.models.caption_model import CaptionModel, GenerationResult
from captioner.models.decoder import AttentionDecoder
from captioner.models.encoder import get_encoder

logger = logging.getLogger(__name__)


class CaptionPredictor:
    """Loads the encoder + trained decoder + vocabulary once and exposes
    a simple ``predict(image)`` method. Designed to be instantiated a
    single time per process (e.g. as a FastAPI startup dependency).
    """

    def __init__(self, cfg: Optional[ConfigDict] = None, checkpoint_path: Optional[str | Path] = None,
                 device: str = "auto"):
        self.cfg = cfg or load_config()
        self.device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available()) else ("cpu" if device == "auto" else device)
        )

        vocab_path = resolve_path(self.cfg.vocab.vocab_path)
        if not vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found at {vocab_path}. Run scripts/build_vocab.py first."
            )
        self.vocab = Vocabulary.load(vocab_path)

        ckpt_path = resolve_path(checkpoint_path or self.cfg.serving.default_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {ckpt_path}. Train a model first with scripts/train.py, "
                f"or download a released checkpoint (see README) into that path."
            )

        self.encoder = get_encoder(
            backbone=self.cfg.features.backbone, pretrained=self.cfg.features.pretrained, fine_tune=False
        ).to(self.device).eval()

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.decoder = AttentionDecoder(
            vocab_size=len(self.vocab),
            feature_dim=self.cfg.features.feature_dim,
            embed_dim=self.cfg.model.embed_dim,
            hidden_dim=self.cfg.model.hidden_dim,
            attention_dim=self.cfg.model.attention_dim,
            rnn_type=self.cfg.model.decoder_type,
            num_layers=self.cfg.model.num_layers,
            dropout=self.cfg.model.dropout,
            pad_id=self.vocab.pad_id,
        )
        self.decoder.load_state_dict(ckpt["model_state_dict"])
        self.decoder.to(self.device).eval()

        self.model = CaptionModel(encoder=self.encoder, decoder=self.decoder, vocab=self.vocab)
        self.transform = build_image_transform(
            image_size=self.cfg.image.size, mean=self.cfg.image.mean, std=self.cfg.image.std, augment=False
        )
        logger.info("CaptionPredictor ready (device=%s, vocab=%d words).", self.device, len(self.vocab))

    @torch.no_grad()
    def predict(self, image: Image.Image, use_beam: bool = True, beam_size: Optional[int] = None) -> GenerationResult:
        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        features = self.encoder(tensor)
        if use_beam:
            return self.model.generate_beam(
                features, beam_size=beam_size or self.cfg.evaluation.beam_size,
                max_len=self.cfg.evaluation.max_gen_len,
            )
        return self.model.generate_greedy(features, max_len=self.cfg.evaluation.max_gen_len)

    def predict_path(self, image_path: str | Path, **kwargs) -> GenerationResult:
        with Image.open(image_path) as img:
            return self.predict(img, **kwargs)
