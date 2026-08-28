"""Pretrained CNN encoder (transfer learning backbone).

Supports ResNet-50/101, EfficientNet-B0, and InceptionV3 -- any of the
three families explicitly allowed by the task spec. The backbone's
classification head is discarded; we keep the last convolutional
feature map (a spatial grid, e.g. 7x7x2048 for ResNet-50 at 224x224)
so the decoder's attention mechanism can attend to different image
regions per generated word, rather than compressing the whole image
into one static vector.

The backbone is frozen by default (``fine_tune_encoder: false`` in
config) -- this is what makes feature caching possible/worthwhile: a
frozen encoder produces the same output for a given image every epoch,
so we can run it once and reuse the result.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


_BACKBONE_REGISTRY = {
    "resnet50": {
        "builder": lambda pretrained: tv_models.resnet50(
            weights=tv_models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        ),
        "feature_dim": 2048,
        "spatial_size": 7,
        "cut_layers": ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"],
    },
    "resnet101": {
        "builder": lambda pretrained: tv_models.resnet101(
            weights=tv_models.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
        ),
        "feature_dim": 2048,
        "spatial_size": 7,
        "cut_layers": ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"],
    },
    "efficientnet_b0": {
        "builder": lambda pretrained: tv_models.efficientnet_b0(
            weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        ),
        "feature_dim": 1280,
        "spatial_size": 7,
        "cut_layers": None,  # handled specially below (use .features)
    },
    "inception_v3": {
        "builder": lambda pretrained: tv_models.inception_v3(
            weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
            aux_logits=True,
        ),
        "feature_dim": 2048,
        "spatial_size": 8,
        "cut_layers": None,  # handled specially below
    },
}


class EncoderCNN(nn.Module):
    """Wraps a torchvision backbone and exposes spatial feature maps."""

    def __init__(self, backbone: str = "resnet50", pretrained: bool = True, fine_tune: bool = False):
        super().__init__()
        if backbone not in _BACKBONE_REGISTRY:
            raise ValueError(f"Unknown backbone '{backbone}'. Choose from {list(_BACKBONE_REGISTRY)}.")
        self.backbone_name = backbone
        spec = _BACKBONE_REGISTRY[backbone]
        self.feature_dim = spec["feature_dim"]
        self.spatial_size = spec["spatial_size"]

        model = spec["builder"](pretrained)

        if backbone.startswith("resnet"):
            layers = [getattr(model, name) for name in spec["cut_layers"]]
            self.body = nn.Sequential(*layers)
        elif backbone == "efficientnet_b0":
            self.body = model.features
        elif backbone == "inception_v3":
            model.aux_logits = False
            model.AuxLogits = None
            self.body = model
        else:  # pragma: no cover - guarded by the ValueError above
            raise ValueError(backbone)

        self.set_fine_tune(fine_tune)

    def set_fine_tune(self, fine_tune: bool) -> None:
        """Freeze/unfreeze backbone weights (transfer learning vs. fine-tuning)."""
        for param in self.body.parameters():
            param.requires_grad = fine_tune

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) -> spatial features: (B, num_pixels, feature_dim)."""
        if self.backbone_name == "inception_v3":
            feats = self.body(images)  # (B, 2048, 8, 8) after removing pooling/fc via forward hooks below
            # torchvision's Inception forward already applies global pooling internally when not
            # accessed via its intermediate layers, so we use its exposed conv stack directly:
            feats = self._inception_features(images)
        else:
            feats = self.body(images)  # (B, C, H, W)

        b, c, h, w = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(b, h * w, c)  # (B, num_pixels, C)
        return feats

    def _inception_features(self, x: torch.Tensor) -> torch.Tensor:
        m = self.body
        x = m.Conv2d_1a_3x3(x)
        x = m.Conv2d_2a_3x3(x)
        x = m.Conv2d_2b_3x3(x)
        x = m.maxpool1(x)
        x = m.Conv2d_3b_1x1(x)
        x = m.Conv2d_4a_3x3(x)
        x = m.maxpool2(x)
        x = m.Mixed_5b(x)
        x = m.Mixed_5c(x)
        x = m.Mixed_5d(x)
        x = m.Mixed_6a(x)
        x = m.Mixed_6b(x)
        x = m.Mixed_6c(x)
        x = m.Mixed_6d(x)
        x = m.Mixed_6e(x)
        x = m.Mixed_7a(x)
        x = m.Mixed_7b(x)
        x = m.Mixed_7c(x)
        return x  # (B, 2048, 8, 8)


def get_encoder(backbone: str, pretrained: bool = True, fine_tune: bool = False) -> EncoderCNN:
    return EncoderCNN(backbone=backbone, pretrained=pretrained, fine_tune=fine_tune)
