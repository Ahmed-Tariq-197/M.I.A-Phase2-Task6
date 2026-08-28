"""Image preprocessing transforms shared by feature extraction, training,
evaluation, and inference (so the exact same pipeline is applied every
time an image is fed to the CNN backbone).
"""

from __future__ import annotations

from torchvision import transforms


def build_image_transform(image_size: int = 224, mean=None, std=None, augment: bool = False):
    """Return the torchvision transform pipeline for CNN input.

    ``augment=True`` adds light, caption-safe augmentation (random crop +
    horizontal flip) for training; feature caching / inference always use
    ``augment=False`` for deterministic, reusable features.
    """
    mean = mean or [0.485, 0.456, 0.406]
    std = std or [0.229, 0.224, 0.225]

    if augment:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
