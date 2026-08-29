"""Configuration loading utilities.

All scripts in this project read their settings from a single YAML file
(``configs/config.yaml``) instead of hard-coding paths or hyperparameters.
This keeps the pipeline reproducible: changing the backbone, batch size,
or split ratios means editing one file, not hunting through scripts.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ConfigDict(dict):
    """A dict that also supports attribute access, recursively.

    Lets calling code write ``cfg.training.batch_size`` instead of
    ``cfg["training"]["batch_size"]`` while still behaving like a normal
    dict everywhere else (so it's easy to serialize / log).
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, ConfigDict):
            value = ConfigDict(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return ConfigDict({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(path: str | Path | None = None) -> ConfigDict:
    """Load and return the project configuration as a ``ConfigDict``.

    Resolution order: explicit ``path`` argument > ``CAPTIONER_CONFIG``
    environment variable > ``configs/config.yaml``. The environment
    variable lets deployment targets (Docker, Hugging Face Spaces) point
    at an alternate config without code changes.
    """
    import os

    cfg_path = Path(path) if path is not None else Path(os.environ.get("CAPTIONER_CONFIG", DEFAULT_CONFIG_PATH))
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f)
    return _wrap(raw)


def resolve_path(relative: str | Path) -> Path:
    """Resolve a path from config.yaml (relative to project root) to an absolute Path."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def to_plain_dict(cfg: ConfigDict) -> Dict[str, Any]:
    """Convert a ConfigDict back into plain nested dicts (for JSON dumps, logging, etc.)."""
    return copy.deepcopy(dict(cfg))


if __name__ == "__main__":
    cfg = load_config()
    print(f"Loaded config for project: {cfg.project.name}")
    print(f"Backbone: {cfg.features.backbone}, decoder: {cfg.model.decoder_type}")
