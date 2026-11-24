"""Runtime configuration, read from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _hosts(value: str) -> tuple[str, ...]:
    return tuple(h.strip() for h in value.split(",") if h.strip())


@dataclass(frozen=True)
class Settings:
    model_path: str = field(default_factory=lambda: os.getenv("MODEL_PATH", "models/plant-disease-resnet9.pth"))
    allowed_image_hosts: tuple[str, ...] = field(
        default_factory=lambda: _hosts(os.getenv("ALLOWED_IMAGE_HOSTS", "firebasestorage.googleapis.com"))
    )
    max_image_bytes: int = field(default_factory=lambda: int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024))))
    download_timeout: float = field(default_factory=lambda: float(os.getenv("DOWNLOAD_TIMEOUT", "10")))
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "3")))
