"""Inference backends. ``mock`` is a development stand-in; ``sam`` is the real model."""
from __future__ import annotations

from ..config import Settings
from .base import Backend


def create_backend(settings: Settings) -> Backend:
    if settings.backend == "mock":
        from .mock import MockBackend

        return MockBackend()
    if settings.backend == "sam":
        from .sam_audio import SamAudioBackend

        return SamAudioBackend(model_id=settings.model_id, device=settings.device, dtype=settings.dtype)
    raise ValueError(f"unknown backend {settings.backend!r}")


__all__ = ["Backend", "create_backend"]
