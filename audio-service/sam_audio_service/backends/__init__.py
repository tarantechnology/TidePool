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

        return SamAudioBackend(
            model_id=settings.model_id,
            device=settings.device,
            dtype=settings.dtype,
            mask_mode=settings.visual_mask_mode,  # type: ignore[arg-type]  # validated in Settings.validate
            load_span_predictor=settings.load_span_predictor,
            load_rankers=settings.load_rankers,
            load_visual_ranker=settings.load_visual_ranker,
            judge_only=settings.judge_only,
            ode_method=settings.ode_method,
            ode_step_size=settings.ode_step_size,
            frame_size=settings.frame_size,
            ffmpeg=settings.ffmpeg,
            ffprobe=settings.ffprobe,
            ffmpeg_timeout=settings.ffmpeg_timeout,
        )
    raise ValueError(f"unknown backend {settings.backend!r}")


__all__ = ["Backend", "create_backend"]
