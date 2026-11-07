"""Runtime configuration, read once from environment variables.

Every setting is prefixed with ``AUDIO_`` so it cannot collide with Person 4's backend
configuration when both run from the same ``.env``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


@dataclass
class Settings:
    # "mock" is a deterministic DSP stand-in for local development and CI. It is NOT
    # source separation. "sam" loads the real SAM Audio checkpoints.
    backend: str = field(default_factory=lambda: os.getenv("AUDIO_BACKEND", "mock"))
    model_id: str = field(default_factory=lambda: os.getenv("AUDIO_MODEL_ID", "facebook/sam-audio-large"))
    # "auto" picks cuda > mps > cpu.
    device: str = field(default_factory=lambda: os.getenv("AUDIO_DEVICE", "auto"))
    dtype: str = field(default_factory=lambda: os.getenv("AUDIO_DTYPE", "float32"))
    load_on_startup: bool = field(default_factory=lambda: _bool("AUDIO_LOAD_ON_STARTUP", True))

    # Inference defaults (overridable per request through metadata.options).
    predict_spans: bool = field(default_factory=lambda: _bool("AUDIO_PREDICT_SPANS", False))
    reranking_candidates: int = field(default_factory=lambda: _int("AUDIO_RERANKING_CANDIDATES", 1))
    # 0 disables chunking. Long inputs are otherwise processed in overlapping windows.
    chunk_seconds: float = field(default_factory=lambda: _float("AUDIO_CHUNK_SECONDS", 0.0))
    chunk_overlap_seconds: float = field(default_factory=lambda: _float("AUDIO_CHUNK_OVERLAP_SECONDS", 2.0))
    # "duplicate": run mono separation and duplicate to source channel count (1 pass).
    # "per_channel": separate each channel independently (N passes, keeps stereo image).
    stereo_mode: str = field(default_factory=lambda: os.getenv("AUDIO_STEREO_MODE", "duplicate"))

    # Limits.
    max_duration_seconds: float = field(default_factory=lambda: _float("AUDIO_MAX_DURATION_SECONDS", 600.0))
    max_input_bytes: int = field(default_factory=lambda: _int("AUDIO_MAX_INPUT_BYTES", 500 * 1024 * 1024))
    max_prompts: int = field(default_factory=lambda: _int("AUDIO_MAX_PROMPTS", 8))

    # Media references. file:// references must resolve under media_root (a shared volume
    # owned by Person 4). http(s) references must match an allow-listed host.
    media_root: Path | None = field(default_factory=lambda: Path(os.environ["AUDIO_MEDIA_ROOT"]).resolve() if os.getenv("AUDIO_MEDIA_ROOT") else None)
    allowed_media_hosts: tuple[str, ...] = field(default_factory=lambda: tuple(h.strip() for h in os.getenv("AUDIO_ALLOWED_MEDIA_HOSTS", "").split(",") if h.strip()))
    # Where stems are written when a request asks for file references instead of base64.
    output_root: Path = field(default_factory=lambda: Path(os.getenv("AUDIO_OUTPUT_ROOT", "output")).resolve())

    ffmpeg: str = field(default_factory=lambda: os.getenv("AUDIO_FFMPEG", "ffmpeg"))
    ffprobe: str = field(default_factory=lambda: os.getenv("AUDIO_FFPROBE", "ffprobe"))
    ffmpeg_timeout: int = field(default_factory=lambda: _int("AUDIO_FFMPEG_TIMEOUT", 600))

    def validate(self) -> None:
        if self.backend not in {"mock", "sam"}:
            raise ValueError(f"AUDIO_BACKEND must be 'mock' or 'sam', got {self.backend!r}")
        if self.stereo_mode not in {"duplicate", "per_channel"}:
            raise ValueError("AUDIO_STEREO_MODE must be 'duplicate' or 'per_channel'")
        if self.chunk_seconds and self.chunk_seconds <= 2 * self.chunk_overlap_seconds:
            raise ValueError("AUDIO_CHUNK_SECONDS must exceed twice AUDIO_CHUNK_OVERLAP_SECONDS")


def load_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings
