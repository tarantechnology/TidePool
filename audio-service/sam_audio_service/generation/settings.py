"""Generate-part settings (``AUDIO_GEN_*``), kept separate from ``config.Settings``."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _b(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw in (None, "") else raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class GenSettings:
    enabled: bool = field(default_factory=lambda: _b("AUDIO_GEN_ENABLED", True))
    samples_dir: str | None = field(default_factory=lambda: os.getenv("AUDIO_GEN_SAMPLES_DIR") or None)  # VSCO-2 CE root with manifest.json
    soundfont: str | None = field(default_factory=lambda: os.getenv("AUDIO_GEN_SOUNDFONT") or None)  # default: system FluidR3_GM.sf2
    beats: str = field(default_factory=lambda: os.getenv("AUDIO_GEN_BEATS", "auto"))  # auto | beat_this | numpy
    transcriber: str = field(default_factory=lambda: os.getenv("AUDIO_GEN_TRANSCRIBER", "auto"))  # auto | basic_pitch | off
    # Experimental engines, OFF by default (listening tests preferred the rule-based arranger).
    amt_model: str = field(default_factory=lambda: os.getenv("AUDIO_GEN_AMT_MODEL", "off"))  # e.g. stanford-crfm/music-medium-800k
    audiogen: str = field(default_factory=lambda: os.getenv("AUDIO_GEN_AUDIOGEN", "off"))  # off | musicgen
    musicgen_model: str = field(default_factory=lambda: os.getenv("AUDIO_GEN_MUSICGEN_MODEL", "facebook/musicgen-melody"))
    # MusicGen weights are CC-BY-NC-4.0: refuse unless the deployment opted into non-commercial use.
    allow_noncommercial: bool = field(default_factory=lambda: _b("AUDIO_GEN_ALLOW_NONCOMMERCIAL", False))
    device: str = field(default_factory=lambda: os.getenv("AUDIO_GEN_DEVICE", "auto"))
    max_seconds: float = field(default_factory=lambda: float(os.getenv("AUDIO_GEN_MAX_SECONDS", "600")))
    audiogen_max_seconds: float = field(default_factory=lambda: float(os.getenv("AUDIO_GEN_AUDIOGEN_MAX_SECONDS", "120")))


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"
