"""Music stem splitters (``AUDIO_STEM_SPLITTER``): one stereo song in, one track per instrument.

``mock`` is a frequency-band stand-in for CI; ``demucs`` and ``chain`` load real models
lazily (``pip install -e ".[music]"``).
"""
from __future__ import annotations

from ..config import Settings
from .base import StemSplitter, mixture_consistency
from .vocabulary import CANONICAL_STEMS, INSTRUMENTAL, STEM_ALIASES, canonical_stem


def create_splitter(settings: Settings) -> StemSplitter | None:
    """The configured splitter, or None when ``AUDIO_STEM_SPLITTER=off``. Nothing is loaded here."""
    kind = settings.stem_splitter
    if kind == "off":
        return None
    if kind == "mock":
        from .mock import MockStemSplitter

        return MockStemSplitter(six_stems=settings.stem_model.endswith("6s"))
    from .demucs import DemucsSplitter

    demucs = DemucsSplitter(settings.stem_model, device=settings.device, shifts=settings.stem_shifts, overlap=settings.stem_overlap, segment=settings.stem_segment)
    if kind == "demucs":
        return demucs
    if kind == "chain":
        from .chain import ChainSplitter

        return ChainSplitter(settings.stem_vocal_model, demucs, model_dir=settings.stem_model_dir)
    raise ValueError(f"unknown stem splitter {kind!r}")


__all__ = ["CANONICAL_STEMS", "INSTRUMENTAL", "STEM_ALIASES", "StemSplitter", "canonical_stem", "create_splitter", "mixture_consistency"]
