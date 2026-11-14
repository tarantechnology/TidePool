"""Music stem splitters (``AUDIO_STEM_SPLITTER``): one stereo song in, one track per instrument.

``mock`` is a frequency-band stand-in for CI; ``demucs``, ``chain`` and ``roformer`` load real models
lazily (``pip install -e ".[music]"``).
"""
from __future__ import annotations

from pathlib import Path

from ..config import Settings
from .base import StemSplitter, mixture_consistency
from .vocabulary import CANONICAL_STEMS, INSTRUMENTAL, STEM_ALIASES, canonical_stem


def vocal_checkpoint(model: str, model_dir: str | None) -> str:
    """The vocal checkpoint path: ``model`` itself when it is a path, else inside ``model_dir``
    (audio-separator's download directory, where the .ckpt and its .yaml live)."""
    path = Path(model)
    if path.is_absolute() or not model_dir:
        return str(path)
    return str(Path(model_dir) / path)


def create_splitter(settings: Settings) -> StemSplitter | None:
    """The configured splitter, or None when ``AUDIO_STEM_SPLITTER=off``. Nothing is loaded here."""
    kind = settings.stem_splitter
    if kind == "off":
        return None
    if kind == "mock":
        from .mock import MockStemSplitter

        return MockStemSplitter(six_stems=settings.stem_model.endswith("6s"))
    if kind == "roformer":
        from .roformer import RoformerSplitter

        return RoformerSplitter(settings.stem_roformer_ckpt, settings.stem_roformer_config, device=settings.device,
                                overlap=settings.stem_roformer_overlap, fp16=settings.stem_roformer_fp16, batch=settings.stem_roformer_batch)
    from .demucs import DemucsSplitter

    demucs = DemucsSplitter(settings.stem_model, device=settings.device, shifts=settings.stem_shifts, overlap=settings.stem_overlap, segment=settings.stem_segment,
                            batch=settings.stem_demucs_batch)
    if kind == "demucs":
        return demucs
    if kind == "chain":
        from .chain import ChainSplitter

        vocals = None
        if settings.stem_vocal_backend == "native":
            from .roformer import RoformerSplitter

            ckpt = vocal_checkpoint(settings.stem_vocal_model, settings.stem_model_dir)
            config = settings.stem_vocal_config or str(Path(ckpt).with_suffix(".yaml"))
            vocals = RoformerSplitter(ckpt, config, device=settings.device, overlap=settings.stem_vocal_overlap, fp16=settings.stem_vocal_fp16,
                                      batch=settings.stem_vocal_batch, license="MIT (ViperX BS-RoFormer vocal weights; bs_roformer code)")
        return ChainSplitter(settings.stem_vocal_model, demucs, model_dir=settings.stem_model_dir, vocal_overlap=settings.stem_vocal_overlap, vocals=vocals)
    raise ValueError(f"unknown stem splitter {kind!r}")


__all__ = ["CANONICAL_STEMS", "INSTRUMENTAL", "STEM_ALIASES", "StemSplitter", "canonical_stem", "create_splitter", "mixture_consistency"]
