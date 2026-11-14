"""Nested ("Separate the rest") splitters: one stem of an earlier split in, the instruments inside it out.

``options.mode = "music_substems"`` (Phase 4 of docs/plans/music-stems.md). The parent is the
residual instrument stem ("other": whatever the main split did not name). The engine:

1. picks the instruments: ``options.stems`` (caller wording -> family via ``SUBSTEM_ALIASES``),
   or, when null, the families ``labeling.detect_instruments`` (the router's tagger, per-family
   calibrated presence thresholds, raised by ``AUDIO_SUBSTEM_MIN_SCORE`` when set) marks present
   in the input, strongest first, up to ``AUDIO_SUBSTEM_MAX``, restricted to what the separator
   can produce;
2. separates them in one model run (:meth:`SubstemSeparator.separate`);
3. makes every model output sum to the input (``mixture_consistency``), folds each requested
   instrument quieter than ``AUDIO_SUBSTEM_SILENT_DB`` (relative to the input) into the
   residual, and returns the targets plus exactly one residual, ``Rest``.

Families a separator cannot separate well are not offered at all (better a smaller, honest split):
the Phase 4 benchmark (benchmarks/nested_stems.py, BabySlakh) found only htdemucs_6s's guitar
and piano outputs clearly better than the input among licence-clean models, so ``demucs``
offers ``guitar``, ``piano`` and ``keys`` (keys = electric piano / harpsichord, which the piano
source carries).
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import numpy as np

from ..contracts import normalise_description
from .base import Progress, check_cancel, report

# Canonical sub-instrument labels, in display order.
SUBSTEM_FAMILIES: tuple[str, ...] = ("guitar", "piano", "keys", "organ", "synth", "strings", "brass", "woodwinds")
RESIDUAL_LABEL = "rest"
RESIDUAL_NAME = "Rest"
DISPLAY_NAMES: dict[str, str] = {
    "guitar": "Guitar", "piano": "Piano", "keys": "Keys", "organ": "Organ", "synth": "Synth",
    "strings": "Strings", "brass": "Brass", "woodwinds": "Woodwinds",
}

# Caller wording (after contracts.normalise_description) -> family.
SUBSTEM_ALIASES: dict[str, str] = {
    **{w: "guitar" for w in ("guitar", "guitars", "electric guitar", "acoustic guitar", "rhythm guitar", "lead guitar")},
    **{w: "piano" for w in ("piano", "pianos", "grand piano", "acoustic piano", "upright piano")},
    **{w: "keys" for w in ("keys", "keyboard", "keyboards", "electric piano", "e piano", "e-piano", "rhodes", "wurlitzer", "harpsichord", "clavinet")},
    **{w: "organ" for w in ("organ", "organs", "hammond", "hammond organ", "church organ")},
    **{w: "synth" for w in ("synth", "synths", "synthesizer", "synthesizers", "synthesiser", "pad", "pads", "synth pad", "synth lead", "lead synth")},
    **{w: "strings" for w in ("strings", "string", "string section", "violin", "violins", "viola", "cello", "cellos", "orchestra strings")},
    **{w: "brass" for w in ("brass", "horns", "horn section", "trumpet", "trumpets", "trombone", "french horn")},
    **{w: "woodwinds" for w in ("woodwinds", "woodwind", "winds", "flute", "flutes", "sax", "saxophone", "clarinet", "oboe", "bassoon")},
}


def canonical_substem(text: str) -> str | None:
    """Family for a caller's wording ("Rhodes" -> keys), or None."""
    text = text.strip().lower()
    return SUBSTEM_ALIASES.get(text) or SUBSTEM_ALIASES.get(normalise_description(text))


def pick_detected(instruments: Iterable[Any], families: dict[str, str], limit: int) -> list[tuple[str, str]]:
    """``(family, name)`` of the offered families (``families``: family -> model output) that
    ``labeling.detect_instruments`` marked present, strongest first, at most ``limit``.
    Families sharing a model output (piano, keys) are returned once: the stronger one names the
    track ("Electric piano" rather than "Piano")."""
    ranked = sorted((i for i in instruments if i.present and i.family in families), key=lambda i: (-i.score, SUBSTEM_FAMILIES.index(i.family)))
    picked: list[tuple[str, str]] = []
    outputs: set[str] = set()
    for i in ranked:
        if families[i.family] not in outputs:
            outputs.add(families[i.family])
            picked.append((i.family, i.name or DISPLAY_NAMES[i.family]))
    return picked[:limit]


class SubstemSeparator(ABC):
    """One parent stem in, one array per requested family plus the model's leftovers out."""

    name: str = "base"
    is_mock: bool = False
    sample_rate: int = 48_000
    # family -> model output it comes from; several families may share an output (piano, keys).
    families: dict[str, str] = {}

    @property
    def offered(self) -> tuple[str, ...]:
        return tuple(f for f in SUBSTEM_FAMILIES if f in self.families)

    @abstractmethod
    def load(self) -> None:
        """Load weights. Idempotent."""

    @property
    @abstractmethod
    def loaded(self) -> bool: ...

    @abstractmethod
    def info(self) -> dict[str, Any]: ...

    @abstractmethod
    def split(self, mixture: np.ndarray, progress: Progress | None = None, cancel: threading.Event | None = None) -> dict[str, np.ndarray]:
        """Every model output (float32, the input's shape) for ``mixture`` ``(channels, n)``
        at :attr:`sample_rate`. The outputs together should approximate the input; the
        engine enforces the exact sum."""

    def unload(self) -> None:  # pragma: no cover - optional
        return None


class DemucsSubstemSeparator(SubstemSeparator):
    """htdemucs_6s on the parent stem: its guitar and piano sources are the targets; drums,
    bass, vocals and other (bleed and everything else) belong to the residual."""

    name = "demucs"
    families = {"guitar": "guitar", "piano": "piano", "keys": "piano"}

    def __init__(self, model_name: str = "htdemucs_6s", device: str = "auto", overlap: float = 0.25, batch: int = 1):
        from .demucs import DemucsSplitter

        self.model_name = model_name
        self.splitter = DemucsSplitter(model_name, device=device, shifts=0, overlap=overlap, batch=batch)

    def load(self) -> None:
        self.splitter.load()

    @property
    def loaded(self) -> bool:
        return self.splitter.loaded

    def info(self) -> dict[str, Any]:
        return {**self.splitter.info(), "families": list(self.offered)}

    def split(self, mixture: np.ndarray, progress: Progress | None = None, cancel: threading.Event | None = None) -> dict[str, np.ndarray]:
        return self.splitter.split(mixture, progress=progress, cancel=cancel)

    def unload(self) -> None:
        self.splitter.unload()


class MockSubstemSeparator(SubstemSeparator):
    """MOCK: frequency bands of the input, NOT source separation (CI / local development)."""

    name = "mock"
    is_mock = True
    families = {"guitar": "guitar", "piano": "piano", "keys": "piano"}
    BANDS = (("other", 0.0, 200.0), ("piano", 200.0, 2000.0), ("guitar", 2000.0, 7000.0), ("rest", 7000.0, float("inf")))

    def __init__(self) -> None:
        from .mock import MockStemSplitter

        self._bands = MockStemSplitter()
        self._bands.bands = self.BANDS
        self._bands.stems = tuple(name for name, _, _ in self.BANDS)
        self._loaded = False

    def load(self) -> None:
        self._loaded = True

    @property
    def loaded(self) -> bool:
        return self._loaded

    def info(self) -> dict[str, Any]:
        return {"model_id": "mock-substems", "device": "cpu", "dtype": "float32", "license": "n/a (no weights)",
                "families": list(self.offered), "warning": "mock nested splitter: frequency bands, not source separation"}

    def split(self, mixture: np.ndarray, progress: Progress | None = None, cancel: threading.Event | None = None) -> dict[str, np.ndarray]:
        check_cancel(cancel)
        report(progress, 0.0, "bands")
        return self._bands.split(mixture, progress=progress, cancel=cancel)


def create_substem_separator(settings) -> SubstemSeparator | None:
    """The configured nested splitter (``AUDIO_SUBSTEM_MODEL``), or None when off. Nothing loads here."""
    kind = settings.substem_model
    if kind == "off":
        return None
    if kind == "mock":
        return MockSubstemSeparator()
    if kind == "htdemucs_6s":
        return DemucsSubstemSeparator(kind, device=settings.device, overlap=settings.stem_overlap, batch=settings.stem_demucs_batch)
    raise ValueError(f"unknown AUDIO_SUBSTEM_MODEL {kind!r}")
