"""Chain splitter: a dedicated vocal model, then Demucs on the instrumental.

RoFormer vocal models (run through ``audio-separator``) score higher on vocals than Demucs;
Demucs then splits the remaining instrumental into drums / bass / other (/ guitar / piano).
Whatever Demucs still calls "vocals" in the instrumental is leakage and goes to ``other``.
Whether this is worth the second model is decided by the Phase 0 benchmark; until then it
is opt-in (``AUDIO_STEM_SPLITTER=chain``).

``audio-separator`` is file based: the mixture goes through a temporary float WAV, and the
separator writes into a private directory created at load time (its output directory is
fixed when the model loads), from which each request's files are read and deleted.

Gain handling matches the benchmark's ``SeparatorRunner``: audio-separator peak-normalises
the input and every output to ``normalization_threshold`` (default 0.9) and never undoes it,
so it is set to 1.0 and the input is scaled to a peak of at most ``HEADROOM`` (undone on the
vocals). The instrumental is ``mixture - vocals``; the library's own instrumental file is
ignored, so production computes exactly what was measured.
"""
from __future__ import annotations

import inspect
import logging
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..audio_io import decode, encode_wav_float32
from .base import Progress, StemSplitter, check_cancel, fit_length, report
from .demucs import DemucsSplitter

log = logging.getLogger("sam_audio_service")

HEADROOM = 0.5
SEPARATOR_RATE = 44_100


def separator_kwargs(separator_cls: type, output_dir: str, model_dir: str | None, vocal_overlap: int) -> dict[str, Any]:
    """Constructor arguments for this audio-separator version (older ones lack some)."""
    params = inspect.signature(separator_cls).parameters
    wanted: dict[str, Any] = {"output_dir": output_dir, "output_format": "WAV", "normalization_threshold": 1.0,
                              "sample_rate": SEPARATOR_RATE, "use_soundfile": True}
    if model_dir:
        wanted["model_file_dir"] = model_dir
    if "mdxc_params" in params:
        default = params["mdxc_params"].default
        wanted["mdxc_params"] = {**(default if isinstance(default, dict) else {}), "overlap": vocal_overlap}
    kwargs = {k: v for k, v in wanted.items() if k in params}
    if "normalization_threshold" not in kwargs:
        log.warning("this audio-separator cannot disable peak normalisation; vocal gain will be wrong")
    return kwargs


class ChainSplitter(StemSplitter):
    name = "chain"

    def __init__(self, vocal_model: str, demucs: DemucsSplitter, model_dir: str | None = None, vocal_overlap: int = 8):
        self.vocal_model = vocal_model
        self.demucs = demucs
        self.model_dir = model_dir
        self.vocal_overlap = vocal_overlap
        self.separator: Any = None
        self.output_dir: Path | None = None
        self.stems = demucs.stems

    @property
    def loaded(self) -> bool:
        return self.separator is not None and self.demucs.loaded

    def info(self) -> dict[str, Any]:
        demucs = self.demucs.info()
        return {**demucs, "model_id": f"{self.vocal_model}+{demucs['model_id']}", "vocal_model": self.vocal_model,
                "vocal_overlap": self.vocal_overlap, "license": "MIT (ViperX BS-RoFormer vocals; Demucs htdemucs* instruments)"}

    def load(self) -> None:
        if self.separator is None:
            from audio_separator.separator import Separator

            self.output_dir = Path(tempfile.mkdtemp(prefix="sam-audio-chain-out-"))
            separator = Separator(**separator_kwargs(Separator, str(self.output_dir), self.model_dir, self.vocal_overlap))
            separator.load_model(model_filename=self.vocal_model)
            self.separator = separator
        self.demucs.load()
        self.stems = self.demucs.stems

    def split(self, mixture: np.ndarray, progress: Progress | None = None, cancel: threading.Event | None = None) -> dict[str, np.ndarray]:
        if not self.loaded:
            raise RuntimeError("chain splitter is not loaded")
        check_cancel(cancel)
        report(progress, 0.0, "vocals")
        vocals = self._vocals(mixture)
        check_cancel(cancel)
        report(progress, 0.5, "instruments")
        stems = self.demucs.split((mixture - vocals).astype(np.float32), cancel=cancel)
        stems["other"] = stems["other"] + stems.pop("vocals")
        stems["vocals"] = vocals
        report(progress, 1.0, "done")
        return stems

    def _vocals(self, mixture: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(mixture))) if mixture.size else 0.0
        gain = HEADROOM / peak if peak > HEADROOM else 1.0
        with tempfile.TemporaryDirectory(prefix="sam-audio-chain-") as tmp:
            src = Path(tmp) / "mix.wav"
            src.write_bytes(encode_wav_float32(mixture * gain, self.sample_rate))
            outputs = [self.output_dir / Path(p).name for p in self.separator.separate(str(src))]
            try:
                # audio-separator names outputs "<input>_(Vocals)_<model>.wav".
                path = next((p for p in outputs if "(vocals)" in p.name.lower()), None)
                if path is None:
                    raise RuntimeError(f"vocal model produced no vocals file: {[p.name for p in outputs]}")
                audio = decode(path=path, max_channels=mixture.shape[0]).samples
            finally:
                for p in outputs:
                    p.unlink(missing_ok=True)
        if audio.shape[0] != mixture.shape[0]:
            audio = audio.mean(axis=0, keepdims=True) if mixture.shape[0] == 1 else np.repeat(audio, mixture.shape[0], axis=0)
        return (fit_length(audio, mixture.shape[1]) / gain).astype(np.float32)

    def unload(self) -> None:
        self.separator = None
        if self.output_dir is not None:
            shutil.rmtree(self.output_dir, ignore_errors=True)
            self.output_dir = None
        self.demucs.unload()
