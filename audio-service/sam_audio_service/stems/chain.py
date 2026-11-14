"""Chain splitter: a dedicated vocal model, then Demucs on the instrumental.

RoFormer vocal models (run through ``audio-separator``) score higher on vocals than Demucs;
Demucs then splits the remaining instrumental into drums / bass / other (/ guitar / piano).
Whatever Demucs still calls "vocals" in the instrumental is leakage and goes to ``other``.
Whether this is worth the second model is decided by the Phase 0 benchmark; until then it
is opt-in (``AUDIO_STEM_SPLITTER=chain``).

``audio-separator`` is file based: the mixture goes through a temporary float WAV, and the
separator writes into a private directory created at load time (its output directory is
fixed when the model loads), from which each request's files are read and deleted.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from ..audio_io import decode, encode_wav_float32
from .base import StemSplitter, fit_length
from .demucs import DemucsSplitter


class ChainSplitter(StemSplitter):
    name = "chain"

    def __init__(self, vocal_model: str, demucs: DemucsSplitter, model_dir: str | None = None):
        self.vocal_model = vocal_model
        self.demucs = demucs
        self.model_dir = model_dir
        self.separator: Any = None
        self.output_dir: Path | None = None
        self.stems = demucs.stems

    @property
    def loaded(self) -> bool:
        return self.separator is not None and self.demucs.loaded

    def info(self) -> dict[str, Any]:
        demucs = self.demucs.info()
        return {**demucs, "model_id": f"{self.vocal_model}+{demucs['model_id']}", "vocal_model": self.vocal_model}

    def load(self) -> None:
        if self.separator is None:
            from audio_separator.separator import Separator

            self.output_dir = Path(tempfile.mkdtemp(prefix="sam-audio-chain-out-"))
            kwargs = {"model_file_dir": self.model_dir} if self.model_dir else {}
            separator = Separator(output_dir=str(self.output_dir), output_format="WAV", **kwargs)
            separator.load_model(model_filename=self.vocal_model)
            self.separator = separator
        self.demucs.load()
        self.stems = self.demucs.stems

    def split(self, mixture: np.ndarray) -> dict[str, np.ndarray]:
        if not self.loaded:
            raise RuntimeError("chain splitter is not loaded")
        vocals, instrumental = self._vocals_and_instrumental(mixture)
        stems = self.demucs.split(instrumental)
        stems["other"] = stems["other"] + stems.pop("vocals")
        stems["vocals"] = vocals
        return stems

    def _vocals_and_instrumental(self, mixture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with tempfile.TemporaryDirectory(prefix="sam-audio-chain-") as tmp:
            src = Path(tmp) / "mix.wav"
            src.write_bytes(encode_wav_float32(mixture, self.sample_rate))
            outputs = [self.output_dir / Path(p).name for p in self.separator.separate(str(src))]
            found: dict[str, np.ndarray] = {}
            try:
                for path in outputs:
                    # audio-separator names outputs "<input>_(Vocals)_<model>.wav" / "(Instrumental)".
                    for key in ("instrumental", "vocals"):
                        if f"({key})" in path.name.lower():
                            audio = decode(path=path, max_channels=mixture.shape[0]).samples
                            if audio.shape[0] != mixture.shape[0]:
                                audio = audio.mean(axis=0, keepdims=True) if mixture.shape[0] == 1 else np.repeat(audio, mixture.shape[0], axis=0)
                            found[key] = fit_length(audio, mixture.shape[1])
            finally:
                for path in outputs:
                    path.unlink(missing_ok=True)
            if "vocals" not in found:
                raise RuntimeError(f"vocal model produced no vocals file: {[p.name for p in outputs]}")
        vocals = found["vocals"]
        instrumental = found.get("instrumental", mixture - vocals)
        return vocals.astype(np.float32), instrumental.astype(np.float32)

    def unload(self) -> None:
        self.separator = None
        if self.output_dir is not None:
            shutil.rmtree(self.output_dir, ignore_errors=True)
            self.output_dir = None
        self.demucs.unload()
