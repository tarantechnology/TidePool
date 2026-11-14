"""Content classifier for the model router: "does this upload sound like music?".

``score(audio)`` returns ``{"music": p, "speech": p}`` (independent AudioSet probabilities,
they need not sum to 1) or ``None`` when no classifier is configured. Implementations:

* ``off``  - always ``None``; the router then decides from the prompts alone.
* ``mock`` - fixed scores, for tests and CI. Says everything is music unless told otherwise.
* ``ast``  - Audio Spectrogram Transformer fine-tuned on AudioSet
  (``MIT/ast-finetuned-audioset-10-10-0.4593``, ~87M parameters) on a few 10 s windows
  spread across the file, averaged. ``transformers``/``torch`` are imported on first use.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .config import Settings

AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


class ContentClassifier(ABC):
    name: str = "base"
    is_mock: bool = False

    @abstractmethod
    def score(self, samples: np.ndarray, sample_rate: int) -> dict[str, float] | None:
        """``samples`` is float32 ``(channels, n)``."""

    @property
    def loaded(self) -> bool:
        return True

    def info(self) -> dict[str, Any]:
        return {"classifier": self.name, "is_mock": self.is_mock, "loaded": self.loaded}

    def unload(self) -> None:  # pragma: no cover - optional
        return None


class NoClassifier(ContentClassifier):
    name = "off"

    def score(self, samples: np.ndarray, sample_rate: int) -> None:
        return None


class MockClassifier(ContentClassifier):
    name = "mock"
    is_mock = True

    def __init__(self, music: float = 1.0, speech: float = 0.0):
        self.scores = {"music": music, "speech": speech}

    def score(self, samples: np.ndarray, sample_rate: int) -> dict[str, float]:
        return dict(self.scores)


def resample_fft(x: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Band-limited resampling of a 1-D signal by spectrum truncation (no scipy needed)."""
    if source_rate == target_rate:
        return x
    n_out = max(1, int(round(len(x) * target_rate / source_rate)))
    spectrum = np.fft.rfft(x)[: n_out // 2 + 1]
    return (np.fft.irfft(spectrum, n=n_out) * (n_out / len(x))).astype(np.float32)


def windows(mono: np.ndarray, size: int, count: int) -> list[np.ndarray]:
    """Up to ``count`` windows of ``size`` samples spread evenly across the signal."""
    if len(mono) <= size:
        return [mono]
    starts = np.linspace(0, len(mono) - size, num=count).astype(int)
    return [mono[s : s + size] for s in sorted(set(starts.tolist()))]


class AstClassifier(ContentClassifier):
    name = "ast"
    rate = 16_000

    def __init__(self, model_id: str = AST_MODEL_ID, device: str = "auto", window_seconds: float = 10.0, max_windows: int = 3):
        self.model_id, self.requested_device, self.device = model_id, device, device
        self.window_seconds, self.max_windows = window_seconds, max_windows
        self.model: Any = None
        self.extractor: Any = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def info(self) -> dict[str, Any]:
        return {**super().info(), "model_id": self.model_id, "device": self.device}

    def load(self) -> None:
        if self.loaded:
            return
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        from .stems.demucs import resolve_device

        self.device = resolve_device(self.requested_device)
        self.extractor = ASTFeatureExtractor.from_pretrained(self.model_id)
        self.model = ASTForAudioClassification.from_pretrained(self.model_id).eval().to(self.device)
        labels = self.model.config.label2id
        self.label_ids = {"music": labels["Music"], "speech": labels["Speech"]}

    def score(self, samples: np.ndarray, sample_rate: int) -> dict[str, float]:
        import torch

        self.load()
        mono = resample_fft(samples.mean(axis=0), sample_rate, self.rate)
        clips = windows(mono, int(self.window_seconds * self.rate), self.max_windows)
        inputs = self.extractor(clips, sampling_rate=self.rate, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            probs = torch.sigmoid(self.model(**inputs).logits).mean(dim=0).cpu()
        return {name: round(float(probs[idx]), 4) for name, idx in self.label_ids.items()}

    def unload(self) -> None:
        self.model = self.extractor = None


def create_classifier(settings: Settings) -> ContentClassifier:
    if settings.router_classifier == "mock":
        return MockClassifier()
    if settings.router_classifier == "ast":
        return AstClassifier(device=settings.device)
    return NoClassifier()
