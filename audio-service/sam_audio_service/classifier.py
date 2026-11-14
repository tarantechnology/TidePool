"""Content classifier for the model router: "is this upload a song?".

``score(audio)`` returns ``{"music": p, "speech": p, "song": s}`` or ``None`` when no
classifier is configured. ``music``/``speech`` are the AudioSet "Music"/"Speech"
probabilities averaged over the analysed windows (independent sigmoids, they need not sum to
1); ``song`` is the routing score (``song_score``) the router compares with
``AUDIO_ROUTER_MUSIC_THRESHOLD``. Implementations:

* ``off``   - always ``None``; the router then decides from the prompts alone.
* ``mock``  - fixed scores, for tests and CI. Says everything is music unless told otherwise.
* ``panns`` - PANNs Cnn14 AudioSet tagger (default with a real splitter; ``panns-inference``).
* ``ast``   - Audio Spectrogram Transformer fine-tuned on AudioSet
  (``MIT/ast-finetuned-audioset-10-10-0.4593``; ``transformers``).

Both taggers score up to 8 non-redundant 10 s windows spread across the file in one batch;
``torch`` and the model package are imported on first use.

Why "music minus speech" and not "music": speech over a music bed (vlogs, video soundtracks)
scores as high on "Music" as a song does (AST 0.5-0.75 vs 0.45-0.8; PANNs 0.71-0.96 vs
0.73-0.97), so "Music" alone cannot keep a "vocals"/"singing" prompt on a vlog away from the
stem splitter. What separates them is "Speech": low on songs (sung vocals), high when someone
talks. Per window we take ``music - speech``; the file score is the median over windows
(robust to a spoken intro or one odd window). Labelled router set (MUSDB mixes and
instrumentals, ~2.5-4 min songs incl. with a spoken intro, LibriSpeech, TTS speech, ESC-50,
speech over instrumentals/songs at -6..-20 dB, 2-3 min vlogs):

=========  ===========  ====================  =========  ==================
tagger     full songs   speech over music     speech     threshold
=========  ===========  ====================  =========  ==================
PANNs      0.75-0.83    <= 0.34 (median 0.02)  0          0.4
AST        0.40-0.57    <= 0.17 (median 0)     0          0.2
=========  ===========  ====================  =========  ==================
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .config import Settings

AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AUDIOSET_SPEECH, AUDIOSET_MUSIC = 0, 137  # AudioSet ontology indices (class_labels_indices.csv)


class ContentClassifier(ABC):
    name: str = "base"
    is_mock: bool = False
    # True when the classifier also exposes per-window AudioSet probabilities (``class_names``
    # and ``tag``), which ``labeling`` uses to name music stems.
    can_tag: bool = False

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

    def __init__(self, music: float = 1.0, speech: float = 0.0, song: float | None = None):
        self.scores = {"music": music, "speech": speech}
        if song is not None:
            self.scores["song"] = song

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
    """Up to ``count`` windows of ``size`` samples spread evenly across the signal, but no more
    than it takes to cover it (a 25 s clip gets 3, not 8 near-duplicates)."""
    if len(mono) <= size:
        return [mono]
    count = min(count, int(np.ceil(len(mono) / size)))
    starts = np.linspace(0, len(mono) - size, num=count).astype(int)
    return [mono[s : s + size] for s in sorted(set(starts.tolist()))]


def song_score(music: np.ndarray, speech: np.ndarray) -> float:
    """Routing score from per-window AudioSet probabilities: ``median(music - speech)``,
    floored at 0. High only when music is present *and* nobody is talking over it."""
    margin = np.asarray(music, dtype=np.float64) - np.asarray(speech, dtype=np.float64)
    return max(0.0, float(np.median(margin))) if margin.size else 0.0


def decision_score(scores: dict[str, float]) -> float:
    """The number the router compares with ``AUDIO_ROUTER_MUSIC_THRESHOLD``: ``song`` when the
    classifier reports it, else ``music - speech`` from the file-level probabilities."""
    if "song" in scores:
        return float(scores["song"])
    return max(0.0, float(scores["music"]) - float(scores.get("speech", 0.0)))


class _AudioSetTagger(ContentClassifier):
    """Shared windowing and scoring for AudioSet taggers; subclasses load a model and map
    ``(n, rate * window_seconds)`` mono clips to ``(n, 527)`` sigmoid probabilities."""

    rate = 16_000

    def __init__(self, device: str = "auto", window_seconds: float = 10.0, max_windows: int = 8):
        self.requested_device, self.device = device, device
        self.window_seconds, self.max_windows = window_seconds, max_windows
        self.model: Any = None
        self.label_ids = {"music": AUDIOSET_MUSIC, "speech": AUDIOSET_SPEECH}

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:  # pragma: no cover - needs the model
        raise NotImplementedError

    def probs(self, clips: list[np.ndarray]) -> np.ndarray:  # pragma: no cover - needs the model
        raise NotImplementedError

    can_tag = True
    tag_batch = 16  # windows per forward pass when tagging stems

    def class_names(self) -> list[str]:
        """The 527 AudioSet class names, in the order of ``probs`` columns."""
        self.load()
        return self._class_names()

    def _class_names(self) -> list[str]:  # pragma: no cover - needs the model
        raise NotImplementedError

    def tag(self, clips: list[np.ndarray], sample_rate: int) -> np.ndarray:
        """``(len(clips), 527)`` probabilities for mono clips at ``sample_rate`` (resampled to
        the model rate, padded/cut to ``window_seconds``), in batches of ``tag_batch``."""
        self.load()
        if not clips:
            return np.zeros((0, len(self._class_names())), dtype=np.float32)
        size = int(round(self.window_seconds * sample_rate))
        padded = np.stack([np.pad(np.asarray(c, dtype=np.float32), (0, max(0, size - len(c))))[:size] for c in clips])
        out = []
        for i in range(0, len(padded), self.tag_batch):
            chunk = self._resample_batch(padded[i : i + self.tag_batch], sample_rate)
            out.append(self.probs(list(chunk)))
        return np.concatenate(out)

    def _resample_batch(self, batch: np.ndarray, sample_rate: int) -> np.ndarray:
        """``resample_fft`` over a ``(n, size)`` batch, on the model's device when torch is there
        (CPU FFTs of 48 ten-second windows cost ~0.4 s; on the GPU it is a few ms)."""
        if sample_rate == self.rate:
            return batch
        try:
            import torch
        except ImportError:  # pragma: no cover - the real taggers always have torch
            return np.stack([resample_fft(x, sample_rate, self.rate) for x in batch])
        n_in = batch.shape[1]
        n_out = max(1, int(round(n_in * self.rate / sample_rate)))
        x = torch.from_numpy(batch).to(self.device if str(self.device) != "auto" else "cpu")
        spectrum = torch.fft.rfft(x.double(), dim=-1)[:, : n_out // 2 + 1]
        return (torch.fft.irfft(spectrum, n=n_out, dim=-1) * (n_out / n_in)).float().cpu().numpy()

    def score(self, samples: np.ndarray, sample_rate: int) -> dict[str, float]:
        self.load()
        mono = resample_fft(samples.mean(axis=0), sample_rate, self.rate)
        probs = self.probs(windows(mono, int(self.window_seconds * self.rate), self.max_windows))
        music, speech = probs[:, self.label_ids["music"]], probs[:, self.label_ids["speech"]]
        return {"music": round(float(music.mean()), 4), "speech": round(float(speech.mean()), 4), "song": round(song_score(music, speech), 4)}

    def unload(self) -> None:
        self.model = None


class AstClassifier(_AudioSetTagger):
    name = "ast"
    rate = 16_000

    def __init__(self, model_id: str = AST_MODEL_ID, device: str = "auto", window_seconds: float = 10.0, max_windows: int = 8):
        super().__init__(device, window_seconds, max_windows)
        self.model_id = model_id
        self.extractor: Any = None

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

    def _class_names(self) -> list[str]:
        id2label = self.model.config.id2label
        return [id2label[i] for i in range(len(id2label))]

    def probs(self, clips: list[np.ndarray]) -> np.ndarray:
        import torch

        inputs = self.extractor(clips, sampling_rate=self.rate, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            return torch.sigmoid(self.model(**inputs).logits).float().cpu().numpy()

    def unload(self) -> None:
        self.model = self.extractor = None


class PannsClassifier(_AudioSetTagger):
    """PANNs Cnn14 (AudioSet mAP 0.431, ~80M parameters, 32 kHz) from ``panns-inference``.

    The model is built directly (not through ``AudioTagging``, which wraps it in DataParallel).
    ``checkpoint`` defaults to panns-inference's ``~/panns_data/Cnn14_mAP=0.431.pth``; note that
    importing ``panns_inference`` needs ``~/panns_data/class_labels_indices.csv`` (it shells
    out to wget for it when missing)."""

    name = "panns"
    rate = 32_000
    model_id = "Cnn14_mAP=0.431"
    url = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"

    def __init__(self, checkpoint: str | None = None, device: str = "auto", window_seconds: float = 10.0, max_windows: int = 8):
        super().__init__(device, window_seconds, max_windows)
        self.checkpoint = checkpoint
        self._names: list[str] | None = None

    def info(self) -> dict[str, Any]:
        return {**super().info(), "model_id": self.model_id, "device": self.device}

    def load(self) -> None:
        if self.loaded:
            return
        import os
        from pathlib import Path

        import torch
        from panns_inference.models import Cnn14

        from .stems.demucs import resolve_device

        path = Path(self.checkpoint or Path.home() / "panns_data" / "Cnn14_mAP=0.431.pth")
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(self.url, str(path))
        self.device = resolve_device(self.requested_device)
        model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        model.load_state_dict(torch.load(os.fspath(path), map_location="cpu")["model"])
        self.model = model.eval().to(self.device)

    def _class_names(self) -> list[str]:
        if self._names is None:
            import csv
            from pathlib import Path

            path = Path.home() / "panns_data" / "class_labels_indices.csv"
            if path.exists():
                with path.open(newline="") as f:
                    self._names = [row[2] for row in list(csv.reader(f))[1:]]
            else:  # panns_inference fetches the csv on import
                from panns_inference.config import labels

                self._names = list(labels)
        return self._names

    def probs(self, clips: list[np.ndarray]) -> np.ndarray:
        import torch

        size = int(self.window_seconds * self.rate)
        batch = np.stack([np.pad(c, (0, max(0, size - len(c))))[:size] for c in clips]).astype(np.float32)
        with torch.inference_mode():
            return self.model(torch.from_numpy(batch).to(self.device), None)["clipwise_output"].float().cpu().numpy()


def create_classifier(settings: Settings) -> ContentClassifier:
    if settings.router_classifier == "mock":
        return MockClassifier()
    if settings.router_classifier == "ast":
        return AstClassifier(device=settings.device)
    if settings.router_classifier == "panns":
        return PannsClassifier(checkpoint=settings.router_panns_checkpoint, device=settings.device)
    return NoClassifier()
