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
from collections.abc import Sequence
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


CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"


class ClapTagger:
    """Zero-shot instrument/voice tagger: LAION-CLAP (HTSAT-large audio tower + RoBERTa text tower,
    ``laion/larger_clap_music_and_speech``, Apache-2.0, ~190M parameters, 48 kHz, 10 s windows).

    Not a router classifier (no ``score``); ``labeling`` uses it to name stems when
    ``AUDIO_STEM_LABEL_MODEL=clap``. ``prompts`` maps a label id to one or more text prompts;
    ``tag`` returns, per clip, the cosine similarity between the clip's audio embedding and each
    label's (prompt- and template-averaged) text embedding, in ``class_names()`` order.

    The log-mel front end runs on the model's device (the transformers ``ClapFeatureExtractor``
    is numpy, ~25 ms a window on the pod's CPUs; on the GPU 48 windows take a few ms). It matches
    the extractor's non-fusion path (Hann 1024 / hop 480, 64 Slaney mel bins 50 Hz-14 kHz, dB)
    to ~1e-4 dB. Note: the music-only ``laion/larger_clap_music`` checkpoint collapses to one
    class on our instrument prompts (11% top-1 on clean BabySlakh sources), so it is not used.
    """

    name = "clap"
    rate = 48_000
    templates = ("{}", "the sound of {}", "{} in a song")
    can_tag = True
    zero_shot = True  # tag() returns cosine similarities per label id, not AudioSet probabilities
    is_mock = False

    def __init__(self, prompts: dict[str, Sequence[str]], model_id: str = CLAP_MODEL_ID, device: str = "auto", window_seconds: float = 10.0, batch: int = 16):
        self.prompts = {k: tuple(v) for k, v in prompts.items()}
        self.model_id, self.requested_device, self.device = model_id, device, device
        self.window_seconds, self.batch = window_seconds, batch
        self.model: Any = None
        self._text: Any = None  # (n_labels, d) normalised text embeddings on the device
        self._mel: Any = None
        self._window: Any = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def info(self) -> dict[str, Any]:
        return {"tagger": self.name, "model_id": self.model_id, "device": self.device, "loaded": self.loaded, "labels": len(self.prompts)}

    def load(self) -> None:  # pragma: no cover - needs the model
        if self.loaded:
            return
        import torch
        from transformers import ClapModel, ClapProcessor
        from transformers.audio_utils import mel_filter_bank

        from .stems.demucs import resolve_device

        self.device = resolve_device(self.requested_device)
        model = ClapModel.from_pretrained(self.model_id).eval().to(self.device)
        tokenizer = ClapProcessor.from_pretrained(self.model_id).tokenizer
        rows = []
        with torch.inference_mode():
            for label, prompts in self.prompts.items():
                texts = [t.format(p) for t in self.templates for p in prompts]
                tokens = tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
                e = torch.nn.functional.normalize(_pooled(model.get_text_features(**tokens)), dim=-1).mean(0)
                rows.append(torch.nn.functional.normalize(e, dim=-1))
        self._text = torch.stack(rows)
        fb = mel_filter_bank(num_frequency_bins=513, num_mel_filters=64, min_frequency=50, max_frequency=14000, sampling_rate=self.rate, norm="slaney", mel_scale="slaney")
        self._mel = torch.from_numpy(np.asarray(fb, dtype=np.float32)).to(self.device)
        self._window = torch.hann_window(1024, periodic=True, device=self.device)
        self.model = model

    def class_names(self) -> list[str]:
        return list(self.prompts)

    def tag(self, clips: list[np.ndarray], sample_rate: int) -> np.ndarray:  # pragma: no cover - needs the model
        """``(len(clips), n_labels)`` cosine similarities for mono clips at ``sample_rate``
        (resampled to 48 kHz, zero-padded/cut to ``window_seconds``)."""
        self.load()
        if not clips:
            return np.zeros((0, len(self.prompts)), dtype=np.float32)
        import torch

        size = int(round(self.window_seconds * sample_rate))
        batch = np.stack([np.pad(np.asarray(c, dtype=np.float32), (0, max(0, size - len(c))))[:size] for c in clips])
        out = []
        with torch.inference_mode():
            x = torch.from_numpy(batch).to(self.device)
            if sample_rate != self.rate:  # band-limited FFT resampling, as _AudioSetTagger does
                n_out = int(round(size * self.rate / sample_rate))
                spectrum = torch.fft.rfft(x.double(), dim=-1)[:, : n_out // 2 + 1]
                x = (torch.fft.irfft(spectrum, n=n_out, dim=-1) * (n_out / size)).float()
            x = x[:, : int(self.window_seconds * self.rate)]
            for i in range(0, len(x), self.batch):
                chunk = x[i : i + self.batch]
                spec = torch.stft(chunk, 1024, 480, window=self._window, center=True, pad_mode="reflect", return_complex=True).abs() ** 2
                mel = 10.0 * torch.log10(torch.clamp(spec.transpose(1, 2) @ self._mel, min=1e-10))  # (n, frames, 64)
                feats = mel[:, None]
                longer = torch.zeros((len(chunk), 1), dtype=torch.bool, device=self.device)
                emb = torch.nn.functional.normalize(_pooled(self.model.get_audio_features(input_features=feats, is_longer=longer)), dim=-1)
                out.append((emb @ self._text.T).float().cpu().numpy())
        return np.concatenate(out)

    def unload(self) -> None:
        self.model = self._text = self._mel = self._window = None


def _pooled(output: Any) -> Any:
    """transformers >= 5 returns a ModelOutput from ``get_*_features``; 4.x a tensor."""
    return getattr(output, "pooler_output", output)


def create_stem_tagger(settings: Settings, router: ContentClassifier) -> Any:
    """The tagger ``labeling`` names stems with (``AUDIO_STEM_LABEL_MODEL``): the router's own
    classifier when it is the one asked for (``router``, or the same ``panns``/``ast``), else a
    separate lazily loaded instance. Nothing is loaded here."""
    kind = settings.stem_label_model
    if kind == "router" or kind == settings.router_classifier:
        return router
    if kind == "panns":
        return PannsClassifier(checkpoint=settings.router_panns_checkpoint, device=settings.device)
    if kind == "ast":
        return AstClassifier(device=settings.device)
    if kind == "clap":
        from .labeling import clap_prompts

        return ClapTagger(clap_prompts(), model_id=settings.stem_label_clap_model, device=settings.device)
    raise ValueError(f"unknown stem label model {kind!r}")


def create_classifier(settings: Settings) -> ContentClassifier:
    if settings.router_classifier == "mock":
        return MockClassifier()
    if settings.router_classifier == "ast":
        return AstClassifier(device=settings.device)
    if settings.router_classifier == "panns":
        return PannsClassifier(checkpoint=settings.router_panns_checkpoint, device=settings.device)
    return NoClassifier()
