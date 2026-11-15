"""Word-level transcription for /v1/words: faster-whisper, plus optional CTC forced alignment.

``Transcriber.transcribe(mono_16k, language, bias_fillers)`` returns a :class:`Transcript`
with every word's start/end in seconds on the input timeline. Implementations:

* ``mock``           deterministic fake for tests/CI: a fixed script spread over the voiced
                     regions of the input. NOT speech recognition.
* ``faster_whisper`` CTranslate2 Whisper (MIT code, MIT weights), word_timestamps=True and the
                     Silero VAD filter. Batched pipeline when AUDIO_WORDS_BATCH_SIZE > 1.

:class:`Wav2Vec2Aligner` (``AUDIO_WORDS_ALIGN=wav2vec2``) re-times Whisper's words WhisperX-style:
per segment, the torchaudio ``WAV2VEC2_ASR_BASE_960H`` character CTC emissions are force-aligned
to the transcript (Viterbi below, numpy, so it does not depend on torchaudio's deprecated
``forced_align``). English only; other languages keep Whisper's own timings.

Whisper drops hesitations ("um", "uh") by default. ``bias_fillers`` passes a disfluent
``initial_prompt`` (the OpenAI cookbook trick) so they are written out and /v1/words can find them.
"""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .lexicon import normalize

log = logging.getLogger("sam_audio_service")

RATE = 16_000
FILLER_PROMPT = "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. Uh, so, um, yeah."


@dataclass
class Transcript:
    words: list[dict[str, Any]]  # {word, start, end, confidence, segment}
    segments: list[dict[str, Any]]  # {id, start, end, text, word_start, word_end}
    language: str | None
    language_probability: float | None
    info: dict[str, Any] = field(default_factory=dict)


def device_split(device: str) -> tuple[str, int]:
    """"cuda:1" -> ("cuda", 1); "cuda" -> ("cuda", 0)."""
    if ":" in device:
        kind, index = device.split(":", 1)
        return kind, int(index)
    return device, 0


class Transcriber(ABC):
    name = "base"
    is_mock = False
    model_id = ""
    license: str | None = None

    @property
    @abstractmethod
    def loaded(self) -> bool: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def transcribe(self, mono: np.ndarray, language: str | None, bias_fillers: bool) -> Transcript:
        """``mono`` is float32 at 16 kHz."""

    def info(self) -> dict[str, Any]:
        return {"backend": self.name, "model_id": self.model_id, "is_mock": self.is_mock, "loaded": self.loaded, "license": self.license}

    def unload(self) -> None:
        return None


class MockTranscriber(Transcriber):
    """Places a fixed script over the voiced parts of the input (RMS gate on 20 ms frames).

    Each contiguous voiced region becomes one word, so a test that synthesises N tone bursts
    gets N words with exact timings."""

    name = "mock"
    is_mock = True
    model_id = "mock-words"
    SCRIPT = ("Um", "so", "what", "the", "fuck,", "like,", "is", "this", "uh", "thing", "damn", "what")

    def __init__(self, script: tuple[str, ...] | None = None):
        self.script = script or self.SCRIPT
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._loaded = True

    def transcribe(self, mono: np.ndarray, language: str | None, bias_fillers: bool) -> Transcript:
        hop = RATE // 50
        n = len(mono) // hop
        rms = np.sqrt(np.mean(mono[: n * hop].reshape(n, hop) ** 2, axis=1)) if n else np.zeros(0)
        voiced = rms > max(1e-3, 0.1 * (rms.max() if rms.size else 0))
        regions, start = [], None
        for i, v in enumerate(voiced.tolist() + [False]):
            if v and start is None:
                start = i
            elif not v and start is not None:
                regions.append((start * hop / RATE, i * hop / RATE))
                start = None
        words = [{"word": self.script[k % len(self.script)], "start": round(s, 3), "end": round(e, 3), "confidence": 0.99, "segment": 0}
                 for k, (s, e) in enumerate(regions)]
        segments = [{"id": 0, "start": words[0]["start"], "end": words[-1]["end"], "text": " ".join(w["word"] for w in words), "word_start": 0, "word_end": len(words)}] if words else []
        return Transcript(words, segments, language or "en", 1.0, {"mock": True})


class FasterWhisperTranscriber(Transcriber):
    name = "faster_whisper"
    license = "code MIT (SYSTRAN/faster-whisper); weights MIT (OpenAI Whisper)"

    def __init__(self, model_id: str, device: str = "auto", compute_type: str = "auto", beam_size: int = 5, batch_size: int = 8, vad: bool = True):
        self.model_id, self.requested_device = model_id, device
        self.device = device
        self.compute_type = compute_type
        self.beam_size, self.batch_size, self.vad = beam_size, batch_size, vad
        self.model: Any = None
        self.pipeline: Any = None

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def info(self) -> dict[str, Any]:
        return {**super().info(), "device": self.device, "compute_type": self.compute_type, "beam_size": self.beam_size, "batch_size": self.batch_size, "vad": self.vad}

    def load(self) -> None:
        if self.loaded:
            return
        # torch first: it loads the CUDA 12 / cuDNN 9 libraries (pip wheels) that CTranslate2 links against.
        import torch  # noqa: F401
        from faster_whisper import BatchedInferencePipeline, WhisperModel

        from ..stems.demucs import resolve_device

        self.device = resolve_device(self.requested_device)
        kind, index = device_split(self.device)
        kind = "cuda" if kind == "cuda" else "cpu"  # CTranslate2 has no MPS backend
        if self.compute_type == "auto":
            self.compute_type = "float16" if kind == "cuda" else "int8"
        self.model = WhisperModel(self.model_id, device=kind, device_index=index, compute_type=self.compute_type)
        self.pipeline = BatchedInferencePipeline(self.model) if self.batch_size > 1 else None

    def transcribe(self, mono: np.ndarray, language: str | None, bias_fillers: bool) -> Transcript:
        self.load()
        audio = np.ascontiguousarray(mono, dtype=np.float32)
        detected_prob = None
        if language is None:
            # Detect first so the English filler prompt never biases a non-English file.
            language, detected_prob, _ = self.model.detect_language(audio, vad_filter=self.vad)
        prompt = FILLER_PROMPT if bias_fillers and language == "en" else None
        kwargs = dict(language=language, beam_size=self.beam_size, word_timestamps=True, initial_prompt=prompt)
        if self.pipeline is not None:
            segments, info = self.pipeline.transcribe(audio, batch_size=self.batch_size, vad_filter=self.vad, **kwargs)
        else:
            segments, info = self.model.transcribe(audio, vad_filter=self.vad, condition_on_previous_text=False, **kwargs)
        words: list[dict[str, Any]] = []
        segs: list[dict[str, Any]] = []
        for seg in segments:
            first = len(words)
            for w in seg.words or []:
                text = w.word.strip()
                if not text:
                    continue
                words.append({"word": text, "start": float(w.start), "end": float(max(w.end, w.start)), "confidence": float(w.probability), "segment": len(segs)})
            if len(words) == first:
                continue
            segs.append({"id": len(segs), "start": float(seg.start), "end": float(seg.end), "text": seg.text.strip(), "word_start": first, "word_end": len(words)})
        prob = detected_prob if detected_prob is not None else getattr(info, "language_probability", None)
        return Transcript(words, segs, getattr(info, "language", language), None if prob is None else float(prob),
                          {"filler_prompt": prompt is not None})

    def unload(self) -> None:
        self.model = self.pipeline = None


# ----------------------------------------------------------------------------------- alignment
def ctc_viterbi(log_probs: np.ndarray, targets: list[int], blank: int = 0) -> np.ndarray | None:
    """Most likely CTC path through ``targets``: returns, for each target, ``[first, last]``
    emission frame, shape ``(len(targets), 2)``; ``None`` when the frames cannot fit them."""
    T, L = log_probs.shape[0], len(targets)
    if L == 0:
        return np.zeros((0, 2), dtype=int)
    S = 2 * L + 1
    z = np.full(S, blank, dtype=np.int64)
    z[1::2] = targets
    # Minimum frames: one per target plus one blank between equal neighbours.
    if T < L + sum(1 for a, b in zip(targets, targets[1:]) if a == b):
        return None
    neg = -1e30
    emit = log_probs[:, z]  # (T, S)
    skip_ok = np.zeros(S, dtype=bool)
    skip_ok[2:] = (z[2:] != blank) & (z[2:] != z[:-2])
    alpha = np.full(S, neg)
    alpha[0] = emit[0, 0]
    alpha[1] = emit[0, 1]
    back = np.zeros((T, S), dtype=np.int8)
    for t in range(1, T):
        stay = alpha
        step = np.concatenate(([neg], alpha[:-1]))
        jump = np.where(skip_ok, np.concatenate(([neg, neg], alpha[:-2])), neg)
        best = np.argmax(np.stack([stay, step, jump]), axis=0)
        alpha = np.choose(best, [stay, step, jump]) + emit[t]
        back[t] = best
    s = S - 1 if alpha[S - 1] >= alpha[S - 2] else S - 2
    if alpha[s] <= neg / 2:
        return None
    spans = np.full((L, 2), -1, dtype=int)
    for t in range(T - 1, -1, -1):
        if s % 2 == 1:
            k = s // 2
            spans[k, 1] = t if spans[k, 1] < 0 else spans[k, 1]
            spans[k, 0] = t
        s -= int(back[t, s])
    return spans


class Wav2Vec2Aligner:
    """WhisperX-style word re-timing with an English character CTC model (torchaudio bundle
    ``WAV2VEC2_ASR_BASE_960H``: fairseq wav2vec 2.0 base fine-tuned on LibriSpeech 960 h, MIT)."""

    name = "wav2vec2"
    model_id = "torchaudio/WAV2VEC2_ASR_BASE_960H"
    license = "MIT (fairseq wav2vec 2.0 weights via torchaudio)"
    languages = ("en",)
    pad_seconds = 0.25
    start_shift = 0.05  # seconds; see align()

    def __init__(self, device: str = "auto"):
        self.requested_device = self.device = device
        self.model: Any = None
        self.vocab: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def info(self) -> dict[str, Any]:
        return {"aligner": self.name, "model_id": self.model_id, "loaded": self.loaded, "license": self.license, "languages": list(self.languages)}

    def load(self) -> None:
        if self.loaded:
            return
        import torch
        import torchaudio

        from ..stems.demucs import resolve_device

        self.device = resolve_device(self.requested_device)
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = bundle.get_model().eval().to(self.device)
        self.vocab = {c: i for i, c in enumerate(bundle.get_labels())}
        self._torch = torch

    def emissions(self, chunk: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))[None].to(self.device)
            logits, _ = self.model(x)
            return torch.log_softmax(logits[0].float(), dim=-1).cpu().numpy()

    def align(self, mono: np.ndarray, transcript: Transcript) -> int:
        """Re-time ``transcript.words`` in place; returns how many words were re-timed."""
        if transcript.language not in self.languages or not transcript.words:
            return 0
        self.load()
        words, done = transcript.words, set()
        duration = len(mono) / RATE
        for seg in transcript.segments:
            idx = range(seg["word_start"], seg["word_end"])
            start = max(0.0, min(seg["start"], words[idx[0]]["start"]) - self.pad_seconds)
            end = min(duration, max(seg["end"], words[idx[-1]]["end"]) + self.pad_seconds)
            chunk = mono[int(start * RATE): int(end * RATE)]
            if len(chunk) < RATE // 10:
                continue
            targets, owner = [], []
            sep = self.vocab.get("|")
            for i in idx:
                raw = words[i]["word"]
                if "*" in raw:  # censored partial: letters do not match the audio
                    continue
                chars = [self.vocab[c] for c in normalize(raw).upper().replace(" ", "") if c in self.vocab and c not in "|-"]
                if not chars:
                    continue  # digits, symbols: keep Whisper's time
                if targets and sep is not None:
                    targets.append(sep); owner.append(-1)
                targets.extend(chars); owner.extend([i] * len(chars))
            if not targets:
                continue
            with self._lock:
                emission = self.emissions(chunk)
            spans = ctc_viterbi(emission, targets, blank=0)
            if spans is None:
                continue
            frame = (len(chunk) / RATE) / emission.shape[0]
            bounds: dict[int, list[int]] = {}
            for (f0, f1), w in zip(spans, owner):
                if w < 0:
                    continue
                b = bounds.setdefault(w, [f0, f1])
                b[0], b[1] = min(b[0], f0), max(b[1], f1)
            for w, (f0, f1) in bounds.items():
                words[w]["start"] = start + f0 * frame
                words[w]["end"] = start + (f1 + 1) * frame
                done.add(w)
        # CTC emits a character's spike slightly after its acoustic onset: aligned starts were
        # +62 ms late against MFA on LibriSpeech and +55 ms against sample-exact spliced TTS
        # (benchmarks/speech.py, 2026-11-14). Shift them earlier, never into the previous word.
        if self.start_shift:
            for i in sorted(done):
                floor = words[i - 1]["end"] if i > 0 else 0.0
                words[i]["start"] = max(words[i]["start"] - self.start_shift, floor, 0.0) if words[i]["start"] > floor else words[i]["start"]
        # Keep the timeline monotonic where an un-aligned word (a number) overlaps an aligned
        # one: the aligned time wins and the Whisper-timed neighbour is clipped.
        for i in range(len(words) - 1):
            a, b = words[i], words[i + 1]
            if b["start"] < a["end"]:
                if i + 1 not in done:
                    b["start"] = a["end"]
                    b["end"] = max(b["end"], b["start"])
                elif i not in done:
                    a["end"] = b["start"]
                    a["start"] = min(a["start"], a["end"])
                else:
                    a["end"] = b["start"]
        return len(done)

    def unload(self) -> None:
        self.model = None


def create_transcriber(settings, device: str) -> Transcriber | None:
    if settings.words_backend == "mock":
        return MockTranscriber()
    if settings.words_backend == "faster_whisper":
        return FasterWhisperTranscriber(settings.words_model, device=device, compute_type=settings.words_compute_type,
                                        beam_size=settings.words_beam_size, batch_size=settings.words_batch_size, vad=settings.words_vad)
    return None
