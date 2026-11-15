"""The speech/rhythm toolset behind /v1/words, /v1/enhance and /v1/beats.

One :class:`SpeechTools` per app. Every model is optional (``AUDIO_*_BACKEND=off``), loaded
on first use (or at startup with AUDIO_LOAD_ON_STARTUP), and guarded by its **own** lock:
these never take the engine's separation lock, so "bleep the swearing" does not queue
behind a 4-minute stem split and a split never waits for a transcription. The models are
small next to the separators (Whisper large-v3-turbo fp16 ~1.6 GB, DeepFilterNet3 and
beat_this < 0.5 GB each), so they share the GPU.

Load failures are remembered (surfaced in /health and /capabilities) and retried on the
next request, like the stem splitter; out-of-memory maps to INSUFFICIENT_GPU_MEMORY.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import numpy as np

from ..audio_io import DecodedAudio, peak_dbfs, rms_dbfs
from ..config import Settings
from ..errors import InferenceFailure, InsufficientGpuMemory, InvalidRequest, MissingModel, UnsupportedOperation, is_out_of_memory
from .beats import BeatTracker, LibrosaBeatTracker, create_beat_tracker, summarise
from .contracts import BeatsMetadata, EnhanceMetadata, WordsMetadata
from .enhance import Enhancer, create_enhancer
from .lexicon import ProfanityList, find_fillers, find_pauses, find_profanity, normalize
from .words import RATE as WORDS_RATE
from .words import Transcriber, Wav2Vec2Aligner, create_transcriber

log = logging.getLogger("sam_audio_service")

WORDS_SAMPLE_RATE = WORDS_RATE  # /v1/words decodes uploads to 16 kHz mono
BEATS_SAMPLE_RATE = 22_050  # beat_this and librosa both work at 22.05 kHz mono
ENHANCE_SAMPLE_RATE = 48_000


def _ms(x: float) -> float:
    return round(float(x), 3)


def _db(x: float) -> float | None:
    return None if not np.isfinite(x) else round(float(x), 2)


class _Slot:
    """One lazily loaded model + its lock + its last load error."""

    def __init__(self, label: str, model: Any, details: dict[str, Any]):
        self.label, self.model, self.details = label, model, details
        self.lock = threading.Lock()
        self.error: str | None = None

    def ensure_loaded(self) -> None:  # call with self.lock held
        if self.model.loaded:
            return
        try:
            self.model.load()
            self.error = None
        except Exception as exc:  # noqa: BLE001 - surfaced through /health and MISSING_MODEL
            self.error = f"{type(exc).__name__}: {exc}"
            log.exception("%s load failed", self.label)
            if is_out_of_memory(exc):
                raise InsufficientGpuMemory(f"not enough GPU memory to load the {self.label}", details=self.details) from exc
            raise MissingModel(f"{self.label} is not available: {self.error}", details=self.details) from exc

    def run(self, fn: Callable[[], Any]) -> Any:
        with self.lock:
            self.ensure_loaded()
            try:
                return fn()
            except (InvalidRequest, UnsupportedOperation):
                raise
            except Exception as exc:  # noqa: BLE001
                if is_out_of_memory(exc):
                    _empty_cache()
                    raise InsufficientGpuMemory(f"the {self.label} ran out of GPU memory; retry with shorter audio", details=self.details) from exc
                log.exception("%s failed", self.label)
                raise InferenceFailure(f"{self.label} failed: {type(exc).__name__}: {exc}", details=self.details) from exc

    def status(self) -> dict[str, Any]:
        return {**self.model.info(), "load_error": self.error}


def _empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


class SpeechTools:
    def __init__(self, settings: Settings, transcriber: Transcriber | None = None, aligner: Any | None = None,
                 enhancer: Enhancer | None = None, beat_tracker: BeatTracker | None = None, beat_fallback: BeatTracker | None = None):
        self.settings = settings
        device = settings.speech_device if settings.speech_device != "auto" else settings.device
        transcriber = transcriber or create_transcriber(settings, device)
        enhancer = enhancer or create_enhancer(settings, device)
        beat_tracker = beat_tracker or create_beat_tracker(settings, device)
        if aligner is None and settings.words_align == "wav2vec2" and settings.words_backend == "faster_whisper":
            aligner = Wav2Vec2Aligner(device=device)
        if beat_fallback is None and settings.beats_fallback == "librosa" and beat_tracker is not None and beat_tracker.name == "beat_this":
            beat_fallback = LibrosaBeatTracker()
        self.words = _Slot("word transcriber", transcriber, {"words_model": settings.words_model}) if transcriber else None
        self.aligner = _Slot("word aligner", aligner, {"align_model": getattr(aligner, "model_id", None)}) if aligner else None
        self.enhancer = _Slot("speech enhancer", enhancer, {"enhance_model": settings.enhance_model}) if enhancer else None
        self.beats = _Slot("beat tracker", beat_tracker, {"beats_model": settings.beats_model}) if beat_tracker else None
        self.beat_fallback = _Slot("fallback beat tracker", beat_fallback, {"beats_model": getattr(beat_fallback, "model_id", None)}) if beat_fallback else None
        self.profanity = ProfanityList.load(settings.words_profanity_file)

    # ------------------------------------------------------------------ lifecycle / status
    def slots(self) -> list[_Slot]:
        return [s for s in (self.words, self.aligner, self.enhancer, self.beats) if s is not None]

    def load(self) -> None:
        """Startup preload; failures are recorded and retried on first request."""
        for slot in self.slots():
            try:
                with slot.lock:
                    slot.ensure_loaded()
            except Exception:  # noqa: BLE001
                log.warning("%s did not load on startup; will retry on first request", slot.label)

    def unload(self) -> None:
        for slot in self.slots() + ([self.beat_fallback] if self.beat_fallback else []):
            slot.model.unload()

    def status(self) -> dict[str, Any]:
        return {
            "words": self.words.status() if self.words else None,
            "aligner": self.aligner.status() if self.aligner else None,
            "enhance": self.enhancer.status() if self.enhancer else None,
            "beats": self.beats.status() if self.beats else None,
        }

    def capabilities(self) -> dict[str, Any]:
        s = self.settings
        words: dict[str, Any] = {"available": False, "backend": None, "model_id": None}
        if self.words:
            m = self.words.model
            words = {"available": True, "backend": m.name, "is_mock": m.is_mock, "model_id": m.model_id, "license": m.license, "loaded": m.loaded, "load_error": self.words.error,
                     "align": {"available": self.aligner is not None, "model_id": getattr(self.aligner.model, "model_id", None) if self.aligner else None,
                               "languages": list(getattr(self.aligner.model, "languages", ())) if self.aligner else [], "default": s.words_align != "off" and self.aligner is not None},
                     "vad": s.words_vad, "input_sample_rate": WORDS_SAMPLE_RATE, "profanity_list": {"source": self.profanity.source, "entries": len(self.profanity)},
                     "min_pause_seconds": s.words_min_pause_seconds, "max_duration_seconds": s.max_duration_seconds}
        enhance: dict[str, Any] = {"available": False, "backend": None, "model_id": None}
        if self.enhancer:
            m = self.enhancer.model
            enhance = {"available": True, "backend": m.name, "is_mock": m.is_mock, "model_id": m.model_id, "license": m.license, "loaded": m.loaded, "load_error": self.enhancer.error,
                       "model_sample_rate": ENHANCE_SAMPLE_RATE, "atten_lim_db": {"default": s.enhance_atten_limit_db, "min": 0, "max": 100},
                       "output": {"formats": ["pcm_f32le", "pcm_s16le"], "return_modes": ["base64", "file"], "length": "same as input", "channels": "same as input (1 or 2)"},
                       "max_duration_seconds": s.max_duration_seconds}
        beats: dict[str, Any] = {"available": False, "backend": None, "model_id": None}
        if self.beats:
            m = self.beats.model
            beats = {"available": True, "backend": m.name, "is_mock": m.is_mock, "model_id": m.model_id, "license": m.license, "loaded": m.loaded, "load_error": self.beats.error,
                     "downbeats": m.has_downbeats, "fallback": self.beat_fallback.model.name if self.beat_fallback else None, "max_duration_seconds": s.max_duration_seconds}
        return {"words": words, "enhance": enhance, "beats": beats}

    # ------------------------------------------------------------------ helpers
    def _check(self, slot: _Slot | None, what: str, env: str, audio: DecodedAudio) -> _Slot:
        if slot is None:
            raise UnsupportedOperation(f"{what} is not enabled on this service (set {env})", details={"setting": env})
        if audio.frames == 0:
            raise InvalidRequest("audio is empty")
        if audio.duration > self.settings.max_duration_seconds:
            raise UnsupportedOperation(f"audio is {audio.duration:.1f}s; the limit is {self.settings.max_duration_seconds:.0f}s",
                                       details={"max_duration_seconds": self.settings.max_duration_seconds})
        return slot

    @staticmethod
    def _source(audio: DecodedAudio, asset_id: str | None) -> dict[str, Any]:
        return {"asset_id": asset_id, "duration": round(audio.duration, 6), "sample_rate": audio.source_sample_rate, "channels": audio.source_channels}

    # ------------------------------------------------------------------ /v1/words
    def transcribe(self, audio: DecodedAudio, meta: WordsMetadata, request_id: str) -> dict[str, Any]:
        """``audio`` is decoded at 16 kHz (mono or not; it is downmixed here)."""
        slot = self._check(self.words, "word timestamps", "AUDIO_WORDS_BACKEND", audio)
        if audio.sample_rate != WORDS_SAMPLE_RATE:
            raise InvalidRequest(f"internal: /v1/words expects {WORDS_SAMPLE_RATE} Hz audio")
        mono = audio.samples.mean(axis=0).astype(np.float32) if audio.channels > 1 else audio.samples[0]
        warnings: list[str] = []
        model = slot.model
        if model.is_mock:
            warnings.append("MOCK word transcriber: one fixed word per voiced region, not speech recognition")
        t0 = time.perf_counter()
        transcript = slot.run(lambda: model.transcribe(mono, meta.language, meta.transcribe_fillers))
        t_asr = time.perf_counter() - t0
        want_align = meta.align if meta.align is not None else self.settings.words_align != "off"
        aligned, t_align = False, 0.0
        if want_align and transcript.words:
            if self.aligner is None:
                if meta.align:
                    warnings.append("forced alignment is not configured (AUDIO_WORDS_ALIGN=off); word times are Whisper's")
            elif transcript.language not in getattr(self.aligner.model, "languages", ()):
                warnings.append(f"forced alignment supports {list(self.aligner.model.languages)}; '{transcript.language}' keeps Whisper's word times")
            else:
                t1 = time.perf_counter()
                try:
                    retimed = self.aligner.run(lambda: self.aligner.model.align(mono, transcript))
                    aligned = retimed > 0
                    if retimed < len(transcript.words):
                        warnings.append(f"{len(transcript.words) - retimed} of {len(transcript.words)} words could not be aligned (numbers, symbols, censored words) and keep Whisper's times")
                except (MissingModel, InsufficientGpuMemory, InferenceFailure) as exc:
                    warnings.append(f"forced alignment skipped: {exc.message}")
                t_align = time.perf_counter() - t1
        words = [{"word": w["word"], "norm": normalize(w["word"]), "start": _ms(w["start"]), "end": _ms(max(w["end"], w["start"])),
                  "confidence": round(float(w["confidence"]), 4), "segment": int(w["segment"])} for w in transcript.words]
        segments = [{**s, "start": _ms(s["start"]), "end": _ms(s["end"])} for s in transcript.segments]
        profanity = self.profanity.with_extra(meta.extra_profanity)
        min_pause = meta.min_pause_seconds or self.settings.words_min_pause_seconds
        info = model.info()
        model_info = {"backend": model.name, "model_id": model.model_id, "is_mock": model.is_mock,
                      **{k: info[k] for k in ("device", "compute_type", "vad") if k in info},
                      "filler_prompt": bool(transcript.info.get("filler_prompt", False)),
                      "align_model": getattr(self.aligner.model, "model_id", None) if aligned and self.aligner else None}
        return {
            "request_id": request_id, "language": transcript.language,
            "language_probability": None if transcript.language_probability is None else round(transcript.language_probability, 4),
            "words": words, "segments": segments,
            "fillers": find_fillers(words), "profanity": find_profanity(words, profanity), "pauses": find_pauses(words, min_pause),
            "aligned": aligned, "model": model_info, "source": self._source(audio, meta.asset_id),
            "profanity_list": {"source": profanity.source, "entries": len(profanity)},
            "timing": {"transcribe_seconds": round(t_asr, 3), "align_seconds": round(t_align, 3)},
            "warnings": warnings,
        }

    # ------------------------------------------------------------------ /v1/enhance
    def enhance(self, audio: DecodedAudio, meta: EnhanceMetadata) -> tuple[np.ndarray, dict[str, Any]]:
        """``audio`` at 48 kHz. Returns (enhanced float32 (channels, n) at 48 kHz, report)."""
        slot = self._check(self.enhancer, "speech enhancement", "AUDIO_ENHANCE_BACKEND", audio)
        if audio.sample_rate != ENHANCE_SAMPLE_RATE:
            raise InvalidRequest(f"internal: /v1/enhance expects {ENHANCE_SAMPLE_RATE} Hz audio")
        model = slot.model
        warnings: list[str] = []
        if model.is_mock:
            warnings.append("MOCK speech enhancer: a fixed spectral gate, not DeepFilterNet")
        lim = meta.atten_lim_db if meta.atten_lim_db is not None else self.settings.enhance_atten_limit_db
        if lim is not None and lim >= 100:
            lim = None
        t0 = time.perf_counter()
        out = slot.run(lambda: model.enhance(audio.samples, lim))
        elapsed = time.perf_counter() - t0
        out = np.nan_to_num(np.asarray(out, dtype=np.float32))
        if out.shape != audio.samples.shape:
            raise InferenceFailure(f"speech enhancer returned shape {out.shape}, expected {audio.samples.shape}")
        report = {"atten_lim_db": lim, "warnings": warnings, "timing": {"enhance_seconds": round(elapsed, 3)},
                  "levels": {"rms_dbfs_in": _db(rms_dbfs(audio.samples)), "rms_dbfs_out": _db(rms_dbfs(out)),
                             "peak_dbfs_in": _db(peak_dbfs(audio.samples)), "peak_dbfs_out": _db(peak_dbfs(out)),
                             "removed_rms_dbfs": _db(rms_dbfs(audio.samples - out))},
                  "model": {"backend": model.name, "model_id": model.model_id, "is_mock": model.is_mock, **{k: v for k, v in model.info().items() if k in ("device", "libdf")}},
                  "source": self._source(audio, meta.asset_id)}
        return out, report

    # ------------------------------------------------------------------ /v1/beats
    def track_beats(self, audio: DecodedAudio, meta: BeatsMetadata, request_id: str) -> dict[str, Any]:
        slot = self._check(self.beats, "beat tracking", "AUDIO_BEATS_BACKEND", audio)
        warnings: list[str] = []
        tracker = slot.model
        t0 = time.perf_counter()
        try:
            result = slot.run(lambda: tracker.track(audio.samples, audio.sample_rate))
        except (MissingModel, InsufficientGpuMemory, InferenceFailure) as exc:
            if self.beat_fallback is None:
                raise
            warnings.append(f"{tracker.name} unavailable ({exc.message}); used {self.beat_fallback.model.name} (no downbeats)")
            tracker = self.beat_fallback.model
            result = self.beat_fallback.run(lambda: tracker.track(audio.samples, audio.sample_rate))
        elapsed = time.perf_counter() - t0
        if tracker.is_mock:
            warnings.append("MOCK beat tracker: a fixed 120 BPM grid, not beat tracking")
        beats = sorted(_ms(b) for b in result["beats"] if 0 <= b <= audio.duration + 1e-6)
        downbeats = sorted(_ms(b) for b in result["downbeats"] if 0 <= b <= audio.duration + 1e-6)
        if not tracker.has_downbeats:
            warnings.append(f"{tracker.name} does not estimate downbeats; downbeats is empty")
        summary = summarise(beats, downbeats)
        return {"request_id": request_id, "beats": beats, "downbeats": downbeats, **summary,
                "downbeats_available": tracker.has_downbeats,
                "model": {"backend": tracker.name, "model_id": tracker.model_id, "is_mock": tracker.is_mock, **{k: v for k, v in tracker.info().items() if k in ("device", "dbn")}},
                "source": self._source(audio, meta.asset_id), "timing": {"track_seconds": round(elapsed, 3)}, "warnings": warnings}
