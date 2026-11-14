"""Runtime configuration, read once from environment variables.

Every setting is prefixed with ``AUDIO_`` so it cannot collide with Person 4's backend
configuration when both run from the same ``.env``.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("sam_audio_service")


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _opt_float(name: str) -> float | None:
    """``None`` when the variable is unset, so a backend-dependent default can apply."""
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else None


def _opt_int(name: str) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else None


@dataclass
class Settings:
    # "mock" is a deterministic DSP stand-in for local development and CI. It is NOT
    # source separation. "sam" loads the real SAM Audio checkpoints.
    backend: str = field(default_factory=lambda: os.getenv("AUDIO_BACKEND", "audiosep"))
    model_id: str = field(default_factory=lambda: os.getenv("AUDIO_MODEL_ID") or 'facebook/sam-audio-large')
    # "auto" picks cuda > mps > cpu.
    device: str = field(default_factory=lambda: os.getenv("AUDIO_DEVICE", "auto"))
    dtype: str = field(default_factory=lambda: os.getenv("AUDIO_DTYPE", "float32"))
    load_on_startup: bool = field(default_factory=lambda: _bool("AUDIO_LOAD_ON_STARTUP", True))

    # Inference defaults (overridable per request through metadata.options).
    predict_spans: bool = field(default_factory=lambda: _bool("AUDIO_PREDICT_SPANS", False))
    reranking_candidates: int = field(default_factory=lambda: _int("AUDIO_RERANKING_CANDIDATES", 1))
    # 0 disables chunking. Long inputs are otherwise processed in overlapping windows.
    # Unset => 10 s for the real model (upstream recommends 10-30 s windows and memory
    # scales with window length), 0 for the mock, which has no memory ceiling.
    chunk_seconds: float = 0.0
    chunk_seconds_env: float | None = field(default_factory=lambda: _opt_float("AUDIO_CHUNK_SECONDS"), repr=False)
    chunk_overlap_seconds: float = field(default_factory=lambda: _float("AUDIO_CHUNK_OVERLAP_SECONDS", 2.0))
    # How many chunk windows go into ONE model batch. This is the only supported way to
    # process several windows at once: the model is shared and must not be entered from
    # two threads (docs/chunk_concurrency.md). Batch k costs roughly k x activation memory.
    chunk_batch: int = 2
    chunk_batch_env: int | None = field(default_factory=lambda: _opt_int("AUDIO_CHUNK_BATCH"), repr=False)
    # Deprecated alias kept so existing compose files keep working: AUDIO_CHUNK_WORKERS
    # used to spawn threads; it now simply sets ``chunk_batch``.
    chunk_workers: int | None = field(default_factory=lambda: _opt_int("AUDIO_CHUNK_WORKERS"), repr=False)
    # "duplicate": run mono separation and duplicate to source channel count (1 pass).
    # "per_channel": separate each channel independently (N passes; independent estimates may alter stereo image).
    stereo_mode: str = field(default_factory=lambda: os.getenv("AUDIO_STEREO_MODE", "duplicate"))
    # Time-frequency mask refinement of each target stem (refine.py): "mask" for the real backend, "off" for mock.
    refine: str = field(default_factory=lambda: os.getenv("AUDIO_REFINE", ""))  # "" -> mask for sam, off for mock (see __post_init__)
    refine_p: float = field(default_factory=lambda: _float("AUDIO_REFINE_P", 2.0))
    refine_n_fft: int = field(default_factory=lambda: _int("AUDIO_REFINE_NFFT", 2048))
    refine_floor: float = field(default_factory=lambda: _float("AUDIO_REFINE_FLOOR", 0.0))
    refine_blend: float = field(default_factory=lambda: _float("AUDIO_REFINE_BLEND", 1.0))
    refine_threshold: float = field(default_factory=lambda: _float("AUDIO_REFINE_THRESHOLD", 0.0))
    refine_residual_threshold: float = field(default_factory=lambda: _float("AUDIO_REFINE_RESIDUAL_THRESHOLD", 0.0))
    refine_gate_db: float = field(default_factory=lambda: _float("AUDIO_REFINE_GATE_DB", 0.0))
    suppression: str = field(default_factory=lambda: os.getenv("AUDIO_SUPPRESSION", ""))
    suppression_strength: float = field(default_factory=lambda: _float("AUDIO_SUPPRESSION_STRENGTH", .9))
    # Noise draws returned per chunk window and how one is picked (candidates.py). Each
    # extra draw costs one more batch item (K x compute and activation memory).
    candidates: int = field(default_factory=lambda: _int("AUDIO_CANDIDATES", 1))
    candidate_select: str = field(default_factory=lambda: os.getenv("AUDIO_CANDIDATE_SELECT", "overlap"))

    # Limits.
    max_duration_seconds: float = field(default_factory=lambda: _float("AUDIO_MAX_DURATION_SECONDS", 600.0))
    max_input_bytes: int = field(default_factory=lambda: _int("AUDIO_MAX_INPUT_BYTES", 500 * 1024 * 1024))
    max_prompts: int = field(default_factory=lambda: _int("AUDIO_MAX_PROMPTS", 8))
    # Upper bound for options.passes (multi-pass separation; each pass is a full model run).
    max_passes: int = field(default_factory=lambda: _int("AUDIO_MAX_PASSES", 3))
    # Multi-pass safeguards (measured 2026-11-13, sam-audio-large: a second "guitar" pass on a
    # residual with no guitar left pulled the drums in and dropped the stem from 19.6 to 0.9 dB).
    # A pass is discarded, and the chain stops, when it extracts more than pass_max_ratio_db
    # relative to pass 1 (a leftover is quieter than the first pass) or when its long-term
    # spectrum is less similar than pass_min_similarity (cosine) to pass 1's (a different sound).
    # Unset -> -3 dB / 0.6 for the real backend; disabled (20 dB / 0) for the mock, whose band-pass
    # legitimately extracts more on a second pass and whose tests pin exact arithmetic.
    pass_max_ratio_db: float | None = None
    pass_min_similarity: float | None = None
    pass_max_ratio_db_env: float | None = field(default_factory=lambda: _opt_float("AUDIO_PASS_MAX_RATIO_DB"), repr=False)
    pass_min_similarity_env: float | None = field(default_factory=lambda: _opt_float("AUDIO_PASS_MIN_SIMILARITY"), repr=False)
    # Visually prompted requests run the whole selected interval as ONE model window (frames
    # must stay aligned with the audio, so no chunking). Upstream reports ~30 s as the
    # practical ceiling for the large model on an 80 GB GPU.
    max_visual_interval_seconds: float = field(default_factory=lambda: _float("AUDIO_MAX_VISUAL_INTERVAL_SECONDS", 30.0))

    # Media references. file:// references must resolve under media_root (a shared volume
    # owned by Person 4). http(s) references must match an allow-listed host.
    media_root: Path | None = field(default_factory=lambda: Path(os.environ["AUDIO_MEDIA_ROOT"]).resolve() if os.getenv("AUDIO_MEDIA_ROOT") else None)
    allowed_media_hosts: tuple[str, ...] = field(default_factory=lambda: tuple(h.strip() for h in os.getenv("AUDIO_ALLOWED_MEDIA_HOSTS", "").split(",") if h.strip()))
    # Where stems are written when a request asks for file references instead of base64.
    output_root: Path = field(default_factory=lambda: Path(os.getenv("AUDIO_OUTPUT_ROOT", "output")).resolve())

    # Real-model knobs (AUDIO_BACKEND=sam). See backends/sam_audio.py for the rationale,
    # the user-reported memory figures and the per-GPU-class recommendations.
    # "keep_object" follows the paper (object kept, background zeroed); "notebook"
    # reproduces the official notebook (object zeroed). A/B documented in sam_audio.py.
    visual_mask_mode: str = field(default_factory=lambda: os.getenv("AUDIO_VISUAL_MASK_MODE", "keep_object"))
    # The PE-A-Frame span predictor is multi-billion-parameter; off it is, no span prediction.
    load_span_predictor: bool = field(default_factory=lambda: _bool("AUDIO_LOAD_SPAN_PREDICTOR", False))
    # CLAP/Judge/ImageBind re-rankers; only used when reranking_candidates > 1.
    load_rankers: bool = field(default_factory=lambda: _bool("AUDIO_LOAD_RANKERS", True))
    load_visual_ranker: bool = field(default_factory=lambda: _bool("AUDIO_LOAD_VISUAL_RANKER", True))
    judge_only: bool = field(default_factory=lambda: _bool("AUDIO_JUDGE_ONLY", False))
    ode_method: str = field(default_factory=lambda: os.getenv("AUDIO_ODE_METHOD", "midpoint"))
    ode_step_size: float = field(default_factory=lambda: _float("AUDIO_ODE_STEP_SIZE", 2 / 32))
    # Frames and masks are pre-resized to this; the vision encoder resizes to 336 anyway.
    frame_size: int = field(default_factory=lambda: _int("AUDIO_FRAME_SIZE", 336))
    # Limits the official PE vision encoder microbatch without changing resolution/weights.
    visual_batch_size: int | None = field(default_factory=lambda: _opt_int("AUDIO_VISUAL_BATCH_SIZE"))

    ffmpeg: str = field(default_factory=lambda: os.getenv("AUDIO_FFMPEG", "ffmpeg"))
    ffprobe: str = field(default_factory=lambda: os.getenv("AUDIO_FFPROBE", "ffprobe"))
    ffmpeg_timeout: int = field(default_factory=lambda: _int("AUDIO_FFMPEG_TIMEOUT", 600))

    def __post_init__(self) -> None:
        if not self.suppression:
            self.suppression = 'conservative' if self.backend == 'audiosep' else 'off'
        if self.backend == 'audiosep' and self.model_id == 'facebook/sam-audio-large' and not os.getenv('AUDIO_MODEL_ID'):
            self.model_id = 'audiosep-base-4M'
        if self.backend == 'audiosep' and self.chunk_batch_env is None and self.chunk_workers is None:
            self.chunk_batch = 1
        if self.chunk_batch_env is not None:
            if self.chunk_workers is not None and self.chunk_workers != self.chunk_batch_env:
                # Silently ignoring the alias is how an operator ends up running a batch
                # size they did not ask for, so say which one won.
                log.warning(
                    "AUDIO_CHUNK_BATCH=%s wins over the deprecated AUDIO_CHUNK_WORKERS=%s; "
                    "remove AUDIO_CHUNK_WORKERS.", self.chunk_batch_env, self.chunk_workers,
                )
            self.chunk_batch = self.chunk_batch_env
        elif self.chunk_workers is not None:
            log.warning(
                "AUDIO_CHUNK_WORKERS is deprecated: chunk windows are batched, not threaded. "
                "Using it as AUDIO_CHUNK_BATCH=%s; set AUDIO_CHUNK_BATCH instead.", self.chunk_workers,
            )
            self.chunk_batch = self.chunk_workers
        self.chunk_workers = self.chunk_batch  # keep the alias readable and in sync
        if self.chunk_seconds_env is not None:
            self.chunk_seconds = self.chunk_seconds_env
        elif self.backend in {"sam", "audiosep"}:
            self.chunk_seconds = 10.0
        if self.pass_max_ratio_db_env is not None:
            self.pass_max_ratio_db = self.pass_max_ratio_db_env
        elif self.pass_max_ratio_db is None:
            self.pass_max_ratio_db = -3.0 if self.backend == "sam" else 20.0
        if self.pass_min_similarity_env is not None:
            self.pass_min_similarity = self.pass_min_similarity_env
        elif self.pass_min_similarity is None:
            self.pass_min_similarity = 0.6 if self.backend == "sam" else 0.0
        if not self.refine:
            # Measured 2026-11-13 on sam-audio-small: the mask improved every fixture (+1.8 to
            # +5.9 dB SI-SDRi, 1.7-6.5 dB less leakage) at no GPU cost, so it is on for the real
            # model. The mock keeps exact band-pass stems so tests and fixtures stay bit-exact.
            self.refine = "mask" if self.backend == "sam" else "off"

    def validate(self) -> None:
        if self.backend == 'audiosep' and self.model_id.startswith('facebook/sam-'):
            raise ValueError('AudioSep selected with an old SAM AUDIO_MODEL_ID; unset AUDIO_MODEL_ID or use audiosep-base-4M')
        if self.suppression not in {'off', 'conservative'}:
            raise ValueError('AUDIO_SUPPRESSION must be off or conservative')
        if not 0 <= self.suppression_strength <= 1:
            raise ValueError('AUDIO_SUPPRESSION_STRENGTH must be within [0, 1]')
        if self.suppression != 'off' and self.backend != 'audiosep':
            raise ValueError('Conservative suppression is available with AudioSep only')
        if self.visual_batch_size is not None and self.visual_batch_size <= 0:
            raise ValueError('AUDIO_VISUAL_BATCH_SIZE must be positive')
        if self.backend not in {"mock", "sam", "audiosep"}:
            raise ValueError(f"AUDIO_BACKEND must be 'mock', 'sam' or 'audiosep', got {self.backend!r}")
        if self.stereo_mode not in {"duplicate", "per_channel"}:
            raise ValueError("AUDIO_STEREO_MODE must be 'duplicate' or 'per_channel'")
        if self.max_visual_interval_seconds <= 0:
            raise ValueError("AUDIO_MAX_VISUAL_INTERVAL_SECONDS must be positive")
        if not 1 <= self.chunk_batch <= 8:
            raise ValueError("AUDIO_CHUNK_BATCH (deprecated alias AUDIO_CHUNK_WORKERS) must be between 1 and 8")
        if not -60 <= self.pass_max_ratio_db <= 20:
            raise ConfigurationError("AUDIO_PASS_MAX_RATIO_DB must be within [-60, 20] (20 effectively disables the guard)")
        if not 0 <= self.pass_min_similarity <= 1:
            raise ConfigurationError("AUDIO_PASS_MIN_SIMILARITY must be within [0, 1]")
        if not 1 <= self.max_passes <= 5:
            raise ValueError("AUDIO_MAX_PASSES must be between 1 and 5")
        if self.chunk_seconds < 0:
            raise ValueError("AUDIO_CHUNK_SECONDS must not be negative")
        if self.chunk_overlap_seconds <= 0:
            raise ValueError("AUDIO_CHUNK_OVERLAP_SECONDS must be positive")
        if self.chunk_seconds and self.chunk_seconds <= 2 * self.chunk_overlap_seconds:
            raise ValueError("AUDIO_CHUNK_SECONDS must exceed twice AUDIO_CHUNK_OVERLAP_SECONDS")
        if self.visual_mask_mode not in {"keep_object", "notebook"}:
            raise ValueError("AUDIO_VISUAL_MASK_MODE must be 'keep_object' or 'notebook'")
        if self.frame_size <= 0:
            raise ValueError("AUDIO_FRAME_SIZE must be positive")
        if not 0 < self.ode_step_size <= 1:
            raise ValueError("AUDIO_ODE_STEP_SIZE must be in (0, 1]")
        if self.refine not in {"off", "mask"}:
            raise ValueError("AUDIO_REFINE must be 'off' or 'mask'")
        if not 0 < self.refine_p <= 8:
            raise ValueError("AUDIO_REFINE_P must be in (0, 8]")
        if self.refine_n_fft not in {512, 1024, 2048, 4096}:
            raise ValueError("AUDIO_REFINE_NFFT must be 512, 1024, 2048 or 4096")
        if not 0 <= self.refine_floor <= 0.5:
            raise ValueError("AUDIO_REFINE_FLOOR must be in [0, 0.5]")
        if not 0 <= self.refine_blend <= 1:
            raise ValueError("AUDIO_REFINE_BLEND must be in [0, 1]")
        if not 0 <= self.refine_threshold < 1:
            raise ValueError("AUDIO_REFINE_THRESHOLD must be in [0, 1)")
        if not 0 <= self.refine_residual_threshold < 1:
            raise ValueError("AUDIO_REFINE_RESIDUAL_THRESHOLD must be in [0, 1)")
        if not 0 <= self.refine_gate_db <= 80:
            raise ValueError("AUDIO_REFINE_GATE_DB must be in [0, 80]")
        if not 1 <= self.candidates <= 8:
            raise ValueError("AUDIO_CANDIDATES must be between 1 and 8")
        if self.candidate_select not in {"first", "overlap", "mean", "auto"}:
            raise ValueError("AUDIO_CANDIDATE_SELECT must be 'first', 'overlap' or 'mean'")


def load_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings
