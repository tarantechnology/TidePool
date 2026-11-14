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
    backend: str = field(default_factory=lambda: os.getenv("AUDIO_BACKEND", "mock"))
    model_id: str = field(default_factory=lambda: os.getenv("AUDIO_MODEL_ID", "facebook/sam-audio-large"))
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
    # Noise draws returned per chunk window and how one is picked (candidates.py). Each
    # extra draw costs one more batch item (K x compute and activation memory).
    candidates: int = field(default_factory=lambda: _int("AUDIO_CANDIDATES", 1))
    candidate_select: str = field(default_factory=lambda: os.getenv("AUDIO_CANDIDATE_SELECT", "overlap"))

    # Music stems (stems/) and the model router (routing.py). The splitter is opt-in for every
    # backend, the mock included: with it off, requests behave exactly as before music mode
    # existed (an empty prompt list is still 400). "mock" = frequency bands for CI; "demucs" /
    # "chain" (vocal model + Demucs) load real weights on first use.
    stem_splitter: str = field(default_factory=lambda: os.getenv("AUDIO_STEM_SPLITTER", "off"))
    # Demucs model name (htdemucs, htdemucs_ft, htdemucs_6s). A name ending in "6s" also gives
    # the mock splitter guitar and piano bands.
    stem_model: str = field(default_factory=lambda: os.getenv("AUDIO_STEM_MODEL", "htdemucs_ft"))
    stem_vocal_model: str = field(default_factory=lambda: os.getenv("AUDIO_STEM_VOCAL_MODEL", "model_bs_roformer_ep_317_sdr_12.9755.ckpt"))  # chain only
    stem_model_dir: str | None = field(default_factory=lambda: os.getenv("AUDIO_STEM_MODEL_DIR") or None)  # chain only
    # chain only: audio-separator mdxc_params["overlap"] for the vocal model (more = slower, smoother).
    stem_vocal_overlap: int = field(default_factory=lambda: _int("AUDIO_STEM_VOCAL_OVERLAP", 8))
    # chain only: how the vocal model runs. "separator" = audio-separator (temp WAV files);
    # "native" = in-process BS-RoFormer (stems/roformer.py) with batched fp16 windows, built from
    # AUDIO_STEM_VOCAL_CONFIG (default: the checkpoint path with .yaml, as audio-separator stores it).
    stem_vocal_backend: str = field(default_factory=lambda: os.getenv("AUDIO_STEM_VOCAL_BACKEND", "separator"))
    stem_vocal_config: str | None = field(default_factory=lambda: os.getenv("AUDIO_STEM_VOCAL_CONFIG") or None)
    stem_vocal_fp16: bool = field(default_factory=lambda: _bool("AUDIO_STEM_VOCAL_FP16", True))  # native only, CUDA only
    stem_vocal_batch: int = field(default_factory=lambda: _int("AUDIO_STEM_VOCAL_BATCH", 4))  # native only: windows per forward
    # roformer only: native multi-stem BS-RoFormer checkpoint + its training YAML.
    stem_roformer_ckpt: str = field(default_factory=lambda: os.getenv("AUDIO_STEM_ROFORMER_CKPT", ""))
    stem_roformer_config: str = field(default_factory=lambda: os.getenv("AUDIO_STEM_ROFORMER_CONFIG", ""))
    stem_roformer_overlap: int = field(default_factory=lambda: _int("AUDIO_STEM_ROFORMER_OVERLAP", 2))
    stem_roformer_fp16: bool = field(default_factory=lambda: _bool("AUDIO_STEM_ROFORMER_FP16", False))
    stem_roformer_batch: int = field(default_factory=lambda: _int("AUDIO_STEM_ROFORMER_BATCH", 1))
    # Demucs apply_model knobs: random-shift averaging (cost x shifts), segment overlap, and
    # segment length in seconds (unset = the model's own; htdemucs caps it at ~7.8 s).
    stem_shifts: int = field(default_factory=lambda: _int("AUDIO_STEM_SHIFTS", 1))
    stem_overlap: float = field(default_factory=lambda: _float("AUDIO_STEM_OVERLAP", 0.25))
    stem_segment: float | None = field(default_factory=lambda: _opt_float("AUDIO_STEM_SEGMENT"))
    # Demucs segments per forward. 1 = Demucs's own apply_model loop; > 1 batches the segments
    # through stems/overlap.py (single-model bags such as htdemucs, shifts 0; ~2x faster on an A40).
    stem_demucs_batch: int = field(default_factory=lambda: _int("AUDIO_STEM_DEMUCS_BATCH", 1))
    # options.mode when a request does not set one: auto | prompted | music_stems.
    routing_default: str = field(default_factory=lambda: os.getenv("AUDIO_ROUTING_DEFAULT", "auto"))
    # Content classifier for auto routing: off | mock | panns | ast. Unset -> "panns" when a real
    # splitter is configured (demucs/chain/roformer), otherwise "off" (see __post_init__).
    router_classifier: str = field(default_factory=lambda: os.getenv("AUDIO_ROUTER_CLASSIFIER", ""))
    # Minimum classifier "song" score (median over 10 s windows of AudioSet Music - Speech) for
    # auto routing to send stem prompts to the splitter. Unset -> per classifier, picked on the
    # labelled router set: panns 0.4 (songs 0.75-0.83, speech over music <= 0.34), ast 0.2
    # (songs 0.40-0.57, speech over music <= 0.17); 0.4 for off/mock.
    router_music_threshold: float | None = field(default_factory=lambda: _opt_float("AUDIO_ROUTER_MUSIC_THRESHOLD"))
    # PANNs Cnn14 checkpoint; unset -> ~/panns_data/Cnn14_mAP=0.431.pth (downloaded if missing).
    router_panns_checkpoint: str | None = field(default_factory=lambda: os.getenv("AUDIO_ROUTER_PANNS_CHECKPOINT") or None)
    # Instrument tags and descriptive names for music stems ("Other" -> "Synth & Strings"),
    # from the router's AudioSet tagger (labeling.py): auto | off. auto labels whenever the
    # router classifier can tag (panns, ast); otherwise tags are null and "other" stays "Other".
    stem_labels: str = field(default_factory=lambda: os.getenv("AUDIO_STEM_LABELS", "auto"))
    # Minimum mean probability for a tag. Unset -> per classifier (see __post_init__).
    stem_label_min_score: float | None = field(default_factory=lambda: _opt_float("AUDIO_STEM_LABEL_MIN_SCORE"))
    # 10 s windows analysed per stem (active regions only, spread across the song).
    stem_label_max_windows: int = field(default_factory=lambda: _int("AUDIO_STEM_LABEL_MAX_WINDOWS", 12))

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
        if self.backend == 'audiosep' and self.model_id == 'facebook/sam-audio-large' and 'AUDIO_MODEL_ID' not in os.environ:
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
        if not self.router_classifier:
            self.router_classifier = "panns" if self.stem_splitter in {"demucs", "chain", "roformer"} else "off"
        if self.router_music_threshold is None:
            self.router_music_threshold = {"panns": 0.4, "ast": 0.2}.get(self.router_classifier, 0.4)
        if self.stem_label_min_score is None:
            self.stem_label_min_score = {"panns": 0.1, "ast": 0.1}.get(self.router_classifier, 0.1)
        if not self.refine:
            # Measured 2026-11-13 on sam-audio-small: the mask improved every fixture (+1.8 to
            # +5.9 dB SI-SDRi, 1.7-6.5 dB less leakage) at no GPU cost, so it is on for the real
            # model. The mock keeps exact band-pass stems so tests and fixtures stay bit-exact.
            self.refine = "mask" if self.backend == "sam" else "off"

    def validate(self) -> None:
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
            raise ValueError("AUDIO_PASS_MAX_RATIO_DB must be within [-60, 20] (20 effectively disables the guard)")
        if not 0 <= self.pass_min_similarity <= 1:
            raise ValueError("AUDIO_PASS_MIN_SIMILARITY must be within [0, 1]")
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
        if self.stem_splitter not in {"off", "mock", "demucs", "chain", "roformer"}:
            raise ValueError("AUDIO_STEM_SPLITTER must be 'off', 'mock', 'demucs', 'chain' or 'roformer'")
        if self.stem_splitter == "roformer" and not (self.stem_roformer_ckpt and self.stem_roformer_config):
            raise ValueError("AUDIO_STEM_SPLITTER=roformer needs AUDIO_STEM_ROFORMER_CKPT and AUDIO_STEM_ROFORMER_CONFIG")
        if not 1 <= self.stem_vocal_overlap <= 16:
            raise ValueError("AUDIO_STEM_VOCAL_OVERLAP must be between 1 and 16")
        if self.stem_vocal_backend not in {"separator", "native"}:
            raise ValueError("AUDIO_STEM_VOCAL_BACKEND must be 'separator' or 'native'")
        if (self.stem_splitter == "chain" and self.stem_vocal_backend == "native" and not self.stem_model_dir
                and not Path(self.stem_vocal_model).is_absolute()):
            raise ValueError("AUDIO_STEM_VOCAL_BACKEND=native needs AUDIO_STEM_MODEL_DIR or an absolute AUDIO_STEM_VOCAL_MODEL path")
        if not 1 <= self.stem_demucs_batch <= 16:
            raise ValueError("AUDIO_STEM_DEMUCS_BATCH must be between 1 and 16")
        if not 1 <= self.stem_vocal_batch <= 16:
            raise ValueError("AUDIO_STEM_VOCAL_BATCH must be between 1 and 16")
        if not 1 <= self.stem_roformer_overlap <= 16:
            raise ValueError("AUDIO_STEM_ROFORMER_OVERLAP must be between 1 and 16")
        if not 1 <= self.stem_roformer_batch <= 16:
            raise ValueError("AUDIO_STEM_ROFORMER_BATCH must be between 1 and 16")
        if not self.stem_model:
            raise ValueError("AUDIO_STEM_MODEL must not be empty")
        if not 0 <= self.stem_shifts <= 20:
            raise ValueError("AUDIO_STEM_SHIFTS must be between 0 and 20")
        if not 0 <= self.stem_overlap < 1:
            raise ValueError("AUDIO_STEM_OVERLAP must be in [0, 1)")
        if self.stem_segment is not None and self.stem_segment <= 0:
            raise ValueError("AUDIO_STEM_SEGMENT must be positive")
        if self.routing_default not in {"auto", "prompted", "music_stems"}:
            raise ValueError("AUDIO_ROUTING_DEFAULT must be 'auto', 'prompted' or 'music_stems'")
        if self.routing_default == "music_stems" and self.stem_splitter == "off":
            raise ValueError("AUDIO_ROUTING_DEFAULT=music_stems needs AUDIO_STEM_SPLITTER")
        if self.stem_labels not in {"auto", "off"}:
            raise ValueError("AUDIO_STEM_LABELS must be 'auto' or 'off'")
        if not 0 <= self.stem_label_min_score <= 1:
            raise ValueError("AUDIO_STEM_LABEL_MIN_SCORE must be in [0, 1]")
        if not 1 <= self.stem_label_max_windows <= 64:
            raise ValueError("AUDIO_STEM_LABEL_MAX_WINDOWS must be between 1 and 64")
        if self.router_classifier not in {"off", "mock", "ast", "panns"}:
            raise ValueError("AUDIO_ROUTER_CLASSIFIER must be 'off', 'mock', 'panns' or 'ast'")
        if not 0 <= self.router_music_threshold <= 1:
            raise ValueError("AUDIO_ROUTER_MUSIC_THRESHOLD must be in [0, 1]")


def load_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings
