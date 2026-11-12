"""SamAudioBackend against a fake ``sam_audio`` module.

Nothing real can run here: the checkpoints are gated, there is no CUDA, and ``sam_audio``
is not installed. So a fake package mirroring the verified upstream signatures is injected
into ``sys.modules`` and the tests assert on *what the backend asked the model to do* —
batch shapes, description casing, anchor tuples, masked-video polarity, output trimming,
config overrides. The fake deliberately returns outputs that are longer than the input,
exactly as the codec does (``ceil(T / 1920) * 1920``).
"""
from __future__ import annotations

import math
import subprocess
import sys
import threading
import time
import types

import numpy as np
import pytest

from conftest import have_ffmpeg

from sam_audio_service.backends.base import BackendOptions, BackendPrompt, VisualInput
from sam_audio_service.errors import InferenceFailure, MissingModel

torch = pytest.importorskip("torch", reason="the fake model builds real tensors")

HOP = 1920
SAMPLE_RATE = 48_000
FPS = 10.0
VIDEO_SIZE = 64


# --------------------------------------------------------------------------- the fake
class FakeBatch:
    """Mirrors ``sam_audio.processor.Batch`` as far as the backend touches it."""

    def __init__(self, audios, descriptions, anchors, masked_video, sizes, hop_length=HOP):
        self.audios = audios  # (B, 1, T_max), pad_sequence-padded like upstream
        self.descriptions = descriptions
        self.masked_video = masked_video
        self.hop_length = hop_length
        self.sizes = sizes  # per-item feature frames: ceil(T_i / hop_length)
        self.device = None
        self.process_anchors(anchors)

    def process_anchors(self, anchors):
        self.anchors = anchors

    def to(self, device):
        self.device = device
        return self


class FakeProcessor:
    audio_sampling_rate = SAMPLE_RATE
    audio_hop_length = HOP

    def __init__(self, model_id):
        self.model_id = model_id
        self.calls: list[dict] = []
        self.mask_calls: list[tuple] = []

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        return cls(model_name_or_path)

    def mask_videos(self, videos, masks):
        self.mask_calls.append((videos, masks))
        return [v * m.eq(0) for v, m in zip(videos, masks, strict=False)]  # upstream formula

    def __call__(self, descriptions, audios, anchors=None, masked_videos=None):
        assert len(descriptions) == len(audios)  # upstream asserts exactly these
        assert anchors is None or len(anchors) == len(audios)
        assert masked_videos is None or len(masked_videos) == len(audios)
        self.calls.append(
            {"descriptions": descriptions, "audios": audios, "anchors": anchors, "masked_videos": masked_videos}
        )
        wavs = [a.mean(0) if a.ndim == 2 else a.reshape(-1) for a in audios]  # upstream wav.mean(0)
        sizes = torch.tensor([math.ceil(max(w.size(-1), 1) / HOP) for w in wavs])
        padded = torch.nn.utils.rnn.pad_sequence(wavs, batch_first=True).unsqueeze(1)
        return FakeBatch(padded, descriptions, anchors, masked_videos, sizes)


class FakeSeparationResult:
    def __init__(self, target, residual):
        self.target, self.residual, self.noise = target, residual, None


class FakeSAMAudio:
    """Mirrors ``SAMAudio``: config overrides through ``from_pretrained``, list outputs."""

    revision = None
    load_calls: list[tuple[str, dict]] = []
    fail_with: BaseException | None = None

    def __init__(self, model_id, kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        self.sample_rate = SAMPLE_RATE
        self.audio_codec = types.SimpleNamespace(hop_length=HOP, sample_rate=SAMPLE_RATE)
        self.vision_encoder = types.SimpleNamespace(dim=1024, image_size=336)
        self.visual_ranker = None if kwargs.get("visual_ranker", "keep") is None else object()
        self.text_ranker = None if kwargs.get("text_ranker", "keep") is None else object()
        if kwargs.get("span_predictor", "pe-a-frame-large") is not None:
            self.span_predictor = object()
        self._param = torch.zeros(7)
        self.separate_calls: list[dict] = []
        self.moved: list[dict] = []
        self.predicted_spans = [["+", 0.25, 0.75]]
        # Concurrency watch: a shared torch.nn.Module must never be entered twice at once.
        self._guard = threading.Lock()
        self.active = self.max_active = 0
        self.dwell = 0.0

    @classmethod
    def from_pretrained(cls, model_name_or_path, **model_kwargs):
        cls.load_calls.append((model_name_or_path, dict(model_kwargs)))
        if cls.fail_with is not None:
            raise cls.fail_with
        return cls(model_name_or_path, model_kwargs)

    def eval(self):
        return self

    def to(self, **kwargs):
        self.moved.append(kwargs)
        return self

    def parameters(self):
        yield self._param

    def separate(self, batch, noise=None, ode_opt=None, reranking_candidates=1, predict_spans=False):
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.separate_calls.append(
                {"batch": batch, "noise": noise, "ode_opt": ode_opt, "batch_size": len(batch.descriptions),
                 "reranking_candidates": reranking_candidates, "predict_spans": predict_spans}
            )
        try:
            time.sleep(self.dwell)  # widen the window a second thread could slip into
            if predict_spans and hasattr(self, "span_predictor") and batch.anchors is None:
                batch.process_anchors([self.predicted_spans] * len(batch.descriptions))
            # Upstream decodes the padded batch then narrows each item to sizes[i] * hop:
            # feature frames * hop_length >= that item's own input length.
            targets = [torch.arange(int(n) * HOP, dtype=torch.float32) + 1000.0 * i for i, n in enumerate(batch.sizes)]
            return FakeSeparationResult(target=targets, residual=[t * -1.0 for t in targets])
        finally:
            with self._guard:
                self.active -= 1


@pytest.fixture
def fake_sam(monkeypatch):
    """Inject a fake ``sam_audio`` package and reset its recordings."""
    FakeSAMAudio.load_calls = []
    FakeSAMAudio.fail_with = None
    module = types.ModuleType("sam_audio")
    module.SAMAudio = FakeSAMAudio
    module.SAMAudioProcessor = FakeProcessor
    monkeypatch.setitem(sys.modules, "sam_audio", module)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    return module


@pytest.fixture
def backend(fake_sam):
    from sam_audio_service.backends.sam_audio import SamAudioBackend

    created = SamAudioBackend(model_id="facebook/sam-audio-large", device="cpu")
    yield created
    created.unload()


def mixture(seconds: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# --------------------------------------------------------------------------- lifecycle
def test_missing_sam_audio_raises_missing_model(monkeypatch):
    from sam_audio_service.backends.sam_audio import SamAudioBackend

    monkeypatch.setitem(sys.modules, "sam_audio", None)  # None makes `import sam_audio` fail
    with pytest.raises(MissingModel, match="sam_audio is not importable"):
        SamAudioBackend(model_id="facebook/sam-audio-large", device="cpu").load()


def test_gated_checkpoint_raises_missing_model(backend):
    FakeSAMAudio.fail_with = OSError("403 Client Error: access to model facebook/sam-audio-large is gated")
    try:
        with pytest.raises(MissingModel, match="gated"):
            backend.load()
    finally:
        FakeSAMAudio.fail_with = None
    assert backend.loaded is False


def test_load_is_idempotent_and_info_reports_facts(backend):
    backend.load()
    backend.load()
    assert len(FakeSAMAudio.load_calls) == 1
    info = backend.info()
    assert info["model_id"] == "facebook/sam-audio-large"
    assert info["device"] == "cpu" and info["dtype"] == "float32"
    assert info["parameters"] == 7 and info["loaded"] is True
    assert info["span_predictor_loaded"] is True
    assert info["visual_ranker_loaded"] is True and info["text_ranker_loaded"] is True
    assert info["processor_sample_rate"] == SAMPLE_RATE and info["hop_length"] == HOP
    assert info["torch_version"] == torch.__version__
    assert info["ode"] == {"method": "midpoint", "step_size": 2 / 32}
    assert info["mask_mode"] == "keep_object"
    assert "cuda_peak_allocated_bytes" not in info  # cpu run


def test_info_before_load_does_not_import_anything():
    from sam_audio_service.backends.sam_audio import SamAudioBackend

    info = SamAudioBackend(model_id="m", device="auto").info()
    assert info["loaded"] is False and info["device"] == "auto" and info["parameters"] is None


def test_unload_clears_the_model(backend):
    backend.load()
    backend.unload()
    assert backend.loaded is False and backend.info()["parameters"] is None


def test_config_overrides_are_forwarded_when_toggles_are_off(fake_sam):
    from sam_audio_service.backends.sam_audio import SamAudioBackend

    created = SamAudioBackend(
        model_id="facebook/sam-audio-small", device="cpu",
        load_span_predictor=False, load_rankers=False,
    )
    created.load()
    model_id, kwargs = FakeSAMAudio.load_calls[-1]
    assert model_id == "facebook/sam-audio-small"
    assert kwargs == {"span_predictor": None, "text_ranker": None, "visual_ranker": None}
    assert created.supports_span_prediction is False
    info = created.info()
    assert info["span_predictor_loaded"] is False
    assert info["visual_ranker_loaded"] is False and info["text_ranker_loaded"] is False


def test_no_overrides_when_everything_is_enabled(backend):
    backend.load()
    assert FakeSAMAudio.load_calls[-1][1] == {}
    assert backend.supports_span_prediction is True


def test_hf_token_is_forwarded_from_the_environment(fake_sam, monkeypatch):
    from sam_audio_service.backends.sam_audio import SamAudioBackend

    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    SamAudioBackend(model_id="m", device="cpu").load()
    assert FakeSAMAudio.load_calls[-1][1]["token"] == "hf_fake"


def test_device_auto_resolves_to_cpu_without_accelerators(fake_sam, monkeypatch):
    from sam_audio_service.backends.sam_audio import SamAudioBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    created = SamAudioBackend(model_id="m", device="auto")
    created.load()
    assert created.info()["device"] == "cpu"


def test_cuda_request_without_cuda_is_a_missing_model(fake_sam, monkeypatch):
    from sam_audio_service.backends.sam_audio import SamAudioBackend

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(MissingModel, match="no CUDA device"):
        SamAudioBackend(model_id="m", device="cuda").load()


def test_float16_is_refused(fake_sam):
    from sam_audio_service.backends.sam_audio import SamAudioBackend

    with pytest.raises(InferenceFailure, match="float16 is refused"):
        SamAudioBackend(model_id="m", device="cpu", dtype="float16").load()


def test_unknown_mask_mode_is_rejected_at_construction():
    from sam_audio_service.backends.sam_audio import SamAudioBackend

    with pytest.raises(ValueError, match="mask_mode"):
        SamAudioBackend(model_id="m", mask_mode="invert")  # type: ignore[arg-type]


# -------------------------------------------------------------------- text prompting
def test_text_prompt_builds_the_batch(backend):
    signal = mixture(0.5)
    result = backend.separate(signal, BackendPrompt(text="A Dog Barking"), BackendOptions())
    call = backend._processor.calls[-1]
    assert call["descriptions"] == ["a dog barking"]  # lowercased
    assert call["anchors"] is None and call["masked_videos"] is None
    audio = call["audios"][0]
    assert audio.shape == (1, len(signal)) and audio.dtype == torch.float32
    assert np.allclose(audio[0].numpy(), signal)
    separate = backend._model.separate_calls[-1]
    assert separate["predict_spans"] is False and separate["reranking_candidates"] == 1
    assert separate["ode_opt"] == {"method": "midpoint", "options": {"step_size": 2 / 32}}
    assert separate["batch"].device == torch.device("cpu")
    assert result.extra["description"] == "a dog barking"
    assert result.spans is None


def test_output_is_trimmed_to_the_input_length(backend):
    signal = mixture(0.5)  # 24000 samples -> codec emits 13 * 1920 = 24960
    result = backend.separate(signal, BackendPrompt(text="siren"), BackendOptions())
    assert len(signal) % HOP != 0, "the test needs a length the codec has to pad"
    assert result.target.shape == (len(signal),) and result.target.dtype == np.float32
    assert result.residual.shape == (len(signal),)
    assert np.allclose(result.residual, -result.target)


def test_anchors_are_passed_as_token_tuples(backend):
    result = backend.separate(
        mixture(2.0),
        BackendPrompt(text="siren", anchors=[("+", 0.5, 1.0), ("-", 1.5, 1.8)]),
        BackendOptions(),
    )
    assert backend._processor.calls[-1]["anchors"] == [[("+", 0.5, 1.0), ("-", 1.5, 1.8)]]
    # Upstream skips span prediction when anchors are supplied; the caller's spans stand.
    assert result.spans is None


def test_predicted_spans_come_from_the_batch(backend):
    result = backend.separate(mixture(0.5), BackendPrompt(text="siren"), BackendOptions(predict_spans=True))
    assert backend._model.separate_calls[-1]["predict_spans"] is True
    assert result.spans == [(0.25, 0.75)]
    assert result.extra["predict_spans"] is True


def test_span_prediction_is_skipped_when_anchors_are_given(backend):
    backend.separate(
        mixture(0.5),
        BackendPrompt(text="siren", anchors=[("+", 0.0, 0.2)]),
        BackendOptions(predict_spans=True),
    )
    assert backend._model.separate_calls[-1]["predict_spans"] is False


def test_span_prediction_is_skipped_without_a_span_predictor(fake_sam):
    from sam_audio_service.backends.sam_audio import SamAudioBackend

    created = SamAudioBackend(model_id="m", device="cpu", load_span_predictor=False)
    created.separate(mixture(0.2), BackendPrompt(text="siren"), BackendOptions(predict_spans=True))
    assert created._model.separate_calls[-1]["predict_spans"] is False


def test_reranking_candidates_and_seed_are_honoured(backend):
    backend.separate(mixture(0.2), BackendPrompt(text="siren"), BackendOptions(reranking_candidates=8, seed=1234))
    assert backend._model.separate_calls[-1]["reranking_candidates"] == 8
    seeded = torch.initial_seed()
    backend.separate(mixture(0.2), BackendPrompt(text="siren"), BackendOptions(seed=1234))
    assert torch.initial_seed() == seeded == 1234


def test_empty_mixture_returns_empty_without_calling_the_model(backend):
    result = backend.separate(np.zeros(0, dtype=np.float32), BackendPrompt(text="siren"), BackendOptions())
    assert result.target.shape == (0,) and result.residual.shape == (0,)
    assert backend._model is None or not backend._model.separate_calls


def test_oom_becomes_insufficient_gpu_memory(backend, monkeypatch):
    from sam_audio_service.errors import InsufficientGpuMemory

    backend.load()

    def boom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")

    monkeypatch.setattr(backend._model, "separate", boom)
    with pytest.raises(InsufficientGpuMemory, match="ran out of memory"):
        backend.separate(mixture(0.2), BackendPrompt(text="siren"), BackendOptions())


def test_other_model_errors_propagate_for_the_engine_to_classify(backend, monkeypatch):
    backend.load()
    monkeypatch.setattr(backend._model, "separate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("shape mismatch")))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        backend.separate(mixture(0.2), BackendPrompt(text="siren"), BackendOptions())


# ----------------------------------------------------------------- batched windows
# Regression cover for the PR #18 failure: windows used to run on threads against this
# shared model and raced `WatermarkEncoderBlock.forward_no_conv`, which swaps a conv for
# `nn.Identity()` mid-forward. See docs/chunk_concurrency.md.
def windows(*seconds: float) -> list[np.ndarray]:
    return [mixture(s) for s in seconds]


def test_separate_many_builds_one_processor_call_for_k_windows(backend):
    mixtures = windows(0.5, 0.25, 0.5)
    prompts = [
        BackendPrompt(text="A Dog Barking", anchors=[("+", 0.0, 0.2)]),
        BackendPrompt(text="siren", anchors=[("-", 0.1, 0.2)]),
        BackendPrompt(text="siren", anchors=[("+", 0.3, 0.5)]),
    ]
    results = backend.separate_many(mixtures, prompts, BackendOptions())

    assert len(backend._processor.calls) == 1, "k windows must be ONE processor batch"
    assert len(backend._model.separate_calls) == 1, "k windows must be ONE model call"
    call = backend._processor.calls[0]
    assert call["descriptions"] == ["a dog barking", "siren", "siren"]
    assert [tuple(a.shape) for a in call["audios"]] == [(1, len(m)) for m in mixtures]
    assert call["anchors"] == [[("+", 0.0, 0.2)], [("-", 0.1, 0.2)], [("+", 0.3, 0.5)]]
    assert call["masked_videos"] is None
    assert backend._model.separate_calls[0]["batch_size"] == 3
    # pad_sequence pads the short window up to the longest, so each output is trimmed to
    # ITS OWN input length -- and item i keeps item i's result (offset 1000 * i).
    assert [len(r.target) for r in results] == [len(m) for m in mixtures]
    assert [float(r.target[0]) for r in results] == [0.0, 1000.0, 2000.0]
    assert all(np.allclose(r.residual, -r.target) for r in results)
    assert [r.extra["batch"] for r in results] == [3, 3, 3]


def test_separate_is_separate_many_with_one_window(backend):
    result = backend.separate(mixture(0.3), BackendPrompt(text="siren"), BackendOptions())
    assert backend._model.separate_calls[-1]["batch_size"] == 1
    assert len(result.target) == len(mixture(0.3))


def test_anchored_and_unanchored_windows_are_not_mixed(backend):
    """``anchors=None`` is the batch-wide "predict spans" gate upstream, so a batch is
    either all-anchored or all-unanchored."""
    backend.separate_many(
        windows(0.2, 0.2, 0.2),
        [BackendPrompt(text="siren", anchors=[("+", 0.0, 0.1)]), BackendPrompt(text="siren"), BackendPrompt(text="siren")],
        BackendOptions(),
    )
    assert [c["anchors"] is None for c in backend._processor.calls] == [False, True]
    assert [len(c["audios"]) for c in backend._processor.calls] == [1, 2]


def test_separate_many_preserves_input_order_when_groups_interleave(backend):
    """Grouping splits the windows into several model calls; the results must still come
    back in INPUT order, or the engine's crossfade merges the wrong window."""
    mixtures = windows(0.5, 0.25, 0.3, 0.4)
    prompts = [
        BackendPrompt(text="first", anchors=[("+", 0.0, 0.2)]),
        BackendPrompt(text="second"),
        BackendPrompt(text="third", anchors=[("+", 0.1, 0.2)]),
        BackendPrompt(text="fourth", anchors=[("-", 0.0, 0.1)]),
    ]
    results = backend.separate_many(mixtures, prompts, BackendOptions())

    assert [len(c["audios"]) for c in backend._processor.calls] == [1, 1, 2]  # A | U | A A
    assert [r.extra["description"] for r in results] == ["first", "second", "third", "fourth"]
    assert [len(r.target) for r in results] == [len(m) for m in mixtures]


def test_predicted_spans_are_read_per_batch_item(backend):
    results = backend.separate_many(
        windows(0.2, 0.2), [BackendPrompt(text="siren")] * 2, BackendOptions(predict_spans=True)
    )
    assert [r.spans for r in results] == [[(0.25, 0.75)], [(0.25, 0.75)]]


def test_separate_many_rejects_a_prompt_count_mismatch(backend):
    with pytest.raises(ValueError, match="one prompt per mixture"):
        backend.separate_many(windows(0.2, 0.2), [BackendPrompt(text="siren")], BackendOptions())


def test_visual_and_empty_windows_are_never_batched():
    """Grouping is a pure function, so assert it without a GPU, model or video file."""
    from sam_audio_service.backends.sam_audio import _batchable_groups

    text, visual = BackendPrompt(text="siren"), BackendPrompt(text="siren", visual=object())  # type: ignore[arg-type]
    full, empty = np.zeros(10, dtype=np.float32), np.zeros(0, dtype=np.float32)
    assert _batchable_groups([full] * 3, [text] * 3) == [[0, 1, 2]]
    assert _batchable_groups([full] * 3, [text, visual, text]) == [[0], [1], [2]]
    assert _batchable_groups([full, empty, full], [text] * 3) == [[0], [1], [2]]
    anchored = BackendPrompt(text="siren", anchors=[("+", 0.0, 0.1)])
    # Interleaved groups must stay consecutive runs of ascending indices, so that
    # concatenating the groups' results reproduces the input order exactly.
    assert _batchable_groups([full] * 4, [anchored, text, anchored, anchored]) == [[0], [1], [2, 3]]
    assert _batchable_groups([full] * 3, [text, anchored, text]) == [[0], [1], [2]]


def test_concurrent_separate_calls_never_enter_the_model_together(backend):
    """The engine batches instead of threading, but the backend lock is the guarantee:
    whatever calls it, a shared torch.nn.Module is entered one call at a time."""
    backend.load()
    backend._model.dwell = 0.05  # make a racing thread's overlap unmissable
    results: list[object] = []
    errors: list[BaseException] = []

    def work():
        try:
            results.append(backend.separate(mixture(0.2), BackendPrompt(text="siren"), BackendOptions()))
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors and len(results) == 2
    assert backend._model.max_active == 1, "two threads were inside model.separate at once"
    assert len(backend._model.separate_calls) == 2


def test_batched_oom_reports_the_batch_size(backend, monkeypatch):
    from sam_audio_service.errors import InsufficientGpuMemory

    backend.load()
    monkeypatch.setattr(
        backend._model, "separate",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB")),
    )
    with pytest.raises(InsufficientGpuMemory, match="AUDIO_CHUNK_BATCH") as caught:
        backend.separate_many(windows(0.2, 0.2), [BackendPrompt(text="siren")] * 2, BackendOptions())
    assert caught.value.details["batch"] == 2


# ------------------------------------------------- OOM variants under memory pressure
# A full GPU does not always say "out of memory": the allocation failure surfaces through
# whichever library asked for the block first. Every one of these is retryable at
# AUDIO_CHUNK_BATCH=1, so all of them must be classified as INSUFFICIENT_GPU_MEMORY (the
# engine's batch-1 retry trigger) rather than INFERENCE_FAILURE, which kills the request.
class FakeTorchOom(torch.OutOfMemoryError):
    """A torch.OutOfMemoryError *subclass* whose text says nothing about memory."""


OOM_VARIANTS = [
    pytest.param(lambda: RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB"), id="cuda-oom"),
    pytest.param(
        lambda: RuntimeError("CUDA error: CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate(handle)"),
        id="cublas-alloc-failed",
    ),
    pytest.param(lambda: RuntimeError("cuDNN error: CUDNN_STATUS_ALLOC_FAILED"), id="cudnn-alloc-failed"),
    pytest.param(
        lambda: RuntimeError("CUDA error: an illegal memory access was encountered"),
        id="illegal-memory-access",
    ),
    pytest.param(lambda: RuntimeError("CUDA driver error: cudaErrorMemoryAllocation"), id="cuda-malloc-failed"),
    pytest.param(lambda: torch.OutOfMemoryError("Allocation on device failed"), id="torch-oom-error"),
    pytest.param(lambda: FakeTorchOom("Allocation on device failed"), id="torch-oom-subclass"),
]


def raising(make_exc):
    def boom(*args, **kwargs):
        raise make_exc()

    return boom


@pytest.mark.parametrize("make_exc", OOM_VARIANTS)
def test_every_oom_variant_becomes_insufficient_gpu_memory(backend, monkeypatch, make_exc):
    from sam_audio_service.errors import InsufficientGpuMemory

    backend.load()
    freed: list[int] = []
    monkeypatch.setattr(backend, "_release", lambda torch_module: freed.append(1))
    monkeypatch.setattr(backend._model, "separate", raising(make_exc))
    with pytest.raises(InsufficientGpuMemory, match="AUDIO_CHUNK_BATCH") as caught:
        backend.separate_many(windows(0.2, 0.2), [BackendPrompt(text="siren")] * 2, BackendOptions())
    assert caught.value.details["batch"] == 2
    assert freed, "the batch must be released before the error is re-raised"


@pytest.mark.parametrize("make_exc", OOM_VARIANTS)
def test_oom_while_moving_the_batch_to_the_device_is_classified_and_freed(backend, monkeypatch, make_exc):
    """``processor(...).to(device)`` allocates k windows of audio before the model is ever
    entered, so an OOM there must free the batch and classify the same way."""
    from sam_audio_service.errors import InsufficientGpuMemory

    backend.load()
    freed: list[int] = []
    monkeypatch.setattr(backend, "_release", lambda torch_module: freed.append(1))
    monkeypatch.setattr(FakeBatch, "to", raising(make_exc))
    with pytest.raises(InsufficientGpuMemory, match="AUDIO_CHUNK_BATCH"):
        backend.separate_many(windows(0.2, 0.2), [BackendPrompt(text="siren")] * 2, BackendOptions())
    assert freed, "the half-built batch must be released before the error is re-raised"
    assert not backend._model.separate_calls


def test_a_real_bug_is_still_an_inference_failure(backend, monkeypatch):
    """The widened classification must not swallow genuine errors."""
    backend.load()
    monkeypatch.setattr(backend._model, "separate", raising(lambda: RuntimeError("shape mismatch")))
    with pytest.raises(RuntimeError, match="shape mismatch"):
        backend.separate_many(windows(0.2, 0.2), [BackendPrompt(text="siren")] * 2, BackendOptions())


@pytest.mark.parametrize("make_exc", OOM_VARIANTS)
def test_engine_retries_the_whole_signal_at_batch_one_for_every_oom_variant(backend, monkeypatch, make_exc):
    """End to end: the real backend raises each variant on a batch of 2 and the engine must
    fall back to batch 1 rather than failing the request."""
    from sam_audio_service.audio_io import DecodedAudio
    from sam_audio_service.config import Settings
    from sam_audio_service.contracts import SeparationMetadata
    from sam_audio_service.engine import Engine

    backend.load()
    real_separate = backend._model.separate
    attempts: list[int] = []

    def separate(batch, **kwargs):
        attempts.append(len(batch.descriptions))
        if len(batch.descriptions) > 1:
            raise make_exc()
        return real_separate(batch, **kwargs)

    monkeypatch.setattr(backend._model, "separate", separate)

    settings = Settings(chunk_batch=2, chunk_overlap_seconds=0.25)
    engine = Engine(settings, backend)
    signal = mixture(1.6)[None, :]
    metadata = SeparationMetadata.model_validate({"prompts": ["siren"], "options": {"chunk_seconds": 1.0}})
    result = engine.separate(DecodedAudio(signal, SAMPLE_RATE, SAMPLE_RATE, 1), metadata)

    assert any("retried sequentially" in w for w in result.warnings)
    assert attempts == [2, 1, 1], "the failed batch of 2, then the whole signal at batch 1"
    assert result.stems[0].samples.shape == signal.shape


# ------------------------------------------------------------------ visual prompting
requires_ffmpeg = pytest.mark.skipif(not have_ffmpeg(), reason="synthetic video needs ffmpeg")


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=size={VIDEO_SIZE}x{VIDEO_SIZE}:rate={FPS:g}:duration=1",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return str(path)


@pytest.fixture
def mask_video(tmp_path):
    path = tmp_path / "mask.mkv"
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"color=c=white:s={VIDEO_SIZE}x{VIDEO_SIZE}:r={FPS:g}:d=1",
         "-c:v", "ffv1", "-pix_fmt", "gray", str(path)],
        check=True, capture_output=True,
    )
    return str(path)


def _half_mask():
    mask = np.zeros((VIDEO_SIZE, VIDEO_SIZE), np.uint8)
    mask[: VIDEO_SIZE // 2] = 1  # top half is the object
    return mask


@requires_ffmpeg
@pytest.mark.parametrize("mask_mode", ["keep_object", "notebook"])
def test_visual_prompt_polarity(fake_sam, video, mask_mode):
    from sam_audio_service.backends.sam_audio import SamAudioBackend

    created = SamAudioBackend(model_id="m", device="cpu", mask_mode=mask_mode)
    visual = VisualInput(
        video_path=video, interval=(0.0, 1.0), frame_rate=FPS,
        frame_masks={i: _half_mask() for i in range(10)}, label="Dog",
    )
    result = created.separate(mixture(1.0), BackendPrompt(text="a dog", visual=visual), BackendOptions())

    frames, masks = created._processor.mask_calls[-1]
    assert frames[0].shape == (10, 3, 336, 336) and frames[0].dtype == torch.uint8
    assert masks[0].shape == (10, 1, 336, 336)
    # mask_videos keeps pixels where the mask is 0.
    object_half, background_half = slice(0, 168), slice(168, 336)
    if mask_mode == "keep_object":
        assert not masks[0][:, 0, object_half].any(), "object must be kept"
        assert masks[0][:, 0, background_half].all(), "background must be zeroed"
    else:
        assert masks[0][:, 0, object_half].all(), "notebook mode zeroes the object"
        assert not masks[0][:, 0, background_half].any()

    masked = created._processor.calls[-1]["masked_videos"][0]
    kept_rows = masked[:, :, object_half].abs().sum() > 0
    assert bool(kept_rows) is (mask_mode == "keep_object")
    assert result.extra["mask_mode"] == mask_mode and result.extra["visual_frames"] == 10
    created.unload()


@requires_ffmpeg
def test_visual_prompt_from_a_mask_video(fake_sam, video, mask_video):
    from sam_audio_service.backends.sam_audio import SamAudioBackend

    created = SamAudioBackend(model_id="m", device="cpu")
    visual = VisualInput(video_path=video, interval=(0.0, 1.0), mask_video_path=mask_video, frame_rate=FPS)
    created.separate(mixture(1.0), BackendPrompt(text="dog", visual=visual), BackendOptions())
    frames, masks = created._processor.mask_calls[-1]
    assert frames[0].shape[0] == masks[0].shape[0] == 10
    assert not masks[0].any()  # white mask video + keep_object => keep everything
    created.unload()


@requires_ffmpeg
def test_visual_interval_selects_only_those_frames(fake_sam, video):
    from sam_audio_service.backends.sam_audio import SamAudioBackend

    created = SamAudioBackend(model_id="m", device="cpu")
    visual = VisualInput(
        video_path=video, interval=(0.3, 0.7), frame_rate=FPS,
        frame_masks={3: _half_mask(), 4: _half_mask(), 5: _half_mask(), 6: _half_mask()},
    )
    # The engine hands over a mixture cropped to the same interval.
    created.separate(mixture(0.4), BackendPrompt(text="dog", visual=visual), BackendOptions())
    frames, _ = created._processor.mask_calls[-1]
    assert frames[0].shape[0] == 4
    created.unload()


@requires_ffmpeg
def test_visual_only_prompt_falls_back_to_the_label(fake_sam, video):
    from sam_audio_service.backends.sam_audio import SamAudioBackend

    created = SamAudioBackend(model_id="m", device="cpu")
    visual = VisualInput(
        video_path=video, interval=(0.0, 1.0), frame_rate=FPS,
        frame_masks={i: _half_mask() for i in range(10)}, label="Acoustic Guitar",
    )
    created.separate(mixture(1.0), BackendPrompt(text="", visual=visual), BackendOptions())
    assert created._processor.calls[-1]["descriptions"] == ["acoustic guitar"]
    created.unload()


@requires_ffmpeg
def test_frame_size_setting_is_applied(fake_sam, video):
    from sam_audio_service.backends.sam_audio import SamAudioBackend

    created = SamAudioBackend(model_id="m", device="cpu", frame_size=112)
    visual = VisualInput(
        video_path=video, interval=(0.0, 1.0), frame_rate=FPS,
        frame_masks={i: _half_mask() for i in range(10)},
    )
    created.separate(mixture(1.0), BackendPrompt(text="dog", visual=visual), BackendOptions())
    frames, masks = created._processor.mask_calls[-1]
    assert frames[0].shape == (10, 3, 112, 112) and masks[0].shape == (10, 1, 112, 112)
    created.unload()


# ------------------------------------------------------------------------- the factory
def test_factory_wires_settings_into_the_backend(monkeypatch, fake_sam):
    monkeypatch.setenv("AUDIO_BACKEND", "sam")
    monkeypatch.setenv("AUDIO_VISUAL_MASK_MODE", "notebook")
    monkeypatch.setenv("AUDIO_LOAD_SPAN_PREDICTOR", "false")
    monkeypatch.setenv("AUDIO_LOAD_RANKERS", "false")
    monkeypatch.setenv("AUDIO_ODE_STEP_SIZE", "0.125")
    monkeypatch.setenv("AUDIO_FRAME_SIZE", "224")
    monkeypatch.setenv("AUDIO_DEVICE", "cpu")
    monkeypatch.delenv("AUDIO_CHUNK_SECONDS", raising=False)
    from sam_audio_service.backends import create_backend
    from sam_audio_service.config import load_settings

    settings = load_settings()
    assert settings.chunk_seconds == 20.0  # sam default when AUDIO_CHUNK_SECONDS is unset
    created = create_backend(settings)
    assert created.mask_mode == "notebook" and created.frame_size == 224
    assert created.load_span_predictor is False and created.load_rankers is False
    assert created.ode_opt == {"method": "midpoint", "options": {"step_size": 0.125}}


def test_mock_backend_keeps_chunking_disabled_by_default(monkeypatch):
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    monkeypatch.delenv("AUDIO_CHUNK_SECONDS", raising=False)
    from sam_audio_service.config import load_settings

    assert load_settings().chunk_seconds == 0.0


def test_explicit_chunk_seconds_wins(monkeypatch):
    monkeypatch.setenv("AUDIO_BACKEND", "sam")
    monkeypatch.setenv("AUDIO_CHUNK_SECONDS", "30")
    from sam_audio_service.config import load_settings

    assert load_settings().chunk_seconds == 30.0
