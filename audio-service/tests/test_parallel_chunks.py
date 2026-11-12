"""Chunk windows are BATCHED, never threaded.

PR #18 ran overlapping windows on a ThreadPoolExecutor against one shared model; see
``docs/chunk_concurrency.md`` for why that is unsafe. These tests pin the replacement:
the engine collects up to ``AUDIO_CHUNK_BATCH`` windows, hands them to the backend in a
single ``separate_many`` call, never enters the backend concurrently, falls back to batch 1
on OOM, and still reassembles the timeline sample-accurately.
"""
import threading

import numpy as np
import pytest

from sam_audio_service.audio_io import DecodedAudio
from sam_audio_service.backends.base import Backend, BackendResult
from sam_audio_service.config import Settings
from sam_audio_service.contracts import SeparationMetadata
from sam_audio_service.engine import Engine
from sam_audio_service.errors import InsufficientGpuMemory

SR = 100
FRAMES = 337


class TrackingBackend(Backend):
    """Records what the engine asked for: batch sizes, per-window lengths, overlap."""

    is_mock = True
    sample_rate = SR

    def __init__(self, fail_batch: bool = False):
        self.lock = threading.Lock()
        self.active = self.peak = self.calls = 0
        self.batches: list[int] = []
        self.lengths: list[int] = []
        self.fail_batch = fail_batch

    def load(self): pass
    def info(self): return {}

    def separate_many(self, mixtures, prompts, options):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.batches.append(len(mixtures))
            fail = self.fail_batch and len(mixtures) > 1
        try:
            if fail:
                self.fail_batch = False
                raise InsufficientGpuMemory("simulated OOM")
            return super().separate_many(mixtures, prompts, options)
        finally:
            with self.lock:
                self.active -= 1

    def separate(self, mixture, prompt, options):
        with self.lock:
            self.calls += 1
            self.lengths.append(len(mixture))
        return BackendResult(target=mixture * 0.25, residual=mixture * 0.75)


def run(backend: Backend, *, batch: int = 2, seed: int | None = None, **kwargs):
    settings = Settings(chunk_overlap_seconds=0.25, **{"chunk_batch": batch, **kwargs})
    engine = Engine(settings, backend)
    signal = np.linspace(-1, 1, FRAMES, dtype=np.float32)[None, :]
    metadata = SeparationMetadata.model_validate({"prompts": ["voice"], "options": {"chunk_seconds": 1, "seed": seed}})
    return signal, engine.separate(DecodedAudio(signal, SR, SR, 1), metadata)


# Windows: size 100, step 75 over 337 frames -> [0,100) [75,175) [150,250) [225,325) [300,337).
EXPECTED_WINDOWS = [100, 100, 100, 100, 37]


def test_windows_are_batched_and_keep_sample_alignment():
    backend = TrackingBackend()
    signal, result = run(backend)
    assert backend.batches == [2, 2, 1]  # 5 windows collected two at a time
    assert backend.lengths == EXPECTED_WINDOWS
    assert backend.peak == 1  # the backend is never entered concurrently
    np.testing.assert_allclose(result.stems[0].samples, signal * 0.25, atol=1e-6)
    np.testing.assert_allclose(sum(s.samples for s in result.stems), signal, atol=1e-6)


@pytest.mark.parametrize(("batch", "expected"), [(1, [1] * 5), (3, [3, 2]), (8, [5])])
def test_chunk_batch_bounds_the_windows_per_model_call(batch, expected):
    backend = TrackingBackend()
    signal, result = run(backend, batch=batch)
    assert backend.batches == expected and backend.lengths == EXPECTED_WINDOWS
    np.testing.assert_allclose(result.stems[0].samples, signal * 0.25, atol=1e-6)


def test_seeded_windows_still_batch_and_warn_about_the_batch_size():
    """Batching a seeded request is reproducible: the seed is set before the single call,
    so there is no RNG race to avoid (only the batch size changes the draw)."""
    backend = TrackingBackend()
    _, result = run(backend, seed=7)
    assert backend.batches == [2, 2, 1]
    assert any("AUDIO_CHUNK_BATCH=2" in w for w in result.warnings)


def test_batch_oom_retries_with_batch_one_without_a_partial_mix():
    backend = TrackingBackend(fail_batch=True)
    signal, result = run(backend)
    assert backend.batches == [2, 1, 1, 1, 1, 1]  # failed batch, then the whole signal again
    assert any("retried sequentially" in w for w in result.warnings)
    np.testing.assert_allclose(result.stems[0].samples, signal * 0.25, atol=1e-6)
    np.testing.assert_allclose(sum(s.samples for s in result.stems), signal, atol=1e-6)


def test_oom_at_batch_one_is_not_retried():
    backend = TrackingBackend()
    backend.separate_many = lambda *a, **k: (_ for _ in ()).throw(InsufficientGpuMemory("simulated OOM"))
    with pytest.raises(InsufficientGpuMemory):
        run(backend, batch=1)


def test_chunk_workers_is_accepted_as_a_deprecated_alias(monkeypatch, caplog):
    monkeypatch.setenv("AUDIO_CHUNK_WORKERS", "4")
    monkeypatch.delenv("AUDIO_CHUNK_BATCH", raising=False)
    with caplog.at_level("WARNING"):
        settings = Settings()
    assert settings.chunk_batch == 4 and settings.chunk_workers == 4
    assert "AUDIO_CHUNK_WORKERS is deprecated" in caplog.text
    monkeypatch.setenv("AUDIO_CHUNK_BATCH", "2")
    assert Settings().chunk_batch == 2  # AUDIO_CHUNK_BATCH wins
