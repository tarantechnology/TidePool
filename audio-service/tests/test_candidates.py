"""Per-window candidates (``options.candidates``) with overlap-consistent selection.

The backend returns K noise draws per chunk window; the engine picks one per window in
timeline order (``candidate_select``: first | overlap | mean). ``JitterBackend`` makes exactly
one draw per window consistent with its neighbour (``+f*x``) and the others sign-flipped, so
the selection rule's effect on the stitched timeline is exactly predictable.
"""
import json
import threading

import numpy as np
import pytest

from sam_audio_service.audio_io import DecodedAudio
from sam_audio_service.backends.base import Backend, BackendOptions, BackendPrompt, BackendResult
from sam_audio_service.candidates import choose, choose_residual
from sam_audio_service.config import Settings
from sam_audio_service.contracts import SeparationMetadata
from sam_audio_service.engine import Engine
from sam_audio_service.errors import InsufficientGpuMemory

from test_parallel_chunks import EXPECTED_WINDOWS, FRAMES, SR, TrackingBackend

F = 0.25
STEP = 75


# --------------------------------------------------------------------- choose()
def _sig(n=200, seed=0):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def test_overlap_picks_the_candidate_consistent_with_the_previous_window():
    x = _sig()
    cands = [-x, x, 0.5 * _sig(seed=1)]
    idx, out = choose(cands, previous=x[:50], overlap=50, mode="overlap")
    assert idx == 1 and out is cands[1]


def test_overlap_clips_to_the_shorter_length():
    x = _sig()
    idx, _ = choose([-x[:30], x[:30]], previous=x[:50], overlap=50, mode="overlap")
    assert idx == 1


def test_silent_previous_falls_back_to_the_medoid_for_three_or_more():
    x = _sig()
    cands = [-x, x, 0.9 * x]  # the two positives agree; the medoid is one of them
    idx, _ = choose(cands, previous=np.zeros(50, np.float32), overlap=50, mode="overlap")
    assert idx in (1, 2)
    idx, _ = choose(cands, previous=np.full(50, 1e-5, np.float32), overlap=50, mode="overlap")  # -100 dBFS
    assert idx in (1, 2)


def test_silent_previous_with_two_candidates_takes_the_first():
    x = _sig()
    assert choose([-x, x], previous=np.zeros(50, np.float32), overlap=50, mode="overlap")[0] == 0


def test_no_previous_takes_the_medoid_or_the_first():
    x = _sig()
    assert choose([-x, x, x], previous=None, overlap=50, mode="overlap")[0] == 1
    assert choose([-x, x], previous=None, overlap=50, mode="overlap")[0] == 0


def test_mean_and_first():
    a, b = _sig(seed=1), _sig(seed=2)
    idx, out = choose([a, b], previous=a[:10], overlap=10, mode="mean")
    assert idx is None
    np.testing.assert_allclose(out, (a + b) / 2, atol=1e-6)
    idx, out = choose([a, b], previous=b[:10], overlap=10, mode="first")
    assert idx == 0 and out is a


def test_choose_residual():
    a, b = _sig(seed=1), _sig(seed=2)
    assert choose_residual([a, b], 1) is b
    np.testing.assert_allclose(choose_residual([a, b], None), (a + b) / 2, atol=1e-6)
    assert choose_residual([a, None], 0) is None
    assert choose_residual(None, 0) is None


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        choose([_sig(), _sig()], None, 10, "best")


# --------------------------------------------------------------------- engine
class JitterBackend(Backend):
    """K draws per window: draw ``w % 2`` is ``+F*x`` (consistent), every other draw ``-F*x``.

    Draw 0 does not depend on K, so ``candidate_select="first"`` must reproduce K = 1.
    """

    is_mock = True
    sample_rate = SR

    def __init__(self, fail_batch=False, ignore=False):
        self.lock = threading.Lock()
        self.batches: list[int] = []
        self.rows: list[int] = []
        self.fail_batch, self.ignore = fail_batch, ignore

    def load(self): pass
    def info(self): return {}
    def separate(self, mixture, prompt, options): raise AssertionError("separate_many is used")

    @staticmethod
    def window_index(mixture):
        start = int(round((float(mixture[0]) + 1) / 2 * (FRAMES - 1)))  # linspace(-1, 1) input
        return start // STEP

    def separate_many(self, mixtures, prompts, options):
        k = 1 if self.ignore else options.candidates
        with self.lock:
            self.batches.append(len(mixtures))
            self.rows.append(len(mixtures) * k)
            if self.fail_batch and len(mixtures) > 1:
                self.fail_batch = False
                raise InsufficientGpuMemory("simulated OOM")
        out = []
        for m in mixtures:
            good = self.window_index(m) % 2
            cands = [((F if j == good else -F) * m).astype(np.float32) for j in range(k)]
            res = [(m - c).astype(np.float32) for c in cands]
            if k == 1:
                out.append(BackendResult(target=cands[0], residual=res[0]))
            else:
                out.append(BackendResult(target=cands[0], residual=res[0], candidates=cands, candidate_residuals=res))
        return out


def run(backend, *, batch=2, **options):
    settings = Settings(chunk_overlap_seconds=0.25, chunk_batch=batch)
    engine = Engine(settings, backend)
    signal = np.linspace(-1, 1, FRAMES, dtype=np.float32)[None, :]
    metadata = SeparationMetadata.model_validate({"prompts": ["voice"], "options": {"chunk_seconds": 1, **options}})
    return signal, engine.separate(DecodedAudio(signal, SR, SR, 1), metadata)


def _reconstructs(signal, result):
    np.testing.assert_allclose(sum(s.samples.astype(np.float64) for s in result.stems), signal, atol=1e-6)


def test_overlap_selection_stitches_the_consistent_chain():
    backend = JitterBackend()
    signal, result = run(backend, candidates=2, candidate_select="overlap")
    np.testing.assert_allclose(result.stems[0].samples, F * signal, atol=1e-6)
    _reconstructs(signal, result)
    assert result.model["candidates"] == 2 and result.model["candidate_select"] == "overlap"


def test_overlap_is_the_default_selection():
    _, result = run(JitterBackend(), candidates=2)
    assert result.model["candidate_select"] == "overlap"


def test_mean_selection_averages_the_draws():
    signal, result = run(JitterBackend(), candidates=3, candidate_select="mean")
    np.testing.assert_allclose(result.stems[0].samples, -F * signal / 3, atol=1e-6)  # one +, two -
    _reconstructs(signal, result)


def test_first_selection_matches_one_candidate():
    _, one = run(JitterBackend(), candidates=1)
    _, first = run(JitterBackend(), candidates=2, candidate_select="first")
    np.testing.assert_allclose(first.stems[0].samples, one.stems[0].samples, atol=1e-7)
    assert one.model["candidates"] == 1 and one.model["candidate_select"] == "first"
    assert not np.allclose(one.stems[0].samples, F * np.linspace(-1, 1, FRAMES)[None, :], atol=1e-3)


def test_candidates_shrink_the_window_batch_so_rows_stay_bounded():
    backend = JitterBackend()
    run(backend, batch=2, candidates=2)
    assert backend.batches == [1] * 5 and max(backend.rows) <= 2
    backend = JitterBackend()
    run(backend, batch=4, candidates=2)
    assert backend.batches == [2, 2, 1] and max(backend.rows) <= 4


def test_oom_retry_with_candidates_still_stitches_correctly():
    backend = JitterBackend(fail_batch=True)
    signal, result = run(backend, batch=4, candidates=2)
    assert backend.batches == [2, 1, 1, 1, 1, 1]
    assert any("retried sequentially" in w for w in result.warnings)
    np.testing.assert_allclose(result.stems[0].samples, F * signal, atol=1e-6)
    _reconstructs(signal, result)


def test_candidate_choices_are_recorded_per_window():
    engine = Engine(Settings(chunk_overlap_seconds=0.25), JitterBackend())
    signal = np.linspace(-1, 1, FRAMES, dtype=np.float32)
    merged, chunked = engine._separate_signal(signal, BackendPrompt("voice"), BackendOptions(candidates=2, candidate_select="overlap"), 1, FRAMES / SR, 2)
    assert chunked and merged.extra["candidate_choices"] == [0, 1, 0, 1, 0]


def test_unchunked_window_selects_without_a_neighbour():
    backend = JitterBackend()
    signal, result = run(backend, candidates=3, candidate_select="overlap", chunk_seconds=0)
    # One window (index 0): draws [+, -, -]; no neighbour, so the medoid (a "-" draw) wins.
    np.testing.assert_allclose(result.stems[0].samples, -F * signal, atol=1e-6)
    _reconstructs(signal, result)


def test_backend_that_ignores_candidates_warns_and_is_unchanged():
    backend = TrackingBackend()
    signal, result = run(backend, candidates=2)
    assert sum("backend ignored options.candidates" in w for w in result.warnings) == 1
    np.testing.assert_allclose(result.stems[0].samples, signal * 0.25, atol=1e-6)
    assert backend.lengths == EXPECTED_WINDOWS


def test_candidates_setting_defaults_and_validation(monkeypatch):
    from sam_audio_service.config import load_settings

    settings = load_settings()
    assert settings.candidates == 1 and settings.candidate_select == "overlap"
    monkeypatch.setenv("AUDIO_CANDIDATES", "9")
    with pytest.raises(ValueError, match="AUDIO_CANDIDATES"):
        load_settings()
    monkeypatch.setenv("AUDIO_CANDIDATES", "2")
    monkeypatch.setenv("AUDIO_CANDIDATE_SELECT", "best")
    with pytest.raises(ValueError, match="AUDIO_CANDIDATE_SELECT"):
        load_settings()


@pytest.mark.parametrize("options", [{"candidates": 0}, {"candidates": 9}, {"candidate_select": "best"}])
def test_invalid_candidate_options_are_rejected(options):
    with pytest.raises(ValueError):
        SeparationMetadata.model_validate({"prompts": ["voice"], "options": options})


def test_separate_endpoint_accepts_candidates(client, wav_path):
    r = client.post(
        "/separate",
        files={"audio": ("audio.wav", wav_path.read_bytes(), "audio/wav")},
        data={"metadata": json.dumps({"prompts": ["dog"], "options": {"candidates": 2, "candidate_select": "mean"}})},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model"]["candidates"] == 2 and body["model"]["candidate_select"] == "mean"
    assert any("backend ignored options.candidates" in w for w in body["warnings"])  # the mock draws once
    r = client.post(
        "/separate",
        files={"audio": ("audio.wav", wav_path.read_bytes(), "audio/wav")},
        data={"metadata": json.dumps({"prompts": ["dog"], "options": {"candidates": 9}})},
    )
    assert r.status_code == 400
