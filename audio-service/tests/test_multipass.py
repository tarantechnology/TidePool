"""Multi-pass ("nested") separation: ``options.passes`` reruns each prompt on its residual.

SAM Audio sometimes only attenuates the described sound. Pass k >= 2 runs the SAME prompt
on ``mixture - sum(previous passes)`` and the extra targets are summed into the stem, so
``sum(stems) == mixture`` keeps holding. These tests pin the chain arithmetic, early stop,
chunking / stereo / anchors / visual-interval interplay and the response shape.
"""
import base64
import json

import numpy as np
import pytest

from sam_audio_service.audio_io import DecodedAudio, decode, encode_wav_pcm16
from sam_audio_service.backends.base import Backend, BackendResult, VisualInput
from sam_audio_service.backends.mock import MockBackend
from sam_audio_service.config import Settings
from sam_audio_service.contracts import SeparationMetadata
from sam_audio_service.engine import Engine
from sam_audio_service.errors import InvalidRequest

SR = 100


class FractionBackend(Backend):
    """Extracts a fixed fraction of whatever it is given and records every call.

    ``schedule`` overrides the fraction per call index (e.g. ``{1: 0.0}`` makes the second
    call silent), so tests can drive early stop deterministically.
    """

    is_mock = True
    supports_spans = True
    supports_visual = True
    sample_rate = SR

    def __init__(self, fraction: float = 0.5, schedule: dict[int, float] | None = None, spans=None):
        self.fraction, self.schedule, self.spans = fraction, schedule or {}, spans
        self.calls: list[dict] = []

    def load(self): pass
    def info(self): return {}

    def separate(self, mixture, prompt, options):
        f = self.schedule.get(len(self.calls), self.fraction)
        self.calls.append({"input": mixture.copy(), "text": prompt.text, "anchors": prompt.anchors, "visual": prompt.visual})
        spans = self.spans[len(self.calls) - 1] if self.spans else None
        return BackendResult(target=(mixture * f).astype(np.float32), residual=(mixture * (1 - f)).astype(np.float32), spans=spans)


def _signal(channels=1, frames=400, seed=0):
    return np.random.default_rng(seed).uniform(-0.5, 0.5, (channels, frames)).astype(np.float32)


def _run(backend, signal, *, settings=None, **meta):
    meta.setdefault("prompts", ["voice"])
    engine = Engine(settings or Settings(chunk_overlap_seconds=0.25), backend)
    return engine.separate(DecodedAudio(signal, SR, SR, signal.shape[0]), SeparationMetadata.model_validate(meta))


def _reconstructs(signal, result):
    total = sum(s.samples.astype(np.float64) for s in result.stems)
    np.testing.assert_allclose(total, signal, atol=1e-6)


def test_residual_chain_arithmetic_and_refinement():
    signal = _signal()
    backend = FractionBackend(0.5)
    result = _run(backend, signal, options={"passes": 3})
    target, residual = result.stems
    # 0.5 + 0.25 + 0.125 of the mixture; each pass ran on the previous residual.
    np.testing.assert_allclose(target.samples, signal * 0.875, atol=1e-6)
    np.testing.assert_allclose(residual.samples, signal * 0.125, atol=1e-6)
    np.testing.assert_allclose(backend.calls[1]["input"], signal[0] * 0.5, atol=1e-6)
    np.testing.assert_allclose(backend.calls[2]["input"], signal[0] * 0.25, atol=1e-6)
    _reconstructs(signal, result)
    ref = target.refinement
    assert ref["passes_requested"] == 3 and ref["passes_run"] == 3 and ref["stopped_early"] is False
    expected = [10 * np.log10(0.25), 10 * np.log10(0.0625), 10 * np.log10(0.015625)]
    np.testing.assert_allclose(ref["per_pass_energy_db"], expected, atol=1e-2)
    assert residual.refinement is None
    assert result.model["passes"] == 3


def test_default_is_one_pass():
    backend = FractionBackend(0.5)
    result = _run(backend, _signal())
    assert len(backend.calls) == 1
    assert result.stems[0].refinement == {"passes_requested": 1, "passes_run": 1, "per_pass_energy_db": [pytest.approx(10 * np.log10(0.25), abs=1e-2)], "stopped_early": False}
    assert result.model["passes"] == 1


def test_early_stop_when_a_pass_is_negligible():
    signal = _signal()
    backend = FractionBackend(0.5, schedule={1: 0.0})
    result = _run(backend, signal, options={"passes": 3})
    assert len(backend.calls) == 2  # pass 3 never ran
    assert "pass 2 extracted negligible energy; stopped after 1 pass(es)" in result.warnings
    ref = result.stems[0].refinement
    assert ref["passes_run"] == 1 and ref["stopped_early"] is True and len(ref["per_pass_energy_db"]) == 1
    np.testing.assert_allclose(result.stems[0].samples, signal * 0.5, atol=1e-6)
    _reconstructs(signal, result)
    assert result.model["passes"] == 1


def test_early_stop_threshold_is_minus_30_db_relative_to_pass_one():
    # Pass 2 extracts 0.5 * 0.02 of the mixture: (0.01 / 0.5)^2 = -34 dB re pass 1 -> stop.
    backend = FractionBackend(0.5, schedule={1: 0.02})
    assert _run(backend, _signal(), options={"passes": 2}).stems[0].refinement["stopped_early"] is True
    # 0.5 * 0.1 is -20 dB re pass 1 -> kept.
    backend = FractionBackend(0.5, schedule={1: 0.1})
    ref = _run(backend, _signal(), options={"passes": 2}).stems[0].refinement
    assert ref["passes_run"] == 2 and ref["stopped_early"] is False


def test_silent_first_pass_stops_without_running_more():
    backend = FractionBackend(0.0)
    result = _run(backend, _signal(), options={"passes": 3})
    assert len(backend.calls) == 1
    ref = result.stems[0].refinement
    assert ref["passes_run"] == 1 and ref["stopped_early"] is True and ref["per_pass_energy_db"] == [-120.0]
    assert any(w.startswith("pass 1 extracted negligible energy") for w in result.warnings)


def test_silent_residual_stops():
    signal = _signal()
    backend = FractionBackend(1.0)
    result = _run(backend, signal, options={"passes": 3})
    assert len(backend.calls) == 1
    assert result.stems[0].refinement["stopped_early"] is True
    _reconstructs(signal, result)


def test_each_prompt_has_its_own_residual_chain():
    signal = _signal()
    backend = FractionBackend(0.5)
    result = _run(backend, signal, prompts=["voice", "dog"], options={"passes": 2})
    texts = [c["text"] for c in backend.calls]
    assert texts == ["voice", "voice", "dog", "dog"]
    np.testing.assert_allclose(backend.calls[2]["input"], signal[0], atol=1e-6)  # prompt 2 starts from the mixture
    np.testing.assert_allclose(backend.calls[3]["input"], signal[0] * 0.5, atol=1e-6)
    _reconstructs(signal, result)  # residual = mixture - all cumulative targets


def test_anchors_apply_to_every_pass():
    backend = FractionBackend(0.5)
    _run(backend, _signal(), prompts=[{"text": "voice", "anchors": [{"token": "+", "start": 1.0, "end": 2.0}]}], options={"passes": 3})
    assert len(backend.calls) == 3
    assert all(c["anchors"] == [("+", 1.0, 2.0)] for c in backend.calls)


def test_spans_are_unioned_across_passes():
    backend = FractionBackend(0.5, spans=[[(0.1, 0.5)], [(0.4, 1.0), (2.0, 2.5)]])
    result = _run(backend, _signal(), options={"passes": 2})
    assert result.stems[0].spans == [[0.1, 1.0], [2.0, 2.5]]


def test_passes_with_chunking():
    signal = _signal(frames=337)
    backend = FractionBackend(0.5)
    result = _run(backend, signal, options={"passes": 2, "chunk_seconds": 1})
    assert result.model["chunked"] is True
    lengths = [len(c["input"]) for c in backend.calls]
    assert lengths == [100, 100, 100, 100, 37] * 2  # every pass re-chunks the full-length residual
    np.testing.assert_allclose(result.stems[0].samples, signal * 0.75, atol=1e-5)
    _reconstructs(signal, result)


def test_passes_with_per_channel_stereo():
    signal = _signal(channels=2)
    backend = FractionBackend(0.5)
    result = _run(backend, signal, options={"passes": 2, "stereo_mode": "per_channel"})
    assert len(backend.calls) == 4
    np.testing.assert_allclose(backend.calls[2]["input"], signal[0] * 0.5, atol=1e-6)
    np.testing.assert_allclose(backend.calls[3]["input"], signal[1] * 0.5, atol=1e-6)
    np.testing.assert_allclose(result.stems[0].samples, signal * 0.75, atol=1e-6)
    _reconstructs(signal, result)


def test_passes_with_duplicate_stereo_reconstruct():
    signal = _signal(channels=2)
    result = _run(FractionBackend(0.5), signal, options={"passes": 3})
    _reconstructs(signal, result)


def test_residual_model_uses_last_pass():
    signal = _signal()
    result = _run(FractionBackend(0.5), signal, options={"passes": 3, "residual": "model"})
    np.testing.assert_allclose(result.stems[-1].samples, signal * 0.125, atol=1e-6)


def test_visual_interval_applies_to_every_pass():
    signal = _signal(frames=400)
    backend = FractionBackend(0.5)
    engine = Engine(Settings(chunk_overlap_seconds=0.25), backend)
    visual = VisualInput(video_path="unused.mp4", interval=(1.0, 2.0))
    engine.translate_visual = lambda *a, **k: (visual, [])
    meta = SeparationMetadata.model_validate({"prompts": ["voice"], "options": {"passes": 2}})
    result = engine.separate(DecodedAudio(signal, SR, SR, 1), meta)
    assert [len(c["input"]) for c in backend.calls] == [100, 100]
    np.testing.assert_allclose(backend.calls[1]["input"], signal[0, 100:200] * 0.5, atol=1e-6)
    target = result.stems[0].samples
    np.testing.assert_allclose(target[:, 100:200], signal[:, 100:200] * 0.75, atol=1e-6)
    assert not target[:, :100].any() and not target[:, 200:].any()
    _reconstructs(signal, result)


@pytest.mark.parametrize("passes", [0, 4, "2", 1.5, True])
def test_invalid_passes_rejected(passes):
    with pytest.raises((InvalidRequest, ValueError)):
        _run(FractionBackend(), _signal(), options={"passes": passes})


def test_max_passes_setting_validated(monkeypatch):
    from sam_audio_service.config import load_settings

    monkeypatch.setenv("AUDIO_MAX_PASSES", "6")
    with pytest.raises(ValueError, match="AUDIO_MAX_PASSES"):
        load_settings()
    monkeypatch.setenv("AUDIO_MAX_PASSES", "5")
    assert load_settings().max_passes == 5


# ------------------------------------------------------------------ mock backend, end to end
def _mock_mixture(seconds=1.0):
    return np.random.default_rng(1).uniform(-0.3, 0.3, (1, int(48000 * seconds))).astype(np.float32)


def test_mock_second_pass_extracts_more_and_reconstructs():
    signal = _mock_mixture()
    audio = DecodedAudio(signal, 48000, 48000, 1)
    # The mock's band-pass extracts MORE on its second pass (a mock artefact), which the real-model
    # hallucination guard would discard; disable the guard here to test the arithmetic alone.
    engine = Engine(Settings(pass_max_ratio_db=20.0, pass_min_similarity=0.0), MockBackend())
    one = engine.separate(audio, SeparationMetadata.model_validate({"prompts": ["siren"], "options": {"passes": 1}}))
    two = engine.separate(audio, SeparationMetadata.model_validate({"prompts": ["siren"], "options": {"passes": 2}}))
    e1 = float(np.sum(one.stems[0].samples.astype(np.float64) ** 2))
    e2 = float(np.sum(two.stems[0].samples.astype(np.float64) ** 2))
    assert two.stems[0].refinement["passes_run"] == 2
    assert e2 > e1
    _reconstructs(signal, two)


def _post(client, wav_bytes, metadata):
    return client.post("/separate", files={"audio": ("audio.wav", wav_bytes, "audio/wav")}, data={"metadata": json.dumps(metadata)})


def test_api_refinement_shape_and_reconstruction(client):
    wav = encode_wav_pcm16(np.repeat(_mock_mixture(), 2, axis=0), 48000)
    r = _post(client, wav, {"prompts": ["siren"], "options": {"passes": 2}})
    assert r.status_code == 200, r.text
    body = r.json()
    target, residual = body["stems"]
    assert set(target["refinement"]) == {"passes_requested", "passes_run", "per_pass_energy_db", "stopped_early"}  # stop_reason only when stopped early
    assert target["refinement"]["passes_requested"] == 2 and target["refinement"]["passes_run"] == 2
    assert len(target["refinement"]["per_pass_energy_db"]) == 2
    assert "refinement" in residual and residual["refinement"] is None
    assert body["model"]["passes"] == 2
    mixture = decode(data=wav).samples.astype(np.float64)
    total = sum(decode(data=base64.b64decode(s["audio_base64"])).samples.astype(np.float64) for s in body["stems"])
    np.testing.assert_allclose(total, mixture, atol=1e-5)


def test_api_rejects_too_many_passes(client):
    wav = encode_wav_pcm16(_mock_mixture(0.2), 48000)
    for passes in (4, 0, "two"):
        r = _post(client, wav, {"prompts": ["siren"], "options": {"passes": passes}})
        assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_REQUEST", (passes, r.text)


def test_api_v1_separate_passes(client, tmp_path):
    path = tmp_path / "mix.wav"
    path.write_bytes(encode_wav_pcm16(_mock_mixture(), 48000))
    r = client.post("/v1/separate", json={"audio_ref": path.as_uri(), "prompts": ["dog barking"], "options": {"passes": 3}})
    assert r.status_code == 200, r.text
    body = r.json()
    ref = body["stems"][0]["refinement"]
    # The mock's band-pass removes almost all of the band on pass 1, so pass 2 is negligible.
    assert ref["passes_requested"] == 3 and ref["passes_run"] == 1 and ref["stopped_early"] is True
    assert "pass 2 extracted negligible energy; stopped after 1 pass(es)" in body["warnings"]
    r = client.post("/v1/separate", json={"audio_ref": path.as_uri(), "prompts": ["dog"], "options": {"passes": 9}})
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_REQUEST"


def test_capabilities_report_passes(client):
    caps = client.get("/capabilities").json()
    assert caps["defaults"]["passes"] == 1
    assert caps["limits"]["max_passes"] == 3


class HallucinatingBackend(FractionBackend):
    """Pass 1 extracts half of the input; later calls return a loud, unrelated high band."""

    def separate(self, mixture, prompt, options):
        if len(self.calls) == 0:
            return super().separate(mixture, prompt, options)
        self.calls.append({"input": mixture.copy(), "text": prompt.text, "anchors": prompt.anchors, "visual": prompt.visual})
        t = np.arange(mixture.shape[-1]) / SR
        other = (0.9 * np.sin(2 * np.pi * (SR / 4) * t)).astype(np.float32)  # one loud tone: not the voice
        other = np.broadcast_to(other, mixture.shape).astype(np.float32)
        return BackendResult(target=other, residual=(mixture - other).astype(np.float32))


def test_second_pass_that_extracts_a_different_loud_sound_is_discarded():
    signal = _signal()
    guarded = Settings(chunk_overlap_seconds=0.25, pass_max_ratio_db=-3.0, pass_min_similarity=0.6)  # the real-backend defaults
    result = _run(HallucinatingBackend(0.5), signal, settings=guarded, prompts=["voice"], options={"passes": 3})
    ref = result.stems[0].refinement
    assert ref["passes_run"] == 1 and ref["stopped_early"] is True and ref["stop_reason"] in ("too_much", "dissimilar")
    np.testing.assert_allclose(result.stems[0].samples, signal * 0.5, atol=1e-6)
    _reconstructs(signal, result)
    assert any("discarded" in w for w in result.warnings)


def test_pass_safeguards_are_configurable(monkeypatch):
    signal = _signal()
    # A second pass that is -6 dB and spectrally identical passes the default guards (-3 dB, 0.6).
    guarded = Settings(chunk_overlap_seconds=0.25, pass_max_ratio_db=-3.0, pass_min_similarity=0.6)
    result = _run(FractionBackend(0.5), signal, settings=guarded, prompts=["voice"], options={"passes": 2})
    assert result.stems[0].refinement["passes_run"] == 2 and "stop_reason" not in result.stems[0].refinement
    # Tightening the ratio to -10 dB discards it.
    result = _run(FractionBackend(0.5), signal, settings=Settings(chunk_overlap_seconds=0.25, pass_max_ratio_db=-10.0), prompts=["voice"], options={"passes": 2})
    assert result.stems[0].refinement["passes_run"] == 1 and result.stems[0].refinement["stop_reason"] == "too_much"
