"""Time-frequency mask refinement (``options.refine = "mask"``), see ``refine.py``.

Pins the STFT round trip, the mask arithmetic on cases with a known answer, the opt-in
threshold / gate controls, and the engine hook: refinement only touches the summed target,
the residual is still ``mixture - targets`` and a visual interval stays exactly zero outside.
"""
import numpy as np
import pytest

from sam_audio_service.audio_io import DecodedAudio
from sam_audio_service.backends.base import VisualInput
from sam_audio_service.backends.mock import MockBackend
from sam_audio_service.config import Settings
from sam_audio_service.contracts import SeparationMetadata
from sam_audio_service.engine import Engine
from sam_audio_service.refine import apply_gate, apply_threshold, istft, mask_refine, stft

from test_multipass import SR as FRACTION_SR, FractionBackend, _reconstructs, _run, _signal

RATE = 48_000


def _noise(seconds=1.0, channels=1, seed=0):
    return np.random.default_rng(seed).uniform(-0.5, 0.5, (channels, int(RATE * seconds))).astype(np.float32)


def _band_energy(x, lo, hi, rate=RATE):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    spec = np.fft.rfft(x * np.hanning(len(x)))  # windowed, so a loud tone does not leak into the band
    freqs = np.fft.rfftfreq(len(x), 1 / rate)
    return float(np.sum(np.abs(spec[(freqs >= lo) & (freqs <= hi)]) ** 2))


# ------------------------------------------------------------------- STFT
@pytest.mark.parametrize("n_fft", [512, 1024, 2048, 4096])
def test_stft_istft_round_trip(n_fft):
    x = _noise()[0].astype(np.float64)
    y = istft(stft(x, n_fft), n_fft, length=len(x))
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) < 1e-6


@pytest.mark.parametrize("length", [2, 3, 100, 1000, 2049])
def test_round_trip_on_signals_shorter_than_the_window(length):
    x = _noise()[0, :length].astype(np.float64)
    assert np.max(np.abs(istft(stft(x, 2048), 2048, length=length) - x)) < 1e-6


# ------------------------------------------------------------------- mask arithmetic
def test_target_equal_to_mixture_is_kept():
    x = _noise()
    np.testing.assert_allclose(mask_refine(x, x), x, atol=1e-5)


def test_silent_target_stays_silent():
    x = _noise()
    out = mask_refine(x, np.zeros_like(x))
    assert out.dtype == np.float32 and out.shape == x.shape
    assert not out.any()


def test_amplitude_ratio_mask_recovers_a_scaled_target():
    x = _noise()
    np.testing.assert_allclose(mask_refine(x, 0.3 * x, p=1.0), 0.3 * x, atol=1e-4)


def test_blend_mixes_refined_and_model_target():
    x = _noise()
    target = 0.3 * x + 0.05 * _noise(seed=1)
    refined = mask_refine(x, target)
    half = mask_refine(x, target, blend=0.5)
    np.testing.assert_allclose(half, 0.5 * refined + 0.5 * target, atol=1e-6)
    np.testing.assert_allclose(mask_refine(x, target, blend=0.0), target, atol=1e-7)


def test_floor_lifts_the_mask():
    x = _noise()
    np.testing.assert_allclose(mask_refine(x, np.zeros_like(x), floor=0.2), 0.2 * x, atol=1e-5)


def test_stereo_mixture_with_a_duplicated_target_is_refined_per_channel():
    left, right = _noise(seed=1)[0], _noise(seed=2)[0]
    mixture = np.stack([left, right])
    mono = 0.5 * mixture.mean(axis=0)
    target = np.stack([mono, mono])  # the "duplicate" stereo mode
    out = mask_refine(mixture, target)
    assert out.shape == mixture.shape
    assert not np.allclose(out[0], out[1])  # the shared mask is applied to each channel's own mixture
    assert np.all(np.isfinite(mixture - out))


# ------------------------------------------------------------------- threshold / gate
def test_threshold_soft_knee():
    m = np.array([0.0, 0.2, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(apply_threshold(m, 0.5), [0.0, 0.0, 0.0, 0.5, 1.0])


def test_threshold_removes_a_partially_attenuated_interferer():
    t = np.arange(RATE) / RATE
    loud = 0.5 * np.sin(2 * np.pi * 1000 * t)
    quiet = 0.05 * np.sin(2 * np.pi * 300 * t)
    mixture = (loud + quiet).astype(np.float32)[None, :]
    target = (loud + 0.1 * quiet).astype(np.float32)[None, :]  # 10% of the 300 Hz leaked in
    plain = mask_refine(mixture, target)
    cut = mask_refine(mixture, target, threshold=0.5)
    # Measured away from the first/last window: reflect padding of a pure sine spreads its
    # spectrum there, which no mask setting can (or should) hide.
    inner = slice(2048, -2048)
    leaked = _band_energy(target[:, inner], 250, 350)
    cut_db = 10 * np.log10(_band_energy(cut[:, inner], 250, 350) / leaked)
    assert cut_db < -20
    assert _band_energy(cut[:, inner], 250, 350) < 0.1 * _band_energy(plain[:, inner], 250, 350)
    # The wanted tone survives.
    assert 10 * np.log10(_band_energy(cut[:, inner], 950, 1050) / _band_energy(target[:, inner], 950, 1050)) > -1


def test_gate_silences_frames_where_the_target_is_absent():
    x = _noise()
    half = x.shape[1] // 2
    target = x.copy()
    target[:, half:] *= 0.1  # the model only attenuated the second half
    n_fft = 2048
    plain = mask_refine(x, target, n_fft=n_fft)
    gated = mask_refine(x, target, n_fft=n_fft, gate_db=30)
    tail = slice(half + 2 * n_fft, None)  # clear of the hold and the window overlap
    assert np.abs(plain[:, tail]).max() > 1e-3
    assert np.abs(gated[:, tail]).max() < 1e-6
    head = slice(n_fft, half - 2 * n_fft)
    np.testing.assert_allclose(gated[:, head], x[:, head], atol=1e-3)


def test_gate_hold_keeps_neighbours_of_open_frames():
    mix = np.ones((7, 4), dtype=complex)
    mask = np.zeros((7, 4))
    mask[3] = 1.0  # only frame 3 carries the target
    gated = apply_gate(np.where(mask > 0, 1.0, 0.5), mix, gate_db=3)  # 0.5 -> -6 dB: closed
    assert [bool(row.any()) for row in gated] == [False, False, True, True, True, False, False]


def test_defaults_are_off_for_threshold_and_gate():
    x = _noise()
    target = 0.3 * x + 0.05 * _noise(seed=1)
    np.testing.assert_array_equal(mask_refine(x, target), mask_refine(x, target, threshold=0.0, gate_db=0.0))


# ------------------------------------------------------------------- engine hook
def _mock_audio(seconds=1.0, channels=1):
    signal = _noise(seconds, channels, seed=3) * 0.6
    return signal, DecodedAudio(signal, RATE, RATE, channels)


def test_engine_mask_refinement_keeps_reconstruction_and_passes():
    signal, audio = _mock_audio()
    engine = Engine(Settings(), MockBackend())
    raw = engine.separate(audio, SeparationMetadata.model_validate({"prompts": ["siren"], "options": {"passes": 2}}))
    refined = engine.separate(audio, SeparationMetadata.model_validate({"prompts": ["siren"], "options": {"passes": 2, "refine": "mask"}}))
    total = sum(s.samples.astype(np.float64) for s in refined.stems)
    np.testing.assert_allclose(total, signal, atol=1e-5)
    assert refined.stems[0].refinement["passes_run"] == raw.stems[0].refinement["passes_run"]
    assert not np.allclose(refined.stems[0].samples, raw.stems[0].samples)
    assert refined.model["refine"] == {"mode": "mask", "p": 2.0, "n_fft": 2048, "floor": 0.0, "blend": 1.0, "threshold": 0.0, "gate_db": 0.0}
    assert raw.model["refine"] is None


def test_engine_refine_options_override_settings():
    _, audio = _mock_audio(0.5)
    engine = Engine(Settings(refine="mask", refine_p=1.0), MockBackend())
    result = engine.separate(audio, SeparationMetadata.model_validate({"prompts": ["siren"], "options": {"refine_n_fft": 1024, "refine_threshold": 0.4, "refine_gate_db": 40}}))
    assert result.model["refine"]["p"] == 1.0 and result.model["refine"]["n_fft"] == 1024
    assert result.model["refine"]["threshold"] == 0.4 and result.model["refine"]["gate_db"] == 40.0
    off = engine.separate(audio, SeparationMetadata.model_validate({"prompts": ["siren"], "options": {"refine": "off"}}))
    assert off.model["refine"] is None


def test_engine_per_channel_refinement_reconstructs():
    signal, audio = _mock_audio(0.5, channels=2)
    engine = Engine(Settings(), MockBackend())
    result = engine.separate(audio, SeparationMetadata.model_validate({"prompts": ["siren"], "options": {"refine": "mask", "stereo_mode": "per_channel"}}))
    np.testing.assert_allclose(sum(s.samples.astype(np.float64) for s in result.stems), signal, atol=1e-5)


def test_engine_refinement_keeps_a_visual_interval_exactly_zero_outside():
    signal = _signal(frames=400)
    engine = Engine(Settings(chunk_overlap_seconds=0.25), FractionBackend(0.5))
    engine.translate_visual = lambda *a, **k: (VisualInput(video_path="unused.mp4", interval=(1.0, 2.0)), [])
    meta = SeparationMetadata.model_validate({"prompts": ["voice"], "options": {"refine": "mask", "refine_n_fft": 512}})
    result = engine.separate(DecodedAudio(signal, FRACTION_SR, FRACTION_SR, 1), meta)
    target = result.stems[0].samples
    assert not target[:, :100].any() and not target[:, 200:].any()
    assert target[:, 100:200].any()
    _reconstructs(signal, result)


def test_default_off_leaves_fraction_backend_exact():
    signal = _signal()
    result = _run(FractionBackend(0.5), signal, options={"passes": 3})
    np.testing.assert_allclose(result.stems[0].samples, signal * 0.875, atol=1e-6)
    _reconstructs(signal, result)
    assert result.model["refine"] is None


@pytest.mark.parametrize("options", [{"refine": "wiener"}, {"refine_p": 0}, {"refine_p": 9}, {"refine_n_fft": 300}, {"refine_floor": 0.6}, {"refine_blend": 1.5}, {"refine_threshold": 1.0}, {"refine_gate_db": 81}])
def test_invalid_refine_options_are_rejected(options):
    with pytest.raises(ValueError):
        SeparationMetadata.model_validate({"prompts": ["voice"], "options": options})


@pytest.mark.parametrize(("var", "value"), [("AUDIO_REFINE", "wiener"), ("AUDIO_REFINE_P", "0"), ("AUDIO_REFINE_NFFT", "1000"), ("AUDIO_REFINE_FLOOR", "0.7"), ("AUDIO_REFINE_BLEND", "2"), ("AUDIO_REFINE_THRESHOLD", "1"), ("AUDIO_REFINE_GATE_DB", "-1")])
def test_refine_settings_are_validated(monkeypatch, var, value):
    from sam_audio_service.config import load_settings

    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError, match=var):
        load_settings()
