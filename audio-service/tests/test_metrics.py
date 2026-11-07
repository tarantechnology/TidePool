"""Known-answer tests for the separation metrics.

Each test pins a property the benchmark relies on: scale invariance, the direction of the
scale, behaviour on silence, and a hard error on length mismatch (a silent truncation
would hide exactly the chunking bugs these metrics are meant to expose).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from sam_audio_service.metrics import (
    as_mono,
    magnitude_spectrogram,
    reconstruction_error,
    sdr,
    si_sdr,
    si_sdr_improvement,
    spectral_leakage_db,
)

RATE = 48_000


def tone(freq: float, seconds: float = 0.5, amplitude: float = 0.5, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


@pytest.fixture
def noise() -> np.ndarray:
    return np.random.default_rng(0).standard_normal(int(0.5 * RATE)).astype(np.float32) * 0.1


# ------------------------------------------------------------------------------ si_sdr
def test_si_sdr_identical_is_infinite():
    reference = tone(440.0)
    assert si_sdr(reference, reference) == math.inf
    assert si_sdr(reference, reference) > 100  # "very high" however it is clamped later


def test_si_sdr_is_scale_invariant():
    reference = tone(440.0)
    estimate = reference + 0.05 * tone(1500.0)
    plain = si_sdr(estimate, reference)
    assert math.isfinite(plain)
    assert si_sdr(estimate * 0.001, reference) == pytest.approx(plain, abs=1e-6)
    assert si_sdr(estimate * 73.0, reference) == pytest.approx(plain, abs=1e-6)
    assert si_sdr(reference * 0.25, reference) == math.inf  # an exactly scaled copy is perfect


def test_si_sdr_noise_estimate_is_low(noise):
    assert si_sdr(noise, tone(440.0)) < 0.0


def test_si_sdr_tracks_interference_level():
    reference = tone(440.0)
    interference = tone(3000.0)
    clean = si_sdr(reference + 0.01 * interference, reference)
    dirty = si_sdr(reference + 0.5 * interference, reference)
    assert clean > dirty > -10.0


def test_si_sdr_accepts_stereo_and_averages_channels():
    reference = tone(440.0)
    stereo = np.stack([reference, reference])
    assert si_sdr(stereo, reference) == math.inf
    assert si_sdr(stereo, np.stack([reference, reference])) == math.inf


# --------------------------------------------------------------------------------- sdr
def test_sdr_is_scale_dependent():
    reference = tone(440.0)
    assert sdr(reference, reference) == math.inf
    assert sdr(reference * 0.5, reference) == pytest.approx(10 * math.log10(4.0), abs=1e-6)
    assert sdr(np.zeros_like(reference), reference) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------- si_sdri
def test_si_sdr_improvement_zero_for_passthrough():
    reference, interference = tone(440.0), tone(3000.0)
    mixture = reference + interference
    assert si_sdr_improvement(mixture, reference, mixture) == pytest.approx(0.0, abs=1e-9)


def test_si_sdr_improvement_positive_when_interference_is_reduced():
    reference, interference = tone(440.0), tone(3000.0)
    mixture = reference + interference
    improvement = si_sdr_improvement(reference + 0.1 * interference, reference, mixture)
    assert improvement > 15.0


def test_si_sdr_improvement_negative_when_target_is_removed(noise):
    reference, interference = tone(440.0), tone(3000.0)
    mixture = reference + interference
    assert si_sdr_improvement(interference + noise, reference, mixture) < 0.0


# -------------------------------------------------------------------------- silence
def test_silence_never_raises_and_is_signposted():
    silence = np.zeros(int(0.2 * RATE), dtype=np.float32)
    signal = tone(440.0, 0.2)
    assert math.isnan(si_sdr(silence, silence))
    assert si_sdr(silence, signal) == -math.inf
    assert si_sdr(signal, silence) == -math.inf
    assert math.isnan(sdr(silence, silence))
    assert sdr(signal, silence) == -math.inf
    assert math.isnan(si_sdr_improvement(silence, silence, silence))
    assert math.isnan(spectral_leakage_db(silence, signal, signal))
    assert reconstruction_error([silence], silence) == 0.0


def test_empty_signals_do_not_raise():
    empty = np.zeros(0, dtype=np.float32)
    assert math.isnan(si_sdr(empty, empty))
    assert math.isnan(sdr(empty, empty))
    assert reconstruction_error([empty], empty) == 0.0


# --------------------------------------------------------------------------- leakage
def test_leakage_low_for_clean_target_high_for_wrong_source():
    target, interference = tone(500.0), tone(6000.0)  # well separated in frequency
    clean = spectral_leakage_db(target, target, interference)
    wrong = spectral_leakage_db(interference, target, interference)
    half = spectral_leakage_db(target + interference, target, interference)
    assert clean < -30.0
    assert wrong > 30.0
    assert abs(half) < 3.0  # equal-energy mixture sits around 0 dB


def test_leakage_orders_two_estimates_correctly():
    target, interference = tone(500.0), tone(6000.0)
    better = spectral_leakage_db(target + 0.05 * interference, target, interference)
    worse = spectral_leakage_db(target + 0.5 * interference, target, interference)
    assert better < worse


# ------------------------------------------------------------------- reconstruction
def test_reconstruction_error_exact_and_broken():
    mixture = tone(440.0) + tone(3000.0)
    target = tone(440.0)
    assert reconstruction_error([target, mixture - target], mixture) == pytest.approx(0.0, abs=1e-6)
    broken = reconstruction_error([target, (mixture - target) * 0.5], mixture)
    assert broken > 0.1


def test_reconstruction_error_requires_stems():
    with pytest.raises(ValueError, match="at least one stem"):
        reconstruction_error([], tone(440.0))


# ------------------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "call",
    [
        lambda a, b: si_sdr(a, b),
        lambda a, b: sdr(a, b),
        lambda a, b: si_sdr_improvement(a, b, b),
        lambda a, b: spectral_leakage_db(a, b, b),
        lambda a, b: reconstruction_error([a], b),
    ],
)
def test_length_mismatch_raises(call):
    with pytest.raises(ValueError, match="length mismatch"):
        call(tone(440.0, 0.5), tone(440.0, 0.4))


def test_bad_shape_and_non_finite_raise():
    with pytest.raises(ValueError, match="channels, frames"):
        as_mono(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="at most"):
        as_mono(np.zeros((40, 4)))
    with pytest.raises(ValueError, match="NaN or inf"):
        as_mono(np.array([0.0, np.nan, 1.0]))


def test_as_mono_averages_channels():
    mono = as_mono(np.array([[1.0, 1.0], [-1.0, 3.0]]))
    assert mono.tolist() == [0.0, 2.0]


def test_magnitude_spectrogram_shape_and_short_input():
    spectrogram = magnitude_spectrogram(tone(440.0, 0.5), n_fft=1024, hop=256)
    assert spectrogram.shape == (1 + (int(0.5 * RATE) - 1024) // 256, 513)
    assert magnitude_spectrogram(np.zeros(10, dtype=np.float32), n_fft=1024, hop=256).shape == (1, 513)
    peak_bin = int(np.argmax(magnitude_spectrogram(tone(1000.0, 0.2), n_fft=4096, hop=1024).mean(axis=0)))
    assert abs(peak_bin * RATE / 4096 - 1000.0) < 20.0
