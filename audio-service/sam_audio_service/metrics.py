"""Audio quality metrics for separation benchmarking (pure numpy, no SciPy/librosa).

Every function accepts float arrays shaped either ``(frames,)`` or ``(channels, frames)``
-- the ``(channels, frames)`` layout used everywhere else in this service. Multi-channel
input is averaged to mono before measuring, so a ``stereo_mode="duplicate"`` estimate can
be compared against panned ground truth.

Conventions
-----------
* Decibel results are ``float``; they may be ``+inf`` (perfect), ``-inf`` (no useful
  energy) or ``nan`` (undefined, e.g. both signals silent). Nothing raises on silence --
  only a genuine mistake (length mismatch, bad shape, non-finite samples) raises.
* Length mismatches raise :class:`ValueError`: silently truncating would hide an
  off-by-one in chunking or resampling, which is exactly what these metrics exist to
  catch.

The short STFT helper below is intentionally local: the mock backend has its own private
STFT for synthesis, and metrics must stay usable when the real SAM Audio backend (and no
mock) is installed.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

__all__ = [
    "si_sdr",
    "sdr",
    "si_sdr_improvement",
    "spectral_leakage_db",
    "reconstruction_error",
    "magnitude_spectrogram",
    "as_mono",
]

_EPS = 1e-20
MAX_CHANNELS = 16


def as_mono(signal: np.ndarray, *, name: str = "signal") -> np.ndarray:
    """Return ``signal`` as a 1-D float64 mono array.

    ``(channels, frames)`` input is averaged across channels. NaN/inf is rejected rather
    than silently zeroed: a model that emits them is a bug the benchmark should report.
    """
    array = np.asarray(signal)
    if array.ndim == 2:
        if array.shape[0] > MAX_CHANNELS:
            raise ValueError(f"{name} has shape {array.shape}; expected (channels, frames) with at most {MAX_CHANNELS} channels")
        array = array.mean(axis=0)
    elif array.ndim != 1:
        raise ValueError(f"{name} must be 1-D (frames,) or 2-D (channels, frames), got shape {array.shape}")
    array = array.astype(np.float64, copy=False)
    if array.size and not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or inf")
    return array


def _pair(estimate: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    est = as_mono(estimate, name="estimate")
    ref = as_mono(reference, name="reference")
    if est.shape != ref.shape:
        raise ValueError(f"length mismatch: estimate has {est.size} frames, reference has {ref.size}")
    return est, ref


def _ratio_db(numerator: float, denominator: float) -> float:
    """``10*log10(num/den)`` with documented behaviour at the degenerate points."""
    if numerator <= 0.0 and denominator <= 0.0:
        return math.nan
    if denominator <= 0.0:
        return math.inf
    if numerator <= 0.0:
        return -math.inf
    return float(10.0 * math.log10(numerator / denominator))


def si_sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant signal-to-distortion ratio in dB (Le Roux et al., 2019).

    The reference is first scaled by the least-squares gain that best explains the
    estimate, so a quieter-but-correct stem scores the same as a level-matched one.
    Returns ``+inf`` for an exact (or exactly scaled) copy, ``-inf`` when the estimate
    carries no component of the reference (silent estimate, silent reference, or an
    estimate orthogonal to it) and ``nan`` when both signals are silent.
    """
    est, ref = _pair(estimate, reference)
    if est.size == 0:
        return math.nan
    ref_energy, est_energy = float(ref @ ref), float(est @ est)
    if ref_energy <= _EPS and est_energy <= _EPS:
        return math.nan
    if ref_energy <= _EPS or est_energy <= _EPS:
        return -math.inf
    projection = (float(est @ ref) / ref_energy) * ref
    noise = est - projection
    return _ratio_db(float(projection @ projection), float(noise @ noise))


def sdr(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Plain (scale-dependent) SDR in dB: ``10*log10(|ref|^2 / |ref - est|^2)``.

    Unlike :func:`si_sdr` this punishes level errors, which is what the editor cares
    about when a stem is dropped straight into a mix.
    """
    est, ref = _pair(estimate, reference)
    if est.size == 0:
        return math.nan
    error = ref - est
    return _ratio_db(float(ref @ ref), float(error @ error))


def si_sdr_improvement(estimate: np.ndarray, reference: np.ndarray, mixture: np.ndarray) -> float:
    """SI-SDRi in dB: how much better the estimate is than doing nothing at all.

    ``si_sdr(estimate, reference) - si_sdr(mixture, reference)``. Zero means "no better
    than handing the untouched mixture back"; negative means actively worse. ``nan`` if
    either term is undefined, and ``nan`` rather than ``inf - inf`` when both are
    infinite in the same direction.
    """
    after = si_sdr(estimate, reference)
    before = si_sdr(mixture, reference)
    if math.isnan(after) or math.isnan(before):
        return math.nan
    if math.isinf(after) and math.isinf(before) and (after > 0) == (before > 0):
        return math.nan
    return float(after - before)


def magnitude_spectrogram(signal: np.ndarray, *, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Magnitude STFT, shape ``(frames, n_fft // 2 + 1)``, Hann window, no padding.

    Short input is zero-padded to one full frame so short fixtures still measure.
    """
    x = as_mono(signal, name="signal")
    if n_fft <= 0 or hop <= 0:
        raise ValueError("n_fft and hop must be positive")
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    window = np.hanning(n_fft + 1)[:-1]
    frames = 1 + (x.size - n_fft) // hop
    index = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    return np.abs(np.fft.rfft(x[index] * window, axis=1))


def spectral_leakage_db(
    estimate: np.ndarray,
    target_reference: np.ndarray,
    interference_reference: np.ndarray,
    *,
    n_fft: int = 2048,
    hop: int = 512,
) -> float:
    """How much non-target sound is still audible in the target stem, in dB.

    The estimate's energy is split per time-frequency bin in proportion to the ground
    truth energy of the target and of everything else, then reported as
    ``10*log10(interference_energy / target_energy)``. Lower is better: ``-20 dB`` means
    the leaked interference sits 20 dB under the wanted sound; ``0 dB`` means the stem is
    half unwanted sound. ``-inf`` for a perfectly clean stem, ``+inf`` for a stem that is
    pure interference, ``nan`` when the estimate is silent.

    This is the metric that decides whether the cinematic demo works: the siren stem must
    not carry the argument, or automating its volume ducks the dialogue too.
    """
    lengths = {
        "estimate": as_mono(estimate, name="estimate").size,
        "target_reference": as_mono(target_reference, name="target_reference").size,
        "interference_reference": as_mono(interference_reference, name="interference_reference").size,
    }
    if len(set(lengths.values())) > 1:
        raise ValueError(f"length mismatch: {lengths}")
    est = magnitude_spectrogram(estimate, n_fft=n_fft, hop=hop) ** 2
    tgt = magnitude_spectrogram(target_reference, n_fft=n_fft, hop=hop) ** 2
    itf = magnitude_spectrogram(interference_reference, n_fft=n_fft, hop=hop) ** 2
    total = tgt + itf
    share_target = np.divide(tgt, total, out=np.zeros_like(total), where=total > _EPS)
    share_interference = np.divide(itf, total, out=np.zeros_like(total), where=total > _EPS)
    return _ratio_db(float((est * share_interference).sum()), float((est * share_target).sum()))


def reconstruction_error(stems: Sequence[np.ndarray], mixture: np.ndarray) -> float:
    """Max absolute sample error of ``sum(stems) - mixture`` (0.0 = perfect).

    The editor mixes the stems back together, so they must add up to the original. Values
    around 1e-7 are float32 round-off; around 1e-4 they are 16-bit PCM quantisation from a
    WAV round-trip; anything larger is a real pipeline bug (lost samples, double gain).
    """
    mix = as_mono(mixture, name="mixture")
    if not stems:
        raise ValueError("reconstruction_error needs at least one stem")
    total = np.zeros_like(mix)
    for position, stem in enumerate(stems):
        array = as_mono(stem, name=f"stems[{position}]")
        if array.shape != mix.shape:
            raise ValueError(f"length mismatch: stems[{position}] has {array.size} frames, mixture has {mix.size}")
        total += array
    if mix.size == 0:
        return 0.0
    return float(np.max(np.abs(total - mix)))
