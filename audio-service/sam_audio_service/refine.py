"""Time-frequency mask refinement (opt-in post-processing, ``options.refine = "mask"``).

SAM Audio generates its target with a codec decoder, so the waveform it returns is not a
filtered copy of the mixture: it can carry decoder artefacts and small phase/level drifts.
Mask refinement keeps the model's *decision* (which time-frequency bins belong to the
target) but rebuilds the waveform from the mixture itself::

    Tm = STFT(mono target), Rm = STFT(mono mixture - mono target)
    M  = |Tm|^p / (|Tm|^p + |Rm|^p + eps)        (a soft, Wiener-like ratio mask)
    T'_c = iSTFT(M * STFT(mixture[c]))            (one shared mask, applied per channel)

``p = 2`` is a power-spectrum (Wiener) ratio, ``p = 1`` an amplitude ratio. ``floor`` lifts
the mask's minimum (fewer musical-noise holes, more leakage) and ``blend`` mixes the refined
target back with the model's own (1 = refined only).

Two further opt-in controls remove what the model only partly attenuated, accepting some
artefacts in exchange:

* ``threshold`` (0..1): soft-knee cut, ``M' = 0 if M < threshold else (M - threshold) /
  (1 - threshold)``. 0.3-0.5 removes background bins the model only partially attenuated.
* ``gate_db`` (0..80, 0 = off): silence the stem where the sound is not really present.
  STFT frames whose masked-target energy is more than ``gate_db`` below the mixture frame
  energy get a zero mask in every bin (3-frame hold so words are not clipped).

The residual is NOT computed here: the
engine still derives it as ``mixture - sum(targets)``, so stems keep summing to the mixture.

Numpy only. STFT: periodic Hann window, hop ``n_fft // 4``, reflect padding of ``n_fft // 2``
on both sides, weighted overlap-add inverse normalised by the summed squared window, so a
round trip is exact to floating-point precision.
"""
from __future__ import annotations

import numpy as np


def _window(n_fft: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)  # periodic Hann


def _hop(n_fft: int) -> int:
    if n_fft < 4 or n_fft % 4:
        raise ValueError("n_fft must be a positive multiple of 4")
    return n_fft // 4


def _padded_length(length: int, n_fft: int, hop: int) -> int:
    """Reflect-padded length rounded up so the frames tile it exactly."""
    total = length + 2 * (n_fft // 2)
    frames = max(1, -(-(total - n_fft) // hop) + 1)
    return (frames - 1) * hop + n_fft


def stft(x: np.ndarray, n_fft: int = 2048, hop: int | None = None) -> np.ndarray:
    """1-D signal -> complex spectrogram of shape (frames, n_fft // 2 + 1)."""
    hop = hop or _hop(n_fft)
    if n_fft % hop:
        raise ValueError("hop must divide n_fft")
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    pad = n_fft // 2
    xp = np.pad(x, (pad, pad), mode="reflect" if len(x) >= 2 else "constant")
    target = _padded_length(len(x), n_fft, hop)
    if len(xp) < target:
        xp = np.pad(xp, (0, target - len(xp)))
    frames = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop]
    return np.fft.rfft(frames * _window(n_fft), axis=-1)


def istft(S: np.ndarray, n_fft: int = 2048, hop: int | None = None, length: int | None = None) -> np.ndarray:
    """Inverse of :func:`stft` (weighted overlap-add), trimmed to ``length`` samples."""
    hop = hop or _hop(n_fft)
    if n_fft % hop:
        raise ValueError("hop must divide n_fft")
    window = _window(n_fft)
    frames = np.fft.irfft(S, n=n_fft, axis=-1) * window  # (F, n_fft)
    count = frames.shape[0]
    ratio = n_fft // hop
    # Frame f starts at f * hop, so its q-th hop-sized block lands on output block f + q.
    blocks = np.zeros((count + ratio - 1, hop))
    wsum = np.zeros((count + ratio - 1, hop))
    parts = frames.reshape(count, ratio, hop)
    wparts = (window**2).reshape(ratio, hop)
    for q in range(ratio):
        blocks[q : q + count] += parts[:, q, :]
        wsum[q : q + count] += wparts[q]
    out = blocks.reshape(-1)
    norm = wsum.reshape(-1)
    out = np.divide(out, norm, out=np.zeros_like(out), where=norm > 1e-10)
    out = out[n_fft // 2 :]
    if length is not None:
        out = out[:length]
        if len(out) < length:
            out = np.pad(out, (0, length - len(out)))
    return out


def apply_threshold(mask: np.ndarray, threshold: float) -> np.ndarray:
    """Soft-knee cut: 0 below ``threshold``, rescaled linearly to [0, 1] above it.

    A hard cut leaves isolated surviving bins at full-ish gain (musical noise); the knee
    makes bins just above the threshold nearly silent too.
    """
    return np.where(mask < threshold, 0.0, (mask - threshold) / (1.0 - threshold))


def apply_residual_threshold(mask: np.ndarray, residual_threshold: float) -> np.ndarray:
    """Mirrored soft knee: 1 above ``1 - residual_threshold``, [0, 1 - t] rescaled to [0, 1].

    Where the target dominates a bin, the whole bin goes to the target, so the residual
    (``mixture - target``) holds no faint copy of the removed sound there.
    """
    top = 1.0 - residual_threshold
    return np.where(mask > top, 1.0, mask / top)


def apply_gate(mask: np.ndarray, mix_spec: np.ndarray, gate_db: float, eps: float = 1e-20) -> np.ndarray:
    """Zero every bin of frames where the masked target is > ``gate_db`` below the mixture.

    Frame energy is summed over bins of ``mask * mix_spec`` vs ``mix_spec``. The open/closed
    decision gets a 3-frame hold (a frame stays open when either neighbour is open) so word
    onsets and tails are not clipped.
    """
    mix_energy = np.sum(np.abs(mix_spec) ** 2, axis=-1)
    tgt_energy = np.sum(np.abs(mask * mix_spec) ** 2, axis=-1)
    tgt_db = 10 * np.log10(tgt_energy + eps)
    mix_db = 10 * np.log10(mix_energy + eps)
    is_open = tgt_db >= mix_db - gate_db
    held = is_open.copy()
    held[1:] |= is_open[:-1]
    held[:-1] |= is_open[1:]
    return mask * held[:, None]


def mask_refine(
    mixture: np.ndarray,
    target: np.ndarray,
    *,
    p: float = 2.0,
    n_fft: int = 2048,
    floor: float = 0.0,
    blend: float = 1.0,
    threshold: float = 0.0,
    residual_threshold: float = 0.0,
    gate_db: float = 0.0,
    eps: float = 1e-8,
) -> np.ndarray:
    """Rebuild ``target`` (C, T) from ``mixture`` (C, T) through a soft ratio mask. See module doc.

    Order: ratio mask -> ``threshold`` (soft knee) -> ``residual_threshold`` (mirrored knee)
    -> ``gate_db`` (frame gate) -> ``floor`` -> clip.
    """
    if not 0.0 <= threshold < 1.0:
        raise ValueError("threshold must be in [0, 1)")
    if not 0.0 <= residual_threshold < 1.0:
        raise ValueError("residual_threshold must be in [0, 1)")
    if gate_db < 0:
        raise ValueError("gate_db must not be negative")
    mixture = np.asarray(mixture, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if mixture.ndim == 1:
        mixture = mixture[None, :]
    if target.ndim == 1:
        target = target[None, :]
    if mixture.shape[-1] != target.shape[-1]:
        raise ValueError(f"mixture and target lengths differ: {mixture.shape[-1]} vs {target.shape[-1]}")
    length = mixture.shape[-1]
    if length < 2:
        return target.copy()
    hop = _hop(n_fft)
    mono_mix = mixture.astype(np.float64).mean(axis=0)
    mono_tgt = target.astype(np.float64).mean(axis=0)
    t_pow = np.abs(stft(mono_tgt, n_fft, hop)) ** p
    r_pow = np.abs(stft(mono_mix - mono_tgt, n_fft, hop)) ** p
    mask = t_pow / (t_pow + r_pow + eps)
    del t_pow, r_pow
    if threshold > 0:
        mask = apply_threshold(mask, threshold)
    if residual_threshold > 0:
        mask = apply_residual_threshold(mask, residual_threshold)
    if gate_db > 0:
        mask = apply_gate(mask, stft(mono_mix, n_fft, hop), gate_db)
    mask = np.clip(np.maximum(mask, floor), 0.0, 1.0)
    refined = np.empty(mixture.shape, dtype=np.float32)
    for c in range(mixture.shape[0]):
        refined[c] = istft(mask * stft(mixture[c], n_fft, hop), n_fft, hop, length).astype(np.float32)
    if blend >= 1.0:
        return refined
    return (blend * refined + (1.0 - blend) * target).astype(np.float32)


__all__ = ["stft", "istft", "mask_refine", "apply_threshold", "apply_residual_threshold", "apply_gate"]
