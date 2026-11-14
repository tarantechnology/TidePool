"""Pick one of K per-window candidates so neighbouring windows agree (``options.candidates``).

SAM Audio is a generative (flow-matching) model: each noise draw gives a slightly different
estimate. When a long input is chunked, the crossfade between two windows that made
*different* choices (e.g. one kept a background voice, the next did not) is audible. With
``options.candidates = K`` the backend returns K independent draws per window and the engine
chooses, window by window, in timeline order:

* ``"first"``: draw 0 (what K = 1 would have produced, modulo batch-size noise);
* ``"mean"``: the average of the K draws (an ensemble; smoother, may sound "averaged");
* ``"auto"``: ``"overlap"`` when chunked; for a single window the medoid when all draws agree
  (min pairwise NCC >= 0.8, K >= 3) and ``"mean"`` when they disagree;
Before any rule, draws more than 30 dB quieter than the loudest draw are dropped when at least
one draw is loud (the presence vote): the model heard the sound in some draws and not others,
and the user would otherwise hear the sound drop out for a whole window.

* ``"overlap"``: the draw whose head best matches what the previous window put into the
  shared overlap (normalised cross-correlation). With no usable previous window (first
  window, or an overlap quieter than -60 dBFS) it takes the medoid draw (highest mean
  correlation with the others) when K >= 3, else draw 0.

This is distinct from ``reranking_candidates``, which the model ranks internally with
CLAP/Judge/ImageBind and which needs a loaded ranker.
"""
from __future__ import annotations

import numpy as np

SILENCE_DBFS = -60.0
AGREEMENT_NCC = 0.8  # single-window draws with min pairwise NCC above this are "the same sound"
PRESENCE_DB = 30.0   # a draw this far below the loudest draw "did not hear" the sound (presence vote)
_EPS = 1e-12
MODES = ("first", "overlap", "mean", "auto")


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a = np.asarray(a[:n], dtype=np.float64)
    b = np.asarray(b[:n], dtype=np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + _EPS))


def _rms_dbfs(x: np.ndarray) -> float:
    if len(x) == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))
    return 20 * np.log10(rms) if rms > 0 else -np.inf


def _agreement(candidates: list[np.ndarray]) -> float:
    """Minimum pairwise normalised cross-correlation among the draws (1 = identical)."""
    k = len(candidates)
    return min(_ncc(candidates[i], candidates[j]) for i in range(k) for j in range(i + 1, k))


def _medoid(candidates: list[np.ndarray]) -> int:
    k = len(candidates)
    scores = [np.mean([_ncc(candidates[i], candidates[j]) for j in range(k) if j != i]) for i in range(k)]
    return int(np.argmax(scores))


def choose(candidates: list[np.ndarray], previous: np.ndarray | None, overlap: int, mode: str) -> tuple[int | None, np.ndarray]:
    """Return ``(index, target)``; ``index`` is None for ``"mean"``. See the module doc."""
    if not candidates:
        raise ValueError("choose() needs at least one candidate")
    if mode not in MODES:
        raise ValueError(f"unknown candidate selection mode {mode!r}; expected one of {MODES}")
    if mode == "first" or len(candidates) == 1:
        return 0, candidates[0]
    # Presence vote: when some draws return (near) silence and others return the sound, the
    # model disagreed about whether the described sound is present in this window. Measured
    # 2026-11-13 on a 48 s cooking clip with sam-audio-large: "narrator speaking" came back
    # silent for whole 10 s windows in some draws and present in others, which the user hears
    # as the voice dropping out for ten seconds at a time. Keep only the draws that heard it;
    # when every draw is silent the sound really is absent and nothing changes.
    levels = [_rms_dbfs(c) for c in candidates]
    loudest = max(levels)
    keep = [i for i, level in enumerate(levels) if level > loudest - PRESENCE_DB]
    if 0 < len(keep) < len(candidates):
        pool = [candidates[i] for i in keep]
    else:
        keep, pool = list(range(len(candidates))), candidates
    if mode == "auto":
        # Chunked: pick the draw that agrees with the previous window (seams do not flip).
        # Single window: when the draws agree (same sound, small differences) take the medoid so
        # nothing is smeared (averaging cost the music fixture 4 dB); when they disagree (the
        # model picked different sources, e.g. the wrong speaker) average them: measured
        # 2026-11-13 with sam-audio-large this turned a -57 dB wrong-speaker stem into ~0 dB.
        if overlap > 0:
            mode = "overlap"
        elif len(pool) >= 3 and _agreement(pool) >= AGREEMENT_NCC:
            return keep[_medoid(pool)], pool[_medoid(pool)]
        elif len(pool) == 2 and _agreement(pool) >= AGREEMENT_NCC:
            return keep[0], pool[0]
        else:
            mode = "mean"
    if mode == "mean":
        return None, np.mean(np.stack(pool).astype(np.float64), axis=0).astype(np.float32)
    if len(pool) == 1:
        return keep[0], pool[0]
    if previous is not None and overlap > 0 and _rms_dbfs(previous[:overlap]) > SILENCE_DBFS:
        n = min(overlap, len(previous))
        local = int(np.argmax([_ncc(c[:n], previous[:n]) for c in pool]))
    elif len(pool) >= 3:
        local = _medoid(pool)
    else:
        local = 0
    return keep[local], pool[local]


def choose_residual(candidate_residuals: list[np.ndarray | None] | None, idx: int | None) -> np.ndarray | None:
    """The residual matching :func:`choose`'s pick (the mean when ``idx`` is None)."""
    if not candidate_residuals or any(r is None for r in candidate_residuals):
        return None
    if idx is None:
        return np.mean(np.stack(candidate_residuals).astype(np.float64), axis=0).astype(np.float32)
    return candidate_residuals[idx]


__all__ = ["choose", "choose_residual", "MODES", "SILENCE_DBFS"]
