"""Named separation-settings sweeps for ``run_benchmark.py --sweep NAME``.

Each sweep is a list of configs; each config is a dict keyed by the runner's CLI dest names
(``ode_step``, ``window_candidates``, ``select``, ``refine``, ``refine_p``, ``refine_nfft``,
``refine_floor``, ``refine_blend``, ``refine_threshold``, ``refine_residual_threshold``,
``refine_gate_db``,
``chunk_seconds``, ``chunk_overlap``, ``candidates``, ``seed``, ``passes``, ``model_id``)
plus a ``"name"``. Keys a config leaves out keep the value given on the command line (or
the default). The **first config is the baseline** every other row's Δ is measured against.

A JSON file passed to ``--sweep`` uses the same shape: a list of these dicts, or
``{"name": ..., "configs": [...]}``.
"""
from __future__ import annotations

from typing import Any

_MASK = {"refine": "mask", "refine_p": 2, "refine_nfft": 2048}

SWEEPS: dict[str, list[dict[str, Any]]] = {
    # ODE solver step: halving it doubles model evaluations (latency) per window.
    "ode": [
        {"name": "step-0.0625", "ode_step": 0.0625},
        {"name": "step-0.03125", "ode_step": 0.03125},
        {"name": "step-0.015625", "ode_step": 0.015625},
    ],
    # Post-separation mask refinement: exponent p x STFT size.
    "refine": [
        {"name": "refine-off", "refine": "off"},
        *[
            {"name": f"mask-p{p}-n{nfft}", "refine": "mask", "refine_p": p, "refine_nfft": nfft}
            for p in (1, 2)
            for nfft in (1024, 2048, 4096)
        ],
    ],
    # Floor / blend around the best-guess mask (p=2, n_fft=2048).
    "refine-floor": [
        {"name": f"mask-floor{floor:g}-blend{blend:g}", **_MASK, "refine_floor": floor, "refine_blend": blend}
        for floor in (0.0, 0.02)
        for blend in (1.0, 0.5)
    ],
    # Leakage: hard threshold and gate on top of the p=2/2048 mask.
    "leakage": [
        *[
            {"name": f"mask-thr{threshold:g}-gate{gate:g}", **_MASK, "refine_threshold": threshold, "refine_gate_db": gate}
            for threshold in (0.0, 0.3, 0.5)
            for gate in (0.0, 30.0)
        ],
        {"name": "mask-p4-thr0.3", **_MASK, "refine_p": 4, "refine_threshold": 0.3},
    ],
    # Per-window candidates (options.candidates) and how one is chosen.
    "candidates": [
        {"name": "baseline-k1-chunk10", "chunk_seconds": 10.0, "window_candidates": 1},
        *[
            {"name": f"k{k}-{select}-chunk10", "chunk_seconds": 10.0, "window_candidates": k, "select": select}
            for k in (2, 3)
            for select in ("overlap", "mean")
        ],
        {"name": "k2-overlap-chunk15", "chunk_seconds": 15.0, "window_candidates": 2, "select": "overlap"},
        {"name": "k1-chunk20", "chunk_seconds": 20.0, "window_candidates": 1},
        {"name": "k2-overlap-chunk20", "chunk_seconds": 20.0, "window_candidates": 2, "select": "overlap"},
        {"name": "k3-overlap-chunk20", "chunk_seconds": 20.0, "window_candidates": 3, "select": "overlap"},
        {"name": "k2-overlap-whole", "chunk_seconds": 0.0, "window_candidates": 2, "select": "overlap"},
    ],
    # Temporal gate strength on top of the leakage threshold (30 dB never triggered in the leakage sweep).
    "gate": [
        {"name": "thr0.5-gate0", "refine": "mask", "refine_p": 2, "refine_nfft": 2048, "refine_threshold": 0.5, "refine_gate_db": 0},
        *[{"name": f"thr0.5-gate{g}", "refine": "mask", "refine_p": 2, "refine_nfft": 2048, "refine_threshold": 0.5, "refine_gate_db": g} for g in (10, 15, 20)],
    ],
    # Leakage in both directions: refine_threshold cleans the target stem, refine_residual_threshold
    # clears the target out of the residual ("Mute target" leftovers); plus passes and candidates.
    "leakage-v2": [
        {"name": "mask", **_MASK},
        {"name": "thr0.5", **_MASK, "refine_threshold": 0.5},
        {"name": "rthr0.5", **_MASK, "refine_residual_threshold": 0.5},
        {"name": "both0.5", **_MASK, "refine_threshold": 0.5, "refine_residual_threshold": 0.5},
        {"name": "both0.7", **_MASK, "refine_threshold": 0.7, "refine_residual_threshold": 0.7},
        {"name": "p4-both0.5", **_MASK, "refine_p": 4, "refine_threshold": 0.5, "refine_residual_threshold": 0.5},
        {"name": "passes2", **_MASK, "passes": 2},
        {"name": "passes2-both0.5", **_MASK, "passes": 2, "refine_threshold": 0.5, "refine_residual_threshold": 0.5},
        {"name": "k3auto-both0.5", **_MASK, "window_candidates": 3, "select": "auto", "refine_threshold": 0.5, "refine_residual_threshold": 0.5},
        {"name": "k3auto-passes2-both0.5", **_MASK, "window_candidates": 3, "select": "auto", "passes": 2,
         "refine_threshold": 0.5, "refine_residual_threshold": 0.5},
    ],
    # Chunk window length (0 = whole file in one window).
    "chunks": [
        {"name": f"chunk{chunk:g}" if chunk else "whole-file", "chunk_seconds": chunk, "window_candidates": 1}
        for chunk in (10.0, 15.0, 20.0, 0.0)
    ],
    # The first quality bundle: each improvement alone, then together.
    "quality-v1": [
        {"name": "off", "refine": "off", "window_candidates": 1},
        {"name": "mask", "refine": "mask", "refine_p": 2, "refine_nfft": 2048, "window_candidates": 1},
        {"name": "mask-k2-auto", "refine": "mask", "refine_p": 2, "refine_nfft": 2048, "window_candidates": 2, "select": "auto"},
        {"name": "mask-k3-auto", "refine": "mask", "refine_p": 2, "refine_nfft": 2048, "window_candidates": 3, "select": "auto"},
        {"name": "quality-profile", "refine": "mask", "refine_p": 2, "refine_nfft": 2048, "refine_threshold": 0.5, "window_candidates": 3, "select": "auto"},
        {"name": "mask-whole-file", "refine": "mask", "refine_p": 2, "refine_nfft": 2048, "window_candidates": 1, "chunk_seconds": 0.0},
        {"name": "mask-chunk20", "refine": "mask", "refine_p": 2, "refine_nfft": 2048, "window_candidates": 1, "chunk_seconds": 20.0},
    ],
}
