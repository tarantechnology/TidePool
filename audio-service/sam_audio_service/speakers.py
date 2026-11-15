"""Who speaks when: split the dialogue layer into one layer per speaker.

Explode's dialogue model (Demucs) gives one voice layer for everyone. This module finds
the speakers in it and cuts it into "person 1", "person 2", ... by turns:

1. Speech activity from the dialogue layer's loudness (10 ms frames, relative to its own
   loud parts, with short gaps filled and blips dropped).
2. A speaker embedding (ECAPA-TDNN, ungated) for 1.5 s windows of speech every 0.25 s.
3. Spectral clustering proposes 2 to 4 speakers; clusters whose average voices are alike
   (cosine above AUDIO_SPEAKER_SAME) merge, and speakers with too little speech fold into the
   closest one. One speaker left means no split, so a monologue stays one "dialogue" layer.
4. Every 10 ms frame goes to the speaker its covering windows match best, smoothed so turns
   don't flicker; silence goes to the nearest speech. Turn changes crossfade over a few
   milliseconds.
5. The layers are the dialogue times those masks; the last is the dialogue minus the others,
   so the speakers sum exactly to the dialogue layer.

Limits, said plainly: this assigns time, not voices. When two people talk at once, the
overlap goes whole to whoever dominates it. Very short interjections (under about half a
second) may land with the neighbouring speaker. Similar voices (two men of similar pitch,
whispering, shouting) can merge into one person.
"""
from __future__ import annotations

import importlib.util
import logging
import math
import os
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("sam_audio_service")

RATE = 16000
HOP = 160  # 10 ms frames
WINDOW_S = 1.5
WINDOW_HOP_S = 0.25


def _env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class SpeakerSplit:
    """``layers``: one (channels, frames) array per speaker, in order of first speech;
    they sum to the input. ``turns``: (start_s, end_s, speaker index) of speech."""
    layers: list[np.ndarray]
    turns: list[tuple[float, float, int]]
    seconds: list[float]


class VoiceEncoder:
    """ECAPA-TDNN speaker embeddings (SpeechBrain ``speechbrain/spkrec-ecapa-voxceleb``,
    192-d, trained on VoxCeleb). The model is public and ungated on Hugging Face; it downloads
    once (about 80 MB) into ``AUDIO_SPEAKER_CACHE``. ``available`` is False without the
    ``speechbrain`` package, and Explode then keeps one dialogue layer."""

    same = 0.70  # centroid cosine above which two clusters are one person (AUDIO_SPEAKER_SAME)

    def __init__(self, source: str | None = None, device: str | None = None):
        self.source = source or os.getenv("AUDIO_SPEAKER_MODEL", "speechbrain/spkrec-ecapa-voxceleb")
        self.cache = os.getenv("AUDIO_SPEAKER_CACHE", os.path.expanduser("~/.cache/tidepool/spkrec-ecapa-voxceleb"))
        self.device = device or os.getenv("AUDIO_SPEAKER_DEVICE")
        self.model = None

    @property
    def available(self) -> bool:
        return importlib.util.find_spec("speechbrain") is not None and importlib.util.find_spec("torch") is not None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError:  # speechbrain < 1.0
            from speechbrain.pretrained import EncoderClassifier

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EncoderClassifier.from_hparams(source=self.source, savedir=self.cache, run_opts={"device": self.device})
        self.model.eval()

    def embed(self, windows: np.ndarray) -> np.ndarray:
        """(n, samples) mono 16 kHz windows -> (n, d) unit embeddings."""
        import torch

        self.load()
        out = []
        with torch.inference_mode():
            for i in range(0, len(windows), 64):
                x = torch.as_tensor(np.asarray(windows[i:i + 64]), dtype=torch.float32, device=self.device)
                e = self.model.encode_batch(x).reshape(x.shape[0], -1).float()
                out.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 192), np.float32)


def _resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return x.astype(np.float32)
    import torch
    import torchaudio

    return torchaudio.functional.resample(torch.as_tensor(x, dtype=torch.float32), src, dst).numpy()


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(d[0::2], d[1::2]))


def speech_frames(mono: np.ndarray, floor_db: float | None = None) -> np.ndarray:
    """Boolean per 10 ms frame: is someone speaking in this (dialogue-only) audio."""
    n = len(mono) // HOP
    if n == 0:
        return np.zeros(0, bool)
    frames = mono[: n * HOP].reshape(n, HOP)
    db = 10 * np.log10(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-10)
    # Smooth over 50 ms so syllable gaps don't chop words.
    k = np.ones(5) / 5
    db = np.convolve(db, k, mode="same")
    floor_db = floor_db if floor_db is not None else _env("AUDIO_SPEAKER_FLOOR_DB", -28.0)
    loud = np.percentile(db, 95)
    active = db > max(loud + floor_db, -60.0)
    # Fill gaps under 250 ms, then drop runs under 200 ms.
    for s, e in _runs(~active):
        if e - s < 25 and s > 0 and e < n:
            active[s:e] = True
    for s, e in _runs(active):
        if e - s < 20:
            active[s:e] = False
    return active


def _cluster(emb: np.ndarray, max_speakers: int, p_keep: float = 0.95) -> np.ndarray:
    """Spectral clustering of unit embeddings (the d-vector recipe of Wang et al. 2018):
    cosine affinity, each row keeps only its strongest links, symmetrised and diffused, then
    the number of speakers is the largest eigengap of the normalised Laplacian (2 to
    ``max_speakers``) and k-means on the eigenvectors labels the windows."""
    n = len(emb)
    if n < 4:
        return np.zeros(n, int)
    A = np.clip(emb @ emb.T, 0, 1)
    np.fill_diagonal(A, A.max(1))
    # Row-wise pruning: keep the top (1 - p_keep) share of each row, shrink the rest.
    keep = max(2, int(round(n * (1 - p_keep))))
    cut = -np.partition(-A, keep - 1, axis=1)[:, keep - 1:keep]
    A = np.where(A >= cut, A, A * 0.01)
    A = np.maximum(A, A.T)
    A = A @ A.T
    A /= A.max(1, keepdims=True) + 1e-12
    A = (A + A.T) / 2
    d = A.sum(1)
    L = np.eye(n) - A / np.sqrt(np.outer(d, d) + 1e-12)
    vals, vecs = np.linalg.eigh(L)
    top = min(max_speakers, n - 1)
    gaps = np.diff(vals[: top + 1])
    k = int(np.argmax(gaps[1:]) + 2) if top >= 2 else 1
    V = vecs[:, :k]
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    return _kmeans(V, k)


def _kmeans(X: np.ndarray, k: int, iters: int = 50, restarts: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    best, best_cost = None, np.inf
    for _ in range(restarts):
        # k-means++ seeding.
        c = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            dist = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), 1)
            c.append(X[rng.choice(len(X), p=dist / dist.sum())] if dist.sum() > 0 else X[rng.integers(len(X))])
        C = np.array(c)
        for _ in range(iters):
            lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        cost = ((X - C[lab]) ** 2).sum()
        if cost < best_cost:
            best, best_cost = lab, cost
    return best


def _centroid_sims(emb: np.ndarray, labels: np.ndarray) -> tuple[list[int], np.ndarray]:
    ids = [int(k) for k in np.unique(labels)]
    c = np.stack([emb[labels == k].mean(0) for k in ids])
    c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
    return ids, c @ c.T


def _merge_close(emb: np.ndarray, labels: np.ndarray, same: float) -> np.ndarray:
    """Merge clusters whose mean voices are more alike than ``same`` (cosine): the clustering
    always proposes at least two speakers, and this is what says one person was talking."""
    labels = labels.copy()
    while True:
        ids, sim = _centroid_sims(emb, labels)
        if len(ids) < 2:
            return labels
        np.fill_diagonal(sim, -1)
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        log.debug("speaker centroid similarity %.3f", sim[i, j])
        if sim[i, j] < same:
            return labels
        labels[labels == ids[j]] = ids[i]


def _smooth_labels(labels: np.ndarray, speakers: int, width: int) -> np.ndarray:
    """Majority vote over a sliding window (a median filter for categories)."""
    onehot = np.eye(speakers)[labels]
    k = np.ones(width) / width
    votes = np.stack([np.convolve(onehot[:, s], k, mode="same") for s in range(speakers)], 1)
    return votes.argmax(1)


def split_speakers(dialogue: np.ndarray, sample_rate: int, encoder: VoiceEncoder,
                   threshold: float | None = None, max_speakers: int | None = None,
                   min_share: float | None = None, min_seconds: float | None = None) -> SpeakerSplit:
    """Split ``dialogue`` (channels, frames) by speaker. One layer back means one speaker."""
    threshold = threshold if threshold is not None else _env("AUDIO_SPEAKER_SAME", getattr(encoder, "same", 0.70))
    max_speakers = int(max_speakers if max_speakers is not None else _env("AUDIO_SPEAKER_MAX", 4))
    min_share = min_share if min_share is not None else _env("AUDIO_SPEAKER_MIN_SHARE", 0.08)
    min_seconds = min_seconds if min_seconds is not None else _env("AUDIO_SPEAKER_MIN_SECONDS", 2.0)
    dialogue = np.asarray(dialogue, dtype=np.float32)
    whole = SpeakerSplit([dialogue], [], [])
    mono = _resample(dialogue.mean(0), sample_rate, RATE)
    active = speech_frames(mono)
    n_frames = len(active)
    if active.sum() * HOP / RATE < 2 * min_seconds:
        return whole

    # Windows centred on speech, 1.5 s long every 0.25 s, kept when at least half is speech.
    win = int(WINDOW_S * RATE // HOP)
    step = int(WINDOW_HOP_S * RATE // HOP)
    starts = [s for s in range(0, max(1, n_frames - win + 1), step) if active[s:s + win].mean() >= 0.5]
    if len(starts) > 1200:  # long clips: keep clustering cheap
        starts = starts[:: math.ceil(len(starts) / 1200)]
    if len(starts) < 4:
        return whole
    padded = np.pad(mono, (0, max(0, (starts[-1] + win) * HOP - len(mono))))
    windows = np.stack([padded[s * HOP:(s + win) * HOP] for s in starts])
    emb = encoder.embed(windows)
    labels = _cluster(emb, max_speakers)
    labels = _merge_close(emb, labels, threshold)

    # Fold speakers with too little speech into their closest neighbour.
    window_s = WINDOW_HOP_S
    while True:
        ids = np.unique(labels)
        if len(ids) == 1:
            break
        talk = {k: (labels == k).sum() * window_s for k in ids}
        total = sum(talk.values())
        small = min(ids, key=lambda k: talk[k])
        if talk[small] >= min_seconds and talk[small] >= min_share * total:
            break
        cent = {k: emb[labels == k].mean(0) for k in ids}
        others = [k for k in ids if k != small]
        nearest = max(others, key=lambda k: float(cent[small] @ cent[k] / (np.linalg.norm(cent[small]) * np.linalg.norm(cent[k]) + 1e-9)))
        labels[labels == small] = nearest
    ids = list(np.unique(labels))
    if len(ids) < 2:
        return whole

    # Per-frame scores: similarity of each covering window to each speaker's centroid.
    cents = np.stack([emb[labels == k].mean(0) for k in ids])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9
    sims = emb @ cents.T  # (windows, speakers)
    score = np.zeros((n_frames, len(ids)))
    cover = np.zeros(n_frames)
    # A window's evidence is strongest in its middle.
    taper = np.hanning(win + 2)[1:-1]
    for s, row in zip(starts, sims):
        e = min(n_frames, s + win)
        score[s:e] += taper[: e - s, None] * row[None, :]
        cover[s:e] += taper[: e - s]
    frame_label = np.full(n_frames, -1)
    has = (cover > 0) & active
    frame_label[has] = score[has].argmax(1)
    # Speech frames no window covered, then silence: take the nearest labelled frame.
    known = np.flatnonzero(frame_label >= 0)
    if len(known) == 0:
        return whole
    pos = np.searchsorted(known, np.arange(n_frames)).clip(1, len(known) - 1)
    left, right = known[pos - 1], known[pos]
    take = np.where(np.abs(np.arange(n_frames) - left) <= np.abs(right - np.arange(n_frames)), left, right)
    take[np.arange(n_frames) <= known[0]] = known[0]
    take[np.arange(n_frames) >= known[-1]] = known[-1]
    frame_label = frame_label[take]
    frame_label = _smooth_labels(frame_label, len(ids), 31)  # 310 ms

    # Order speakers by first speech; drop any that the smoothing emptied.
    spoken = frame_label[active]
    present, first = np.unique(spoken, return_index=True)
    order = [int(present[i]) for i in np.argsort(first)]
    if len(order) < 2:
        return whole
    remap = np.zeros(len(ids), int)
    remap[order] = np.arange(len(order))
    frame_label = remap[frame_label]

    # Frame masks -> sample masks at the dialogue rate, crossfaded over 30 ms.
    k = len(order)
    masks = np.eye(k)[frame_label].T  # (k, n_frames)
    fade = max(1, int(0.03 * RATE / HOP))
    kernel = np.ones(fade) / fade
    masks = np.stack([np.convolve(m, kernel, mode="same") for m in masks])
    masks /= masks.sum(0, keepdims=True)
    frames_total = dialogue.shape[-1]
    t_frames = (np.arange(n_frames) + 0.5) * HOP / RATE
    t = np.arange(frames_total) / sample_rate
    layers = []
    for m in masks[:-1]:
        layers.append((dialogue * np.interp(t, t_frames, m)[None, :]).astype(np.float32))
    layers.append((dialogue - sum(layers)).astype(np.float32))  # exact: the speakers sum to the dialogue

    turns: list[tuple[float, float, int]] = []
    for s, e in _runs(active):
        seg = frame_label[s:e]
        cuts = np.flatnonzero(np.diff(seg)) + 1
        for a, b in zip(np.concatenate([[0], cuts]), np.concatenate([cuts, [len(seg)]])):
            turns.append((round((s + a) * HOP / RATE, 2), round((s + b) * HOP / RATE, 2), int(seg[a])))
    seconds = [float(round(sum(b - a for a, b, sp in turns if sp == i), 1)) for i in range(k)]
    return SpeakerSplit(layers, turns, seconds)
