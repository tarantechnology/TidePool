"""Instrument tags and descriptive names for music stems.

Demucs's fourth stem is "other": whatever is not vocals, drums or bass. Calling that track
"Other" tells the user nothing, so after a split every stem is run through the router's
AudioSet tagger (PANNs Cnn14 or AST, already loaded for routing; no second model):

* ``tags``: up to ``MAX_TAGS`` instrument tags per stem, ``{"label", "score"}`` with the
  score the mean probability over the stem's active windows, each ``>= min_score``, sorted
  by score. Labels are human display names (``DISPLAY``), not raw AudioSet names.
* ``name`` of the canonical ``other`` stem: its top tag, or ``"A & B"`` when the second tag
  scores at least ``PAIR_RATIO`` of the first ("Synth & Strings"). ``"Other"`` when the stem
  is near-silent or no tag clears ``min_score``.

Windows: the stem is cut into 10 s windows; a window is *active* when it is within
``ACTIVE_STEM_DB`` of the stem's loudest window and within ``ACTIVE_MIX_DB`` of the mixture
in the same window (so bleed in a verse where the stem is silent is not analysed). Up to
``max_windows`` active windows spread across the song are scored; each is level-normalised
before tagging, since a quiet stem would otherwise read as "silence" to the tagger.

The whitelist is the AudioSet "Musical instrument" subtree, the singing classes, and
Beatboxing / Scratching, by name: PANNs (``class_labels_indices.csv``) and AST
(``model.config.id2label``) use the same 527 names. No ontology is downloaded at runtime.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

MAX_TAGS = 3
PAIR_RATIO = 0.6  # second tag >= 60% of the first -> "A & B"
SILENT_DB = -45.0  # stem energy relative to the mixture below which the stem is "near-silent"
ACTIVE_STEM_DB = -20.0  # window vs the stem's loudest window
ACTIVE_MIX_DB = -30.0  # window vs the mixture in the same window
WINDOW_SECONDS = 10.0
TARGET_RMS_DB = -20.0  # each analysed window is normalised to this RMS (dBFS) ...
MAX_GAIN_DB = 30.0  # ... with at most this much gain
CHILD_RATIO = 0.5  # a child tag replaces its parent when it scores >= 50% of the parent
FALLBACK_NAME = "Other"

# AudioSet class name -> display name. Several classes may share a display name; their
# scores are merged with max().
DISPLAY: dict[str, str] = {
    # Singing (AudioSet "Human voice" > "Singing") and other vocal performance.
    "Singing": "Vocals",
    "Male singing": "Male singing",
    "Female singing": "Female singing",
    "Child singing": "Child singing",
    "Synthetic singing": "Synth vocals",
    "Choir": "Choir",
    "Yodeling": "Yodeling",
    "Chant": "Chant",
    "Rapping": "Rapping",
    "Humming": "Humming",
    "Beatboxing": "Beatboxing",
    # Musical instrument subtree.
    "Musical instrument": "Instruments",
    "Plucked string instrument": "Plucked strings",
    "Guitar": "Guitar",
    "Electric guitar": "Electric guitar",
    "Bass guitar": "Bass guitar",
    "Acoustic guitar": "Acoustic guitar",
    "Steel guitar, slide guitar": "Slide guitar",
    "Tapping (guitar technique)": "Guitar",
    "Strum": "Guitar",
    "Banjo": "Banjo",
    "Sitar": "Sitar",
    "Mandolin": "Mandolin",
    "Zither": "Zither",
    "Ukulele": "Ukulele",
    "Keyboard (musical)": "Keys",
    "Piano": "Piano",
    "Electric piano": "Electric piano",
    "Organ": "Organ",
    "Electronic organ": "Organ",
    "Hammond organ": "Organ",
    "Synthesizer": "Synth",
    "Sampler": "Sampler",
    "Harpsichord": "Harpsichord",
    "Percussion": "Percussion",
    "Drum kit": "Drum kit",
    "Drum machine": "Drum machine",
    "Drum": "Drums",
    "Snare drum": "Snare",
    "Rimshot": "Snare",
    "Drum roll": "Drums",
    "Bass drum": "Kick drum",
    "Timpani": "Timpani",
    "Tabla": "Tabla",
    "Cymbal": "Cymbals",
    "Hi-hat": "Hi-hat",
    "Wood block": "Wood block",
    "Tambourine": "Tambourine",
    "Rattle (instrument)": "Shaker",
    "Maraca": "Shaker",
    "Gong": "Gong",
    "Tubular bells": "Bells",
    "Mallet percussion": "Mallets",
    "Marimba, xylophone": "Marimba",
    "Glockenspiel": "Glockenspiel",
    "Vibraphone": "Vibraphone",
    "Steelpan": "Steel drums",
    "Orchestra": "Orchestra",
    "Brass instrument": "Brass",
    "French horn": "Horn",
    "Trumpet": "Trumpet",
    "Trombone": "Trombone",
    "Bowed string instrument": "Strings",
    "String section": "Strings",
    "Violin, fiddle": "Violin",
    "Pizzicato": "Strings",
    "Cello": "Cello",
    "Double bass": "Double bass",
    "Wind instrument, woodwind instrument": "Woodwinds",
    "Flute": "Flute",
    "Saxophone": "Saxophone",
    "Clarinet": "Clarinet",
    "Harp": "Harp",
    "Bell": "Bells",
    "Church bell": "Bells",
    "Change ringing (campanology)": "Bells",
    "Jingle bell": "Sleigh bells",
    "Chime": "Chimes",
    "Wind chime": "Wind chimes",
    "Harmonica": "Harmonica",
    "Accordion": "Accordion",
    "Bagpipes": "Bagpipes",
    "Didgeridoo": "Didgeridoo",
    "Theremin": "Theremin",
    "Singing bowl": "Singing bowl",
    "Scratching (performance technique)": "Scratching",
}

# Never a name or a tag while anything more specific clears the threshold.
GENERIC: frozenset[str] = frozenset({"Instruments", "Plucked strings", "Keys", "Percussion", "Sampler"})
# Never a stem name or tag ("Instruments" says as little as "Other").
UNNAMEABLE: frozenset[str] = frozenset({"Instruments"})
VOCAL: frozenset[str] = frozenset({
    "Vocals", "Male singing", "Female singing", "Child singing", "Synth vocals", "Choir", "Yodeling", "Chant", "Rapping", "Humming", "Beatboxing",
})
# parent display name -> more specific display names. When both clear the threshold, a child
# scoring >= CHILD_RATIO of the parent replaces it; otherwise the (less sure) child is dropped.
CHILDREN: dict[str, frozenset[str]] = {
    "Vocals": frozenset({"Male singing", "Female singing", "Child singing", "Synth vocals", "Choir", "Yodeling", "Chant", "Rapping"}),
    "Guitar": frozenset({"Electric guitar", "Acoustic guitar", "Slide guitar", "Bass guitar"}),
    "Drums": frozenset({"Drum kit", "Snare", "Kick drum", "Timpani", "Tabla"}),
    "Brass": frozenset({"Horn", "Trumpet", "Trombone"}),
    "Woodwinds": frozenset({"Flute", "Saxophone", "Clarinet"}),
    "Mallets": frozenset({"Marimba", "Glockenspiel", "Vibraphone"}),
    "Bells": frozenset({"Sleigh bells", "Chimes", "Wind chimes"}),
    "Strings": frozenset({"Violin", "Cello", "Double bass"}),
}

# Per canonical stem: display names that say nothing new about that stem ("Drums" on the
# drums track) and are left out of its tags. Vocal classes are left out of instrument stems,
# where they are bleed from the vocal track rather than something the user would want named.
_NO_VOCALS = VOCAL - {"Beatboxing"}
TRIVIAL: dict[str, frozenset[str]] = {
    "vocals": frozenset({"Vocals"}),
    "drums": frozenset({"Drums", "Percussion"}) | _NO_VOCALS,
    "bass": frozenset({"Guitar", "Plucked strings"}) | _NO_VOCALS,
    "guitar": frozenset({"Guitar", "Plucked strings"}) | _NO_VOCALS,
    "piano": frozenset({"Piano", "Keys"}) | _NO_VOCALS,
    "other": _NO_VOCALS,
    "instrumental": _NO_VOCALS,
}


class Tagger(Protocol):
    """What the router's AudioSet tagger offers labeling (``classifier._AudioSetTagger``)."""

    def class_names(self) -> list[str]: ...

    def tag(self, clips: list[np.ndarray], sample_rate: int) -> np.ndarray:
        """``(len(clips), n_classes)`` probabilities for mono clips at ``sample_rate``."""
        ...


@dataclass
class StemLabel:
    tags: list[dict[str, float | str]]
    name: str | None  # descriptive name (always computed; the engine decides whether to use it)
    candidates: list[tuple[str, float]] = field(default_factory=list)  # every whitelisted display name >= min_score
    energy_db: float = 0.0  # stem energy relative to the mixture
    windows: int = 0


def _energy(x: np.ndarray) -> float:
    flat = np.ascontiguousarray(x, dtype=np.float32).ravel()
    return float(np.dot(flat, flat))


def _db(x: float, ref: float) -> float:
    if x <= 0 or ref <= 0:
        return -120.0
    return 10.0 * float(np.log10(x / ref))


def _mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    if x.shape[0] == 2:
        return (x[0] + x[1]) * np.float32(0.5)
    return x.mean(axis=0, dtype=np.float32)


def window_energies(mono: np.ndarray, size: int) -> tuple[list[int], np.ndarray]:
    """Starts and mean-square energy of the non-overlapping ``size``-sample windows covering
    ``mono`` (the last one aligned to the end, so it is full length)."""
    n = len(mono)
    if n <= size:
        return [0], np.array([float(np.mean(np.square(mono, dtype=np.float64))) if n else 0.0])
    full = n // size
    sq = np.square(mono, dtype=np.float32)
    energy = sq[: full * size].reshape(full, size).mean(axis=1, dtype=np.float64)
    starts = list(range(0, full * size, size))
    if full * size < n:
        starts.append(n - size)
        energy = np.append(energy, float(sq[n - size :].mean(dtype=np.float64)))
    return starts, energy


def active_windows(stem: np.ndarray, mix_energy: np.ndarray, sample_rate: int, max_windows: int, window_seconds: float = WINDOW_SECONDS) -> list[np.ndarray]:
    """Up to ``max_windows`` active mono windows of ``stem``, spread evenly across the song.
    ``mix_energy`` is ``window_energies`` of the mixture's mono with the same window size."""
    size = int(window_seconds * sample_rate)
    mono = _mono(stem)
    starts, energy = window_energies(mono, size)
    peak = float(energy.max()) if energy.size else 0.0
    if peak <= 0:
        return []
    active = [i for i in range(len(starts)) if _db(energy[i], peak) >= ACTIVE_STEM_DB and _db(energy[i], mix_energy[i]) >= ACTIVE_MIX_DB]
    if not active:
        active = [int(np.argmax(energy))]
    if len(active) > max_windows:
        picks = np.linspace(0, len(active) - 1, num=max_windows).round().astype(int)
        active = [active[i] for i in sorted(set(picks.tolist()))]
    clips = []
    for i in active:
        clip = mono[starts[i] : starts[i] + size].astype(np.float32)
        rms = float(np.sqrt(energy[i]))
        if rms > 0:
            gain_db = min(MAX_GAIN_DB, TARGET_RMS_DB - 20 * np.log10(rms))
            clip = np.clip(clip * np.float32(10 ** (gain_db / 20)), -1.0, 1.0)
        clips.append(clip)
    return clips


def whitelist(class_names: Sequence[str]) -> dict[str, list[int]]:
    """Display name -> indices of the tagger's classes that map to it."""
    out: dict[str, list[int]] = {}
    for i, name in enumerate(class_names):
        display = DISPLAY.get(name)
        if display is not None:
            out.setdefault(display, []).append(i)
    return out


def candidates(mean_probs: np.ndarray, classes: dict[str, list[int]], min_score: float) -> list[tuple[str, float]]:
    """Whitelisted display names scoring ``>= min_score``, generic parents and redundant
    parent/child pairs resolved, sorted by score (desc)."""
    scores = {d: float(max(mean_probs[i] for i in idx)) for d, idx in classes.items()}
    kept = {d: s for d, s in scores.items() if s >= min_score}
    for parent, children in CHILDREN.items():
        if parent not in kept:
            continue
        present = [c for c in children if c in kept]
        if not present:
            continue
        strong = [c for c in present if kept[c] >= CHILD_RATIO * kept[parent]]
        if strong:
            del kept[parent]
        for c in present:
            if c not in strong:
                del kept[c]
    if any(d not in GENERIC for d in kept):
        kept = {d: s for d, s in kept.items() if d not in GENERIC}
    return sorted(kept.items(), key=lambda kv: (-kv[1], kv[0]))


def describe(cands: list[tuple[str, float]]) -> str | None:
    """Descriptive name from (already filtered) candidates: top, or "A & B"; None if nothing."""
    names = [(d, s) for d, s in cands if d not in UNNAMEABLE]
    if not names:
        return None
    (first, s1), rest = names[0], names[1:]
    if rest and rest[0][1] >= PAIR_RATIO * s1:
        return f"{first} & {rest[0][0]}"
    return first


def label_stems(
    tagger: Tagger,
    stems: Sequence[tuple[str, np.ndarray]],
    mixture: np.ndarray,
    sample_rate: int,
    *,
    min_score: float,
    max_windows: int = 12,
) -> list[StemLabel]:
    """Label each ``(canonical, samples)`` stem. All windows go to the tagger in one call."""
    classes = whitelist(tagger.class_names())
    mix_energy = _energy(mixture)
    _, mix_windows = window_energies(_mono(mixture), int(WINDOW_SECONDS * sample_rate))
    plans: list[tuple[str, float, list[np.ndarray]]] = []
    for canonical, samples in stems:
        energy_db = _db(_energy(samples), mix_energy)
        clips = [] if energy_db < SILENT_DB else active_windows(samples, mix_windows, sample_rate, max_windows)
        plans.append((canonical, energy_db, clips))
    flat = [c for _, _, clips in plans for c in clips]
    probs = tagger.tag(flat, sample_rate) if flat else np.zeros((0, 0))
    out: list[StemLabel] = []
    offset = 0
    for canonical, energy_db, clips in plans:
        if not clips:
            out.append(StemLabel([], None, [], round(energy_db, 2), 0))
            continue
        mean = np.asarray(probs[offset : offset + len(clips)], dtype=np.float64).mean(axis=0)
        offset += len(clips)
        cands = candidates(mean, classes, min_score)
        trivial = TRIVIAL.get(canonical, frozenset())
        tags = [{"label": d, "score": round(s, 3)} for d, s in cands if d not in trivial and d not in UNNAMEABLE][:MAX_TAGS]
        name_cands = [(d, s) for d, s in cands if d not in _NO_VOCALS] if canonical != "vocals" else cands
        out.append(StemLabel(tags, describe(name_cands), cands, round(energy_db, 2), len(clips)))
    return out
