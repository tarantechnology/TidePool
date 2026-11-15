"""Hear which sounds are in a clip, so Explode can suggest its own list.

The backend scores 10-second windows of the mixture against a fixed vocabulary of
separation-ready descriptions (AudioSep does this with the CLAP encoder it already
uses for text prompts: audio and text embed into one space). Per window the scores
become probabilities over the vocabulary. A sound is suggested when it is the most
likely sound in some window, or likely enough anywhere; one per group (one music,
one voice), strongest first.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (prompt sent to the separator, group, tier). Group dedupes near-synonyms; tier is
# the Explode ordering the editor applies (background, hits, other, music, voices).
VOCABULARY: tuple[tuple[str, str, str], ...] = (
    ("heavy rain", "rain", "background"),
    ("light rain dripping", "rain", "background"),
    ("wind blowing", "wind", "background"),
    ("ocean waves", "water", "background"),
    ("river or stream flowing", "water", "background"),
    ("crowd chatter", "crowd", "background"),
    ("cheering crowd and applause", "crowd", "background"),
    ("city traffic", "traffic", "background"),
    ("fire crackling", "fire", "background"),
    ("birds chirping", "nature", "background"),
    ("insects and crickets at night", "nature", "background"),
    ("engine humming", "engine", "background"),
    ("electrical hum and static", "hum", "background"),
    ("thunder rumbling", "thunder", "hits"),
    ("explosion", "explosion", "hits"),
    ("gunshots", "gunshot", "hits"),
    ("punches and fight impacts", "impact", "hits"),
    ("metal clanging and crashing", "impact", "hits"),
    ("glass breaking", "glass", "hits"),
    ("footsteps", "footsteps", "hits"),
    ("door slamming", "door", "hits"),
    ("whoosh sound effect", "whoosh", "hits"),
    ("high-pitched screech", "screech", "hits"),
    ("siren wailing", "siren", "hits"),
    ("car driving by", "vehicle", "hits"),
    ("helicopter", "vehicle", "hits"),
    ("dog barking", "animal", "hits"),
    ("screaming", "scream", "hits"),
    ("heavy breathing", "breathing", "hits"),
    ("phone ringing", "phone", "hits"),
    ("orchestral music", "music", "music"),
    ("tense cinematic music", "music", "music"),
    ("electronic music", "music", "music"),
    ("rock music with drums", "music", "music"),
    ("piano music", "music", "music"),
    ("people talking", "speech", "voices"),
    ("deep male voice speaking", "speech", "voices"),
    ("woman speaking", "speech", "voices"),
    ("person singing", "singing", "voices"),
    ("laughter", "laughter", "voices"),
)

WINDOW_SECONDS = 10.0


@dataclass
class HeardSound:
    prompt: str
    group: str
    tier: str
    score: float  # highest probability in any window, 0..1
    windows: list[int]  # windows (10 s each, from 0) where it is the top sound or likely


def windows_of(mono: np.ndarray, sample_rate: int, seconds: float = WINDOW_SECONDS) -> list[np.ndarray]:
    """Cut mono audio into full-length windows; the last is zero-padded. At least one."""
    size = max(1, int(round(seconds * sample_rate)))
    count = max(1, int(np.ceil(len(mono) / size)))
    padded = np.zeros(count * size, dtype=np.float32)
    padded[: len(mono)] = mono
    return [padded[i * size : (i + 1) * size] for i in range(count)]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


# Least window probability for a sound to be suggested, by tier. Per-window probabilities
# are shared out over ~40 sounds, so a clip full of rain and screech "crowds out" speech
# and music even when both are plainly there: they get a low bar (and a rank rule below).
# Hits get a high one, since one-off thumps are easy to mistake for each other.
# Measured on the Batman v Superman clip on AudioSep's CLAP: speech and music scored
# under 0.12 although present; screech 0.55 and punches 0.43 were right, gunshots 0.41,
# door 0.29 and footsteps 0.22 were not.
TIER_MIN_PROB = {"voices": 0.02, "music": 0.02, "background": 0.12, "hits": 0.42, "other": 0.2}
# Speech and music also count when they rank this high in any window.
ALWAYS_RANK = {"voices": 5, "music": 5}


def tier_thresholds() -> dict[str, float]:
    import os
    out = dict(TIER_MIN_PROB)
    for tier in out:
        raw = os.getenv(f"AUDIO_TAG_MIN_PROB_{tier.upper()}")
        if raw:
            try: out[tier] = float(raw)
            except ValueError: pass
    return out


def pick(similarity: np.ndarray, scale: float, *, max_sounds: int = 6, min_prob: dict[str, float] | float | None = None,
         active: np.ndarray | None = None, vocabulary=VOCABULARY) -> list[HeardSound]:
    """Choose sounds from a [windows, vocabulary] cosine-similarity matrix.

    A sound qualifies by its tier's threshold (``min_prob``; a float applies to every
    tier), by being the top sound of a window (not for hits), or for speech and music by
    ranking high in some window. One per group. Voices and music are kept first, then the
    rest strongest first, up to ``max_sounds``. ``active`` marks windows with audible
    signal; silent padding never votes.
    """
    sims = np.asarray(similarity, dtype=np.float64)
    if sims.ndim != 2 or sims.shape[1] != len(vocabulary):
        raise ValueError("similarity must be [windows, vocabulary]")
    keep = np.ones(sims.shape[0], dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if not keep.any():
        return []
    limits = tier_thresholds() if min_prob is None else ({t: float(min_prob) for t in TIER_MIN_PROB} if isinstance(min_prob, (int, float)) else {**TIER_MIN_PROB, **min_prob})
    active_sims=sims[keep]
    probs = softmax(active_sims * scale, axis=1)
    baseline=np.median(active_sims,axis=1)
    index = np.flatnonzero(keep)
    ranks = (-probs).argsort(axis=1).argsort(axis=1)  # 0 = top sound of that window
    best: dict[str, HeardSound] = {}
    for j, (prompt, group, tier) in enumerate(vocabulary):
        bar = limits.get(tier, 0.2)
        top_counts = tier != "hits"
        # Do not turn tied, unsupported labels into detections. Compare the
        # unscaled similarity to the window's median: a crowded clip can push
        # real speech below the softmax's uniform probability baseline.
        hits = [int(index[w]) for w in range(probs.shape[0])
                if active_sims[w, j] > baseline[w] and
                (probs[w, j] >= bar or (top_counts and ranks[w, j] == 0) or ranks[w, j] < ALWAYS_RANK.get(tier, 0))]
        if not hits:
            continue
        sound = HeardSound(prompt, group, tier, float(probs[:, j].max()), hits)
        if group not in best or sound.score > best[group].score:
            best[group] = sound
    chosen = sorted(best.values(), key=lambda s: -s.score)
    core = [s for s in chosen if s.tier in ALWAYS_RANK]
    rest = [s for s in chosen if s.tier not in ALWAYS_RANK]
    # One voice and one music layer at most is what Explode can make cleanly.
    core = [next(s for s in core if s.tier == t) for t in ("voices", "music") if any(s.tier == t for s in core)]
    return (core + rest)[:max_sounds]


# Second opinion from the AudioSet tagger (PANNs, already loaded for routing). AudioSep's CLAP
# shares probability out over ~40 descriptions, so in a clip of only speech and music something
# still comes out on top: on a talking-over-music clip it suggested "fire crackling" and
# "door slamming", and Explode named layers after them. PANNs scores each class on its own, so
# a sound is kept only if its AudioSet classes are actually likely somewhere in the clip.
# Voices and music are trusted to CLAP's rank rule. Groups with no mapping are left alone.
AUDIOSET_CLASSES: dict[str, tuple[str, ...]] = {
    "rain": ("Rain", "Raindrop", "Rain on surface"),
    "wind": ("Wind", "Wind noise (microphone)", "Rustling leaves"),
    "water": ("Ocean", "Waves, surf", "Stream", "Water", "Gurgling"),
    "crowd": ("Crowd", "Chatter", "Hubbub, speech noise, speech babble", "Cheering", "Applause"),
    "traffic": ("Traffic noise, roadway noise", "Vehicle", "Car", "Motor vehicle (road)"),
    "fire": ("Fire", "Crackle"),
    "nature": ("Bird", "Bird vocalization, bird call, bird song", "Chirp, tweet", "Insect", "Cricket"),
    "engine": ("Engine", "Idling", "Motor vehicle (road)", "Vehicle"),
    "hum": ("Hum", "Mains hum", "Static", "Buzz"),
    "thunder": ("Thunder", "Thunderstorm"),
    "explosion": ("Explosion", "Boom", "Eruption"),
    "gunshot": ("Gunshot, gunfire", "Machine gun", "Cap gun", "Artillery fire"),
    "impact": ("Smash, crash", "Thump, thud", "Slap, smack", "Whack, thwack", "Bang", "Breaking", "Clang", "Thunk"),
    "glass": ("Glass", "Shatter", "Chink, clink"),
    "footsteps": ("Walk, footsteps", "Run"),
    "door": ("Door", "Slam", "Sliding door", "Knock"),
    "whoosh": ("Whoosh, swoosh, swish",),
    "screech": ("Squeal", "Screech", "Whistle", "Sine wave", "Siren", "Alarm", "Fire alarm", "Power tool", "Drill",
                "Dental drill, dentist's drill", "Tools", "Train", "Rail transport", "Vehicle", "Aircraft", "Jet engine",
                "Helicopter", "Feedback", "Beep, bleep", "Mechanisms", "Ratchet, pawl", "Gears", "Whir"),
    # On 10 s windows of the Batman v Superman sonic trap, PANNs hears Mechanisms 0.42, Ratchet 0.30.
    "siren": ("Siren", "Civil defense siren", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)"),
    "vehicle": ("Vehicle", "Car", "Car passing by", "Helicopter", "Aircraft", "Motor vehicle (road)"),
    "animal": ("Dog", "Bark", "Animal", "Domestic animals, pets"),
    "scream": ("Screaming", "Yell", "Shout"),
    "breathing": ("Breathing", "Pant", "Gasp"),
    "phone": ("Telephone", "Telephone bell ringing", "Ringtone"),
}
VERIFY_MIN_PROB = 0.08


def verify(heard: list[HeardSound], probs: np.ndarray, class_names: list[str], min_prob: float = VERIFY_MIN_PROB) -> list[HeardSound]:
    """Drop suggestions the AudioSet tagger does not hear: ``probs`` is [windows, classes]."""
    if probs is None or len(probs) == 0:
        return list(heard)
    index = {name: i for i, name in enumerate(class_names)}
    out = []
    for h in heard:
        classes = [index[c] for c in AUDIOSET_CLASSES.get(h.group, ()) if c in index]
        if h.tier in ("voices", "music") or not classes or float(np.max(probs[:, classes])) >= min_prob:
            out.append(h)
    return out
