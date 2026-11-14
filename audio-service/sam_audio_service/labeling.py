"""Instrument tags, descriptive names and instrument presence for music stems.

After a split every stem is tagged; the canonical ``other`` stem also gets a descriptive name
("Other" tells the user nothing). The default tagger is the router's AudioSet model (PANNs
Cnn14, already loaded for routing: no second model, ~0.1 s for four stems of a 4-min song);
``AUDIO_STEM_LABEL_MODEL`` can pick AST or a zero-shot CLAP model instead (``classifier.py``).

What changed from "top AudioSet classes of the stem" (2026-11-14 evaluation on BabySlakh and
MUSDB stems separated by our chain; numbers in ``docs/plans/music-stems.md``):

* **Per-stem vocabularies** (``STEM_VOCAB``). A separated stem is a strong prior: the vocals
  stem is only ever described with voice labels, the bass stem with bass labels, the drums stem
  with drum labels, ``other`` with melodic instruments. The free-for-all whitelist produced
  "Gong"/"Didgeridoo" on bass stems and "Chant"/"Humming"/"Yodeling" on sung pop vocals.
* **Voice evidence is pooled.** On isolated vocals AudioSet spreads singing over "A capella",
  "Vocal music", "Mantra", "Chant", "Speech" and the gendered classes; the generic "Singing"
  class is low (PANNs mean 0.07 on MUSDB vocals) and the old parent/child rule then kept
  whatever odd child survived ("Chant"). Now "Chant"/"Mantra"/"A capella"/... only count as
  evidence that someone sings; the tags are Male/Female singing (by margin), Rapping, Choir,
  Synth vocals, or plain "Singing".
* **Stems always get their family tag** (Drum kit / Drum machine, Bass guitar / Synth bass /
  Double bass, Singing) when they are not near-silent; before, a drums stem whose only
  confident class was the trivial "Drum" came back with no tags at all (38 of 140 MUSDB clips).
* **Presence** (``detect_instruments``, used by "separate the rest"): a family is present when
  its mean window score clears a per-family, per-model threshold fitted on a synthetic
  presence set (``PRESENCE``), instead of one global 0.1 that suits "Guitar" (scores 0.3-0.9)
  but not "Brass" or "Mallets" (0.02-0.2 even when clearly audible).

Windows: the stem is cut into 10 s windows; a window is *active* when it is within
``ACTIVE_STEM_DB`` of the stem's loudest window and within ``ACTIVE_MIX_DB`` of the mixture in
the same window (bleed in a verse where the stem is silent is not analysed). Up to
``max_windows`` active windows spread across the song are scored, each level-normalised first.

AudioSet names are matched by string: PANNs (``class_labels_indices.csv``) and AST
(``model.config.id2label``) share the 527 names. A zero-shot tagger (``zero_shot = True``)
instead reports one cosine similarity per label id (``LABELS`` with prompts); those become
probabilities by a softmax over the stem's vocabulary.
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
CHILD_RATIO = 0.5  # a specific label ("Electric guitar") replaces its family name when >= 50% of the family score
FALLBACK_NAME = "Other"
CLAP_TEMPERATURE = 0.05  # softmax temperature over cosine similarities (zero-shot taggers)


@dataclass(frozen=True)
class Label:
    id: str
    display: str
    family: str  # canonical family: vocals, drums, bass, piano, keys, organ, guitar, synth, strings, brass, woodwinds, ...
    audioset: tuple[str, ...] = ()  # AudioSet classes; the label's window score is their max
    prompts: tuple[str, ...] = ()  # zero-shot text prompts
    generic: bool = False  # the family-level label (shown when no specific label is confident)


def _l(id: str, display: str, family: str, audioset: Sequence[str] = (), prompts: Sequence[str] = (), generic: bool = False) -> Label:
    return Label(id, display, family, tuple(audioset), tuple(prompts), generic)


# Voice. "singing" pools every class AudioSet uses for sung voice: separated vocals score as
# "A capella" / "Vocal music" / "Mantra" / "Chant" at least as often as "Singing".
SINGING_EVIDENCE = ("Singing", "Male singing", "Female singing", "Child singing", "Synthetic singing", "Choir", "Rapping",
                    "A capella", "Vocal music", "Chant", "Mantra", "Humming", "Yodeling")
LABELS: tuple[Label, ...] = (
    _l("singing", "Singing", "vocals", SINGING_EVIDENCE, ("singing", "vocals", "a singer singing"), generic=True),
    _l("male_singing", "Male singing", "vocals", ("Male singing",), ("a man singing", "male vocals", "male singer singing")),
    _l("female_singing", "Female singing", "vocals", ("Female singing",), ("a woman singing", "female vocals", "female singer singing")),
    _l("child_singing", "Child singing", "vocals", ("Child singing",), ("a child singing",)),
    _l("rapping", "Rapping", "vocals", ("Rapping",), ("rapping", "a rapper rapping", "hip hop rap vocals")),
    _l("choir", "Choir", "vocals", ("Choir",), ("a choir singing", "choir vocals", "a group of people singing")),
    _l("synth_vocals", "Synth vocals", "vocals", ("Synthetic singing",), ("vocoder vocals", "auto-tuned robotic vocals")),
    _l("speech", "Spoken word", "vocals", ("Speech", "Narration, monologue"), ("a person talking", "spoken word", "a man speaking")),
    _l("beatboxing", "Beatboxing", "vocals", ("Beatboxing",), ("beatboxing",)),
    # Drums.
    _l("drum_kit", "Drum kit", "drums", ("Drum kit", "Drum", "Snare drum", "Bass drum", "Drum roll", "Rimshot", "Hi-hat", "Cymbal"),
       ("drum kit", "acoustic drums", "a drummer playing a drum kit"), generic=True),
    _l("drum_machine", "Drum machine", "drums", ("Drum machine",), ("drum machine", "electronic drum beat", "808 drum machine beat")),
    _l("hihat", "Hi-hat", "drums", ("Hi-hat",)),
    _l("snare", "Snare", "drums", ("Snare drum", "Rimshot")),
    _l("kick", "Kick drum", "drums", ("Bass drum",)),
    _l("cymbals", "Cymbals", "drums", ("Cymbal",)),
    _l("percussion", "Percussion", "percussion", ("Percussion", "Tambourine", "Maraca", "Rattle (instrument)", "Wood block", "Tabla", "Clapping"),
       ("hand percussion", "congas and bongos", "shaker and tambourine"), generic=True),
    _l("tambourine", "Tambourine", "percussion", ("Tambourine",)),
    _l("shaker", "Shaker", "percussion", ("Maraca", "Rattle (instrument)")),
    _l("claps", "Claps", "percussion", ("Clapping",)),
    _l("tabla", "Tabla", "percussion", ("Tabla",)),
    # Bass. A synth bass reads as "Synthesizer" to AudioSet.
    _l("bass_guitar", "Bass guitar", "bass", ("Bass guitar",), ("electric bass guitar", "bass guitar", "a bass guitar playing a bassline"), generic=True),
    _l("synth_bass", "Synth bass", "bass", ("Synthesizer",), ("synth bass", "synthesizer bass line", "electronic bass")),
    _l("double_bass", "Double bass", "bass", ("Double bass",), ("upright double bass", "acoustic double bass", "jazz upright bass")),
    # Melodic instruments.
    _l("piano", "Piano", "piano", ("Piano",), ("piano", "acoustic piano", "a grand piano playing"), generic=True),
    _l("electric_piano", "Electric piano", "keys", ("Electric piano",), ("electric piano", "rhodes electric piano", "wurlitzer electric piano"), generic=True),
    _l("harpsichord", "Harpsichord", "keys", ("Harpsichord",), ("harpsichord", "clavinet")),
    _l("organ", "Organ", "organ", ("Organ", "Electronic organ", "Hammond organ"), ("organ", "hammond organ", "church organ"), generic=True),
    _l("guitar", "Guitar", "guitar", ("Guitar", "Electric guitar", "Acoustic guitar", "Steel guitar, slide guitar", "Strum", "Tapping (guitar technique)"),
       ("guitar", "a guitar playing"), generic=True),
    _l("electric_guitar", "Electric guitar", "guitar", ("Electric guitar",), ("electric guitar", "distorted electric guitar", "clean electric guitar")),
    _l("acoustic_guitar", "Acoustic guitar", "guitar", ("Acoustic guitar",), ("acoustic guitar", "strummed acoustic guitar", "nylon string guitar")),
    _l("slide_guitar", "Slide guitar", "guitar", ("Steel guitar, slide guitar",)),
    _l("synth", "Synth", "synth", ("Synthesizer",), ("synthesizer", "synth lead", "synth pad", "electronic synthesizer"), generic=True),
    _l("strings", "Strings", "strings", ("Bowed string instrument", "String section", "Violin, fiddle", "Cello", "Pizzicato", "Orchestra"),
       ("string section", "orchestral strings", "violins and cellos"), generic=True),
    _l("violin", "Violin", "strings", ("Violin, fiddle",), ("violin", "fiddle")),
    _l("cello", "Cello", "strings", ("Cello",), ("cello",)),
    _l("brass", "Brass", "brass", ("Brass instrument", "French horn", "Trumpet", "Trombone"), ("brass section", "brass instruments", "horns"), generic=True),
    _l("trumpet", "Trumpet", "brass", ("Trumpet",), ("trumpet",)),
    _l("trombone", "Trombone", "brass", ("Trombone",), ("trombone",)),
    _l("horn", "Horn", "brass", ("French horn",)),
    _l("woodwinds", "Woodwinds", "woodwinds", ("Wind instrument, woodwind instrument", "Flute", "Saxophone", "Clarinet"),
       ("woodwind instruments", "woodwinds"), generic=True),
    _l("saxophone", "Saxophone", "woodwinds", ("Saxophone",), ("saxophone", "sax solo")),
    _l("flute", "Flute", "woodwinds", ("Flute",), ("flute", "pan flute", "recorder")),
    _l("clarinet", "Clarinet", "woodwinds", ("Clarinet",), ("clarinet", "oboe", "bassoon")),
    _l("mallets", "Mallets", "mallets", ("Mallet percussion", "Marimba, xylophone", "Glockenspiel", "Vibraphone"),
       ("marimba", "vibraphone", "xylophone", "glockenspiel"), generic=True),
    _l("marimba", "Marimba", "mallets", ("Marimba, xylophone",)),
    _l("glockenspiel", "Glockenspiel", "mallets", ("Glockenspiel",)),
    _l("vibraphone", "Vibraphone", "mallets", ("Vibraphone",)),
    _l("harp", "Harp", "harp", ("Harp",), ("harp",), generic=True),
    _l("banjo", "Banjo", "banjo", ("Banjo",), ("banjo",), generic=True),
    _l("mandolin", "Mandolin", "mandolin", ("Mandolin",), ("mandolin",), generic=True),
    _l("ukulele", "Ukulele", "ukulele", ("Ukulele",), ("ukulele",), generic=True),
    _l("sitar", "Sitar", "sitar", ("Sitar",), ("sitar",), generic=True),
    _l("accordion", "Accordion", "accordion", ("Accordion",), ("accordion",), generic=True),
    _l("harmonica", "Harmonica", "harmonica", ("Harmonica",), ("harmonica",), generic=True),
    _l("bells", "Bells", "bells", ("Bell", "Church bell", "Tubular bells", "Chime", "Wind chime", "Jingle bell"), ("bells", "chimes"), generic=True),
    _l("steelpan", "Steel drums", "steelpan", ("Steelpan",), generic=True),
)
LABEL: dict[str, Label] = {lab.id: lab for lab in LABELS}
NOISE_PROMPTS = ("noise", "silence", "static")  # zero-shot sink label, so a junk window is not forced onto an instrument

def clap_prompts() -> dict[str, tuple[str, ...]]:
    """Label id -> text prompts for a zero-shot tagger (labels without prompts are left out),
    plus the "noise" sink."""
    return {**{lab.id: lab.prompts for lab in LABELS if lab.prompts}, "noise": NOISE_PROMPTS}


VOCAL_IDS = ("singing", "male_singing", "female_singing", "child_singing", "rapping", "choir", "synth_vocals", "speech", "beatboxing")
DRUM_IDS = ("drum_kit", "drum_machine", "hihat", "snare", "kick", "cymbals", "percussion", "tambourine", "shaker", "claps", "tabla")
BASS_IDS = ("bass_guitar", "synth_bass", "double_bass")
INSTRUMENT_IDS = tuple(lab.id for lab in LABELS if lab.family not in {"vocals", "drums", "percussion", "bass"})
# Canonical stem -> labels it may be described with. Anything else -> INSTRUMENT_IDS.
STEM_VOCAB: dict[str, tuple[str, ...]] = {
    "vocals": VOCAL_IDS,
    "drums": DRUM_IDS,
    "bass": BASS_IDS,
    "instrumental": DRUM_IDS[:2] + ("percussion",) + BASS_IDS + INSTRUMENT_IDS,
    "residual": VOCAL_IDS[:-1] + DRUM_IDS[:2] + ("percussion",) + BASS_IDS + INSTRUMENT_IDS,  # whatever was not asked for, voice included
}
# Families "separate the rest" can be asked to find (and the order ties are broken in).
INSTRUMENT_FAMILIES: tuple[str, ...] = tuple(dict.fromkeys(LABEL[i].family for i in INSTRUMENT_IDS))
# Old display names (and AudioSet names) that callers may still send, mapped to families.
FAMILY_ALIASES: dict[str, str] = {
    "keyboard": "keys", "keyboards": "keys", "electric piano": "keys", "rhodes": "keys", "synthesizer": "synth", "synths": "synth",
    "string": "strings", "violin": "strings", "cello": "strings", "horns": "brass", "trumpet": "brass", "saxophone": "woodwinds",
    "sax": "woodwinds", "flute": "woodwinds", "woodwind": "woodwinds", "guitars": "guitar", "electric guitar": "guitar",
    "acoustic guitar": "guitar", "marimba": "mallets", "vibraphone": "mallets", "glockenspiel": "mallets",
}

# Presence thresholds on a family's mean window score, per tagger. Fitted 2026-11-14 on 240
# synthetic "other" stems (1-4 BabySlakh instruments + drums/bass, 30 s, through the chain
# splitter): per family, the lowest threshold with >= 60% precision there (recall 45-85%),
# rounded. Nothing reached 60% for "synth" (AudioSet "Synthesizer" barely separates Slakh's
# synth patches from the rest), so it gets a conservative 0.3. On the 20 real BabySlakh songs
# this is ~0.85 precision; recall is lower there because dense arrangements hide quiet parts.
# Recall is preferred over precision: "separate the rest" folds an instrument that comes back
# near-silent into "Rest", so a false positive costs less than a missed instrument.
PRESENCE: dict[str, dict[str, float]] = {
    "panns": {"piano": 0.18, "keys": 0.19, "organ": 0.15, "guitar": 0.15, "synth": 0.3, "strings": 0.07, "brass": 0.03, "woodwinds": 0.02, "mallets": 0.01},
    "ast": {"piano": 0.08, "keys": 0.05, "organ": 0.045, "guitar": 0.045, "synth": 0.3, "strings": 0.04, "brass": 0.02, "woodwinds": 0.02, "mallets": 0.01},
    "clap": {"piano": 0.03, "keys": 0.28, "organ": 0.3, "guitar": 0.22, "synth": 0.3, "strings": 0.05, "brass": 0.06, "woodwinds": 0.17, "mallets": 0.02},
}
DEFAULT_PRESENCE: dict[str, float] = {"panns": 0.15, "ast": 0.1, "clap": 0.2}  # harp, banjo, accordion, ... (not calibrated)
NAME_RATIO = 0.5  # nothing present: the top family still names the stem when it reaches half its threshold

# Voice rules (AudioSet probabilities, mean over active windows; MUSDB vocals separated by the chain).
GENDER_MIN = 0.08  # the winning gendered class must reach this ...
GENDER_MARGIN = 0.05  # ... and beat the other by this much (PANNs: female 97% / male ~80% precise at 0.05)
RAP_MIN = 0.08  # "Rapping" above this and at least half the gendered score
SPEECH_MIN, SPEECH_RATIO = 0.6, 6.0  # "Spoken word" only when speech dominates sung evidence 6:1


class Tagger(Protocol):
    """What labeling needs from a tagger (``classifier._AudioSetTagger``, ``classifier.ClapTagger``)."""

    def class_names(self) -> list[str]: ...

    def tag(self, clips: list[np.ndarray], sample_rate: int) -> np.ndarray:
        """``(len(clips), n_classes)`` scores for mono clips at ``sample_rate``."""
        ...


@dataclass
class Instrument:
    """One instrument family found in a stem (``detect_instruments``)."""

    family: str  # canonical family id (piano, keys, organ, guitar, synth, strings, brass, woodwinds, ...)
    name: str  # display name, the most specific label that is confident ("Electric guitar")
    score: float  # mean window score of the family
    present: bool  # score >= the family's presence threshold
    coverage: float = 0.0  # fraction of analysed windows where the family clears its threshold

    def as_dict(self) -> dict[str, float | str | bool]:
        return {"family": self.family, "name": self.name, "score": round(self.score, 3), "present": self.present, "coverage": round(self.coverage, 3)}


@dataclass
class StemLabel:
    tags: list[dict[str, float | str]]
    name: str | None  # descriptive name (always computed; the engine decides whether to use it)
    candidates: list[tuple[str, float]] = field(default_factory=list)  # every in-vocabulary display name with its score
    energy_db: float = 0.0  # stem energy relative to the mixture
    windows: int = 0
    instruments: list[Instrument] = field(default_factory=list)  # families in the stem's vocabulary, by score


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


def active_windows(stem: np.ndarray, mix_energy: np.ndarray | None, sample_rate: int, max_windows: int, window_seconds: float = WINDOW_SECONDS) -> list[np.ndarray]:
    """Up to ``max_windows`` active mono windows of ``stem``, spread evenly across the song.
    ``mix_energy`` is ``window_energies`` of the mixture's mono with the same window size, or
    None to judge activity by the stem alone."""
    size = int(window_seconds * sample_rate)
    mono = _mono(stem)
    starts, energy = window_energies(mono, size)
    peak = float(energy.max()) if energy.size else 0.0
    if peak <= 0:
        return []
    active = [i for i in range(len(starts)) if _db(energy[i], peak) >= ACTIVE_STEM_DB and (mix_energy is None or _db(energy[i], mix_energy[i]) >= ACTIVE_MIX_DB)]
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


def tagger_kind(tagger: object) -> str:
    """``panns`` | ``ast`` | ``clap`` (calibration key); unknown AudioSet taggers count as panns."""
    if getattr(tagger, "zero_shot", False):
        return "clap"
    name = str(getattr(tagger, "name", "panns"))
    return name if name in PRESENCE else "panns"


def vocabulary(canonical: str) -> tuple[str, ...]:
    return STEM_VOCAB.get(canonical, INSTRUMENT_IDS)


def label_scores(probs: np.ndarray, class_names: Sequence[str], vocab: Sequence[str], *, zero_shot: bool = False,
                 temperature: float = CLAP_TEMPERATURE) -> np.ndarray:
    """``(windows, len(vocab))`` label scores in [0, 1] from a tagger's ``(windows, classes)`` output.

    AudioSet taggers: a label's score is the max probability over its AudioSet classes (labels
    whose classes the tagger does not have score 0). Zero-shot taggers report one cosine
    similarity per label id; a softmax over the vocabulary plus a "noise" sink (when the tagger
    has one) turns them into probabilities, so labels compete only with the stem's own vocabulary.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return np.zeros((0, len(vocab)))
    index = {n: i for i, n in enumerate(class_names)}
    if zero_shot:
        cols = [index[v] for v in vocab if v in index]
        present = [v for v in vocab if v in index]
        sink = [index["noise"]] if "noise" in index else []
        logits = probs[:, cols + sink] / temperature
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        out = np.zeros((probs.shape[0], len(vocab)))
        for j, v in enumerate(present):
            out[:, list(vocab).index(v)] = p[:, j]
        return out
    out = np.zeros((probs.shape[0], len(vocab)))
    for j, v in enumerate(vocab):
        cols = [index[c] for c in LABEL[v].audioset if c in index]
        if cols:
            out[:, j] = probs[:, cols].max(axis=1)
    return out


def family_scores(scores: np.ndarray, vocab: Sequence[str], *, zero_shot: bool = False) -> dict[str, np.ndarray]:
    """Per-window family scores: the max over the family's labels (AudioSet; the generic label
    already pools its children's classes) or the sum of their probabilities (zero-shot)."""
    out: dict[str, np.ndarray] = {}
    for j, v in enumerate(vocab):
        fam = LABEL[v].family
        col = scores[:, j]
        if fam not in out:
            out[fam] = col.copy()
        else:
            out[fam] = out[fam] + col if zero_shot else np.maximum(out[fam], col)
    if zero_shot:
        out = {k: np.minimum(v, 1.0) for k, v in out.items()}
    return out


def _display(family: str, mean: dict[str, float], vocab: Sequence[str]) -> str:
    """The family's most specific confident label: a child scoring >= CHILD_RATIO of the family."""
    members = [v for v in vocab if LABEL[v].family == family]
    generic = next((v for v in members if LABEL[v].generic), members[0])
    fam_score = max(mean[v] for v in members)
    children = sorted((v for v in members if not LABEL[v].generic), key=lambda v: -mean[v])
    if children and fam_score > 0 and mean[children[0]] >= CHILD_RATIO * fam_score and mean[children[0]] >= mean[generic] * CHILD_RATIO:
        return LABEL[children[0]].display
    return LABEL[generic].display


def presence_threshold(kind: str, family: str, min_score: float | None = None) -> float:
    """The calibrated per-family threshold, raised to ``min_score`` (``AUDIO_STEM_LABEL_MIN_SCORE``,
    0 = calibrated values only) when that is higher."""
    calibrated = PRESENCE.get(kind, PRESENCE["panns"]).get(family, DEFAULT_PRESENCE.get(kind, 0.1))
    return max(calibrated, float(min_score or 0.0))


def instruments_from_scores(scores: np.ndarray, vocab: Sequence[str], kind: str, min_score: float | None = None) -> list[Instrument]:
    """Families of ``vocab`` from ``(windows, len(vocab))`` label scores, sorted by score."""
    if scores.shape[0] == 0:
        return []
    zero_shot = kind == "clap"
    fams = family_scores(scores, vocab, zero_shot=zero_shot)
    mean = {v: float(scores[:, j].mean()) for j, v in enumerate(vocab)}
    out = []
    for fam, col in fams.items():
        thr = presence_threshold(kind, fam, min_score)
        s = float(col.mean())
        out.append(Instrument(fam, _display(fam, mean, vocab), s, s >= thr, float((col >= thr).mean())))
    order = {f: i for i, f in enumerate(INSTRUMENT_FAMILIES)}
    return sorted(out, key=lambda x: (-x.score, order.get(x.family, 99)))


def _vocal_tags(scores: np.ndarray, vocab: Sequence[str], kind: str) -> list[tuple[str, float]]:
    """Voice tags: gender by margin, Rapping, Choir, Synth vocals; "Singing" as the fallback;
    "Spoken word" only when speech clearly dominates."""
    m = {v: float(scores[:, j].mean()) for j, v in enumerate(vocab)}
    sung = m.get("singing", 0.0)
    male, female = m.get("male_singing", 0.0), m.get("female_singing", 0.0)
    tags: list[tuple[str, float]] = []
    if kind == "clap":  # probabilities over the voice vocabulary: take the top voice labels that matter
        ranked = sorted(((v, s) for v, s in m.items() if v != "singing"), key=lambda kv: -kv[1])
        tags = [(LABEL[v].display, s) for v, s in ranked if s >= 0.2][:2]
        return tags or [("Singing", max(sung, ranked[0][1] if ranked else 0.0))]
    if m.get("speech", 0.0) >= SPEECH_MIN and m["speech"] >= SPEECH_RATIO * sung:
        tags.append(("Spoken word", m["speech"]))
    rap = m.get("rapping", 0.0)
    if rap >= RAP_MIN and rap >= 0.5 * max(male, female):
        tags.append(("Rapping", rap))
    hi, lo = (("Male singing", male), ("Female singing", female)) if male >= female else (("Female singing", female), ("Male singing", male))
    if hi[1] >= GENDER_MIN and hi[1] - lo[1] >= GENDER_MARGIN:
        tags.append(hi)
    elif hi[1] >= GENDER_MIN and lo[1] >= GENDER_MIN and scores.shape[0] >= 2:
        # Both gendered classes are strong: a duet if each clearly wins some windows.
        jm, jf = list(vocab).index("male_singing"), list(vocab).index("female_singing")
        wins_m = float(np.mean(scores[:, jm] - scores[:, jf] >= GENDER_MARGIN))
        wins_f = float(np.mean(scores[:, jf] - scores[:, jm] >= GENDER_MARGIN))
        if min(wins_m, wins_f) >= 0.25:
            tags += [hi, lo]
    for v in ("choir", "synth_vocals", "child_singing"):
        if m.get(v, 0.0) >= 0.15 and m[v] >= 0.5 * max(male, female, 1e-9):
            tags.append((LABEL[v].display, m[v]))
    if m.get("beatboxing", 0.0) >= 0.2:
        tags.append(("Beatboxing", m["beatboxing"]))
    if not any(t in {"Male singing", "Female singing", "Rapping", "Choir", "Synth vocals", "Child singing", "Spoken word"} for t, _ in tags):
        tags.insert(0, ("Singing", sung))
    return sorted(tags, key=lambda kv: -kv[1])


def _family_first_tags(scores: np.ndarray, vocab: Sequence[str], primaries: Sequence[str], details: Sequence[str], min_score: float, prefer: str) -> list[tuple[str, float]]:
    """Drums/bass: always the best primary label (``prefer`` unless another primary beats it),
    then confident detail labels."""
    m = {v: float(scores[:, j].mean()) for j, v in enumerate(vocab)}
    best = max(primaries, key=lambda v: (m.get(v, 0.0) - (0.0 if v == prefer else 1e-9)))
    tags = [(LABEL[best].display, m.get(best, 0.0))]
    for v in sorted(details, key=lambda v: -m.get(v, 0.0)):
        if m.get(v, 0.0) >= min_score and LABEL[v].display != tags[0][0]:
            tags.append((LABEL[v].display, m[v]))
    return tags


# Display names that say nothing new on a stem of that name ("Guitar" on the guitar stem).
TRIVIAL: dict[str, frozenset[str]] = {"guitar": frozenset({"Guitar"}), "piano": frozenset({"Piano"})}


def _instrument_tags(instruments: list[Instrument], canonical: str, kind: str, min_score: float | None) -> list[tuple[str, float]]:
    """Present families; when none is, the top family if it reaches ``NAME_RATIO`` of its threshold."""
    trivial = TRIVIAL.get(canonical, frozenset())
    usable = [i for i in instruments if i.name not in trivial]
    tags = [(i.name, i.score) for i in usable if i.present]
    if not tags and usable and usable[0].score >= NAME_RATIO * presence_threshold(kind, usable[0].family, min_score):
        tags = [(usable[0].name, usable[0].score)]
    return tags


def describe(cands: list[tuple[str, float]]) -> str | None:
    """Descriptive name from (already filtered) candidates: top, or "A & B"; None if nothing."""
    if not cands:
        return None
    (first, s1), rest = cands[0], cands[1:]
    if rest and rest[0][1] >= PAIR_RATIO * s1:
        return f"{first} & {rest[0][0]}"
    return first


def _plan(stem: np.ndarray, mix_windows: np.ndarray | None, mix_energy: float | None, sample_rate: int, max_windows: int) -> tuple[float, list[np.ndarray]]:
    """Energy of the stem relative to the mixture (0 dB without one; -120 if silent) and its active windows."""
    energy = _energy(stem)
    energy_db = _db(energy, mix_energy) if mix_energy is not None else (0.0 if energy > 0 else -120.0)
    clips = [] if energy_db < SILENT_DB else active_windows(stem, mix_windows, sample_rate, max_windows)
    return energy_db, clips


def label_stems(
    tagger: Tagger,
    stems: Sequence[tuple[str, np.ndarray]],
    mixture: np.ndarray,
    sample_rate: int,
    *,
    min_score: float | None = None,
    max_windows: int = 12,
) -> list[StemLabel]:
    """Label each ``(canonical, samples)`` stem. All windows go to the tagger in one call.

    ``min_score`` (``AUDIO_STEM_LABEL_MIN_SCORE``; 0 or None = calibrated thresholds only) raises
    the per-family presence thresholds and the drum detail floor, and filters ``candidates``."""
    kind = tagger_kind(tagger)
    zero_shot = kind == "clap"
    names = tagger.class_names()
    mix_energy = _energy(mixture)
    _, mix_windows = window_energies(_mono(mixture), int(WINDOW_SECONDS * sample_rate))
    plans = [(canonical, *_plan(samples, mix_windows, mix_energy, sample_rate, max_windows)) for canonical, samples in stems]
    flat = [c for _, _, clips in plans for c in clips]
    probs = tagger.tag(flat, sample_rate) if flat else np.zeros((0, len(names)))
    out: list[StemLabel] = []
    offset = 0
    floor = float(min_score or 0.0)
    for canonical, energy_db, clips in plans:
        if not clips:
            out.append(StemLabel([], None, [], round(energy_db, 2), 0))
            continue
        window_probs = np.asarray(probs[offset : offset + len(clips)])
        offset += len(clips)
        vocab = vocabulary(canonical)
        scores = label_scores(window_probs, names, vocab, zero_shot=zero_shot)
        instruments = instruments_from_scores(scores, vocab, kind, min_score)
        mean = {v: float(scores[:, j].mean()) for j, v in enumerate(vocab)}
        cands = sorted(((LABEL[v].display, s) for v, s in mean.items() if s >= floor and s > 0), key=lambda kv: (-kv[1], kv[0]))
        if canonical == "vocals":
            tags = _vocal_tags(scores, vocab, kind)
            name = tags[0][0] if tags else None
        elif canonical == "drums":
            tags = _family_first_tags(scores, vocab, ("drum_kit", "drum_machine"), ("percussion", "tambourine", "shaker", "claps", "tabla", "hihat", "snare", "kick", "cymbals"),
                                      max(floor, 0.2 if zero_shot else 0.15), "drum_kit")
            name = tags[0][0]
        elif canonical == "bass":
            tags = _family_first_tags(scores, vocab, BASS_IDS, (), 1.0, "bass_guitar")
            name = tags[0][0]
        else:
            tags = _instrument_tags(instruments, canonical, kind, min_score)
            name = describe(tags)
        dedup: list[tuple[str, float]] = []
        for t, s in tags:
            if t not in {d for d, _ in dedup}:
                dedup.append((t, s))
        out.append(StemLabel([{"label": t, "score": round(s, 3)} for t, s in dedup[:MAX_TAGS]], name, cands, round(energy_db, 2), len(clips), instruments))
    return out


def detect_instruments(
    tagger: Tagger,
    samples: np.ndarray,
    sample_rate: int,
    *,
    parent: str = "other",
    mixture: np.ndarray | None = None,
    min_score: float | None = None,
    max_windows: int = 12,
) -> list[Instrument]:
    """Instrument families in one stem (for "separate the rest"), best first, each with
    ``present``. ``parent`` picks the vocabulary (``other`` -> melodic instruments); ``mixture``
    (optional) excludes windows where the stem is only bleed. A near-silent stem gives []."""
    mix_energy = _energy(mixture) if mixture is not None else None
    mix_windows = window_energies(_mono(mixture), int(WINDOW_SECONDS * sample_rate))[1] if mixture is not None else None
    _, clips = _plan(samples, mix_windows, mix_energy, sample_rate, max_windows)
    if not clips:
        return []
    kind = tagger_kind(tagger)
    vocab = vocabulary(parent)
    scores = label_scores(tagger.tag(clips, sample_rate), tagger.class_names(), vocab, zero_shot=kind == "clap")
    return instruments_from_scores(scores, vocab, kind, min_score)


def present_families(instruments: Sequence[Instrument], limit: int | None = None) -> list[str]:
    """Family ids marked present, best first, at most ``limit``."""
    fams = [i.family for i in instruments if i.present]
    return fams[:limit] if limit is not None else fams
