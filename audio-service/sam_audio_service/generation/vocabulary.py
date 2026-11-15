"""Open instrument vocabulary: a free-text ask -> (instrument, family, role, register, articulation,
density, renderer).

"add a funky bassline" -> bass guitar, bassline, funk style; "add an 80s synth arpeggio" -> built-in
synth 'arp80s' preset, ostinato; "a trumpet line in the chorus" -> trumpet, counter_melody;
"add a choir" -> GM choir aahs, pad. Anything not in the table falls back to the General MIDI
program-name table, and descriptive texture asks ("ambient shimmer", "lo-fi guitar loop") are
flagged ``audio_generative_suggested`` so the caller can route them to the audio-generative engine.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

ROLES = ("pad", "counter_melody", "ostinato", "bassline", "riff", "stabs", "chords", "lead")
ROLE_ALIASES = {
    "pad": "pad", "pads": "pad", "bed": "pad", "sustain": "pad", "drone": "pad", "swell": "pad", "under": "pad",
    "counter": "counter_melody", "countermelody": "counter_melody", "counter-melody": "counter_melody", "line": "counter_melody",
    "melody": "lead", "lead": "lead", "solo": "lead", "hook": "lead", "topline": "lead",
    "arpeggio": "ostinato", "arp": "ostinato", "arpeggiated": "ostinato", "ostinato": "ostinato", "pattern": "ostinato", "pulse": "ostinato", "plucks": "ostinato",
    "bassline": "bassline", "bass line": "bassline", "bass": "bassline",
    "riff": "riff", "groove": "riff", "lick": "riff", "licks": "riff",
    "stabs": "stabs", "stab": "stabs", "hits": "stabs", "punches": "stabs", "horn section": "stabs",
    "chords": "chords", "comping": "chords", "comp": "chords", "rhythm": "chords", "strumming": "chords", "strum": "chords", "part": "chords",
}
STYLE_WORDS = {
    "funky": "funk", "funk": "funk", "disco": "funk", "slap": "funk",
    "80s": "80s", "eighties": "80s", "retro": "80s", "synthwave": "80s", "outrun": "80s",
    "ambient": "ambient", "dreamy": "ambient", "ethereal": "ambient", "lush": "ambient", "cinematic": "cinematic", "epic": "cinematic",
    "jazzy": "jazz", "jazz": "jazz", "walking": "jazz", "lofi": "lofi", "lo-fi": "lofi", "chill": "lofi",
    "rock": "rock", "heavy": "rock", "distorted": "rock", "gentle": "gentle", "soft": "gentle", "simple": "gentle", "sparse": "gentle", "busy": "busy",
}
TEXTURE_WORDS = ("shimmer", "texture", "atmosphere", "atmospheric", "soundscape", "loop", "sample", "noise", "vinyl", "field recording", "glitch", "riser", "whoosh", "foley")


@dataclass
class Instrument:
    key: str
    display: str
    family: str  # strings | keys | brass | winds | synth | bass | guitar | voice | mallets | plucked
    aliases: tuple[str, ...]
    renderer: str  # vsco:<preset> | gm:<program> | synth:<preset>
    fallback: str | None  # used when the primary renderer is unavailable
    low: int
    high: int
    default_role: str
    articulations: dict[str, str] = field(default_factory=dict)  # role -> articulation hint
    monophonic: bool = False


INSTRUMENTS: list[Instrument] = [
    Instrument("strings", "Strings", "strings", ("strings", "string section", "string pad", "orchestra", "orchestral"), "vsco:strings", "gm:48", 36, 91, "pad", {"ostinato": "spiccato", "stabs": "spiccato"}),
    Instrument("violins", "Violins", "strings", ("violins", "violin section", "fiddles"), "vsco:violins", "gm:48", 55, 96, "counter_melody", {"ostinato": "spiccato", "stabs": "spiccato", "riff": "spiccato"}),
    Instrument("violin", "Violin", "strings", ("violin", "solo violin", "fiddle"), "vsco:solo_violin", "gm:40", 55, 98, "counter_melody", {"ostinato": "pizzicato"}, monophonic=True),
    Instrument("viola", "Violas", "strings", ("viola", "violas"), "vsco:violas", "gm:41", 48, 84, "pad"),
    Instrument("cello", "Cellos", "strings", ("cello", "cellos", "celli"), "vsco:cellos", "gm:42", 36, 76, "counter_melody", {"ostinato": "spiccato", "bassline": "spiccato"}),
    Instrument("contrabass", "Double bass", "bass", ("double bass", "upright bass", "contrabass", "acoustic bass"), "vsco:contrabass_pizz", "gm:32", 28, 55, "bassline", monophonic=True),
    Instrument("pizzicato", "Pizzicato strings", "strings", ("pizzicato", "pizz"), "vsco:violins_pizz", "gm:45", 43, 91, "ostinato"),
    Instrument("piano", "Piano", "keys", ("piano", "keys", "grand piano", "upright piano", "acoustic piano"), "vsco:piano", "gm:0", 33, 96, "chords"),
    Instrument("epiano", "Electric piano", "keys", ("electric piano", "rhodes", "wurlitzer", "e-piano", "ep"), "gm:4", None, 36, 91, "chords"),
    Instrument("organ", "Organ", "keys", ("organ", "hammond", "b3"), "gm:16", None, 36, 91, "pad"),
    Instrument("harpsichord", "Harpsichord", "keys", ("harpsichord",), "gm:6", None, 41, 89, "ostinato"),
    Instrument("celesta", "Celesta", "mallets", ("celesta", "music box", "glockenspiel", "glock", "bells"), "gm:8", None, 60, 100, "ostinato"),
    Instrument("vibraphone", "Vibraphone", "mallets", ("vibraphone", "vibes", "marimba", "xylophone"), "gm:11", None, 53, 89, "chords"),
    Instrument("harp", "Harp", "plucked", ("harp",), "gm:46", None, 36, 96, "ostinato"),
    Instrument("acoustic_guitar", "Acoustic guitar", "guitar", ("acoustic guitar", "nylon guitar", "classical guitar", "folk guitar"), "gm:25", None, 40, 81, "chords"),
    Instrument("guitar", "Guitar", "guitar", ("guitar", "electric guitar", "clean guitar", "guitar riff", "funk guitar"), "gm:27", None, 40, 84, "riff"),
    Instrument("rock_guitar", "Distorted guitar", "guitar", ("distorted guitar", "overdriven guitar", "rock guitar", "power chords"), "gm:30", None, 40, 76, "riff"),
    Instrument("bass", "Bass", "bass", ("bass", "bass guitar", "electric bass", "bassline", "bass line", "funky bassline", "slap bass"), "gm:33", None, 28, 55, "bassline", monophonic=True),
    Instrument("synth_bass", "Synth bass", "synth", ("synth bass", "808", "sub bass", "moog bass"), "synth:bass", None, 24, 55, "bassline", monophonic=True),
    Instrument("synth_pad", "Synth pad", "synth", ("synth pad", "warm pad", "pad", "analog pad"), "synth:pad", None, 43, 88, "pad"),
    Instrument("synth_arp", "Synth arpeggio", "synth", ("synth arpeggio", "synth arp", "arp synth", "80s synth", "arpeggiator", "synth"), "synth:arp80s", None, 48, 91, "ostinato"),
    Instrument("synth_lead", "Synth lead", "synth", ("synth lead", "lead synth", "saw lead", "chiptune"), "synth:lead", None, 55, 96, "lead", monophonic=True),
    Instrument("synth_pluck", "Synth pluck", "synth", ("pluck", "synth pluck", "plucks"), "synth:pluck", None, 55, 91, "ostinato"),
    Instrument("trumpet", "Trumpet", "brass", ("trumpet", "trumpets", "cornet", "flugelhorn"), "gm:56", None, 54, 84, "counter_melody", monophonic=True),
    Instrument("trombone", "Trombone", "brass", ("trombone", "trombones"), "gm:57", None, 40, 72, "counter_melody", monophonic=True),
    Instrument("horn", "French horn", "brass", ("french horn", "horn", "horns"), "vsco:horn", "gm:60", 41, 77, "pad"),
    Instrument("brass", "Brass section", "brass", ("brass", "brass section", "horn section", "horn stabs", "brass stabs"), "gm:61", None, 46, 82, "stabs"),
    Instrument("tuba", "Tuba", "brass", ("tuba",), "gm:58", None, 28, 58, "bassline", monophonic=True),
    Instrument("saxophone", "Saxophone", "winds", ("sax", "saxophone", "alto sax", "tenor sax"), "gm:65", None, 49, 81, "counter_melody", monophonic=True),
    Instrument("flute", "Flute", "winds", ("flute", "flutes", "piccolo", "pan flute", "recorder"), "vsco:flute", "gm:73", 60, 96, "counter_melody", monophonic=True),
    Instrument("clarinet", "Clarinet", "winds", ("clarinet",), "gm:71", None, 50, 89, "counter_melody", monophonic=True),
    Instrument("oboe", "Oboe", "winds", ("oboe", "english horn", "bassoon"), "gm:68", None, 58, 91, "counter_melody", monophonic=True),
    Instrument("choir", "Choir", "voice", ("choir", "choirs", "voices", "aahs", "oohs", "backing vocals", "vocal pad", "chorus voices"), "gm:52", None, 48, 79, "pad"),
]
BY_KEY = {i.key: i for i in INSTRUMENTS}

# General MIDI level 1 program names (for "add a banjo", "accordion" ... not in the table).
GM_NAMES = [
    "acoustic grand piano", "bright acoustic piano", "electric grand piano", "honky-tonk piano", "electric piano 1", "electric piano 2", "harpsichord", "clavinet",
    "celesta", "glockenspiel", "music box", "vibraphone", "marimba", "xylophone", "tubular bells", "dulcimer",
    "drawbar organ", "percussive organ", "rock organ", "church organ", "reed organ", "accordion", "harmonica", "tango accordion",
    "nylon guitar", "steel guitar", "jazz guitar", "clean guitar", "muted guitar", "overdriven guitar", "distortion guitar", "guitar harmonics",
    "acoustic bass", "finger bass", "pick bass", "fretless bass", "slap bass 1", "slap bass 2", "synth bass 1", "synth bass 2",
    "violin", "viola", "cello", "contrabass", "tremolo strings", "pizzicato strings", "orchestral harp", "timpani",
    "string ensemble 1", "string ensemble 2", "synth strings 1", "synth strings 2", "choir aahs", "voice oohs", "synth voice", "orchestra hit",
    "trumpet", "trombone", "tuba", "muted trumpet", "french horn", "brass section", "synth brass 1", "synth brass 2",
    "soprano sax", "alto sax", "tenor sax", "baritone sax", "oboe", "english horn", "bassoon", "clarinet",
    "piccolo", "flute", "recorder", "pan flute", "blown bottle", "shakuhachi", "whistle", "ocarina",
    "square lead", "sawtooth lead", "calliope lead", "chiff lead", "charang lead", "voice lead", "fifths lead", "bass and lead",
    "new age pad", "warm pad", "polysynth pad", "choir pad", "bowed pad", "metallic pad", "halo pad", "sweep pad",
    "rain fx", "soundtrack fx", "crystal fx", "atmosphere fx", "brightness fx", "goblins fx", "echoes fx", "sci-fi fx",
    "sitar", "banjo", "shamisen", "koto", "kalimba", "bagpipe", "fiddle", "shanai",
    "tinkle bell", "agogo", "steel drums", "woodblock", "taiko drum", "melodic tom", "synth drum", "reverse cymbal",
]
GM_FAMILY = ["keys"] * 8 + ["mallets"] * 8 + ["keys"] * 8 + ["guitar"] * 8 + ["bass"] * 8 + ["strings"] * 8 + ["strings"] * 8 + ["brass"] * 8 + ["winds"] * 16 + ["synth"] * 24 + ["plucked"] * 8 + ["mallets"] * 8
GM_RANGE = {"keys": (36, 91), "mallets": (55, 96), "guitar": (40, 84), "bass": (28, 55), "strings": (43, 91), "brass": (46, 82), "winds": (55, 91), "synth": (45, 91), "plucked": (48, 88)}
GM_ROLE = {"keys": "chords", "mallets": "ostinato", "guitar": "riff", "bass": "bassline", "strings": "pad", "brass": "counter_melody", "winds": "counter_melody", "synth": "pad", "plucked": "ostinato"}

DENSITY_WORDS = {"sparse": 0.25, "simple": 0.3, "gentle": 0.3, "subtle": 0.3, "light": 0.35, "busy": 0.8, "dense": 0.85, "driving": 0.75, "fast": 0.8, "funky": 0.7}


@dataclass
class PartSpec:
    instrument: str
    display_name: str
    family: str
    role: str
    renderer: str
    fallback_renderer: str | None
    low: int
    high: int
    articulation: str
    density: float
    style: str | None
    monophonic: bool
    prompt: str
    matched: str  # table | gm | default
    audio_generative_suggested: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_alias(text: str) -> tuple[Instrument | None, str]:
    best: tuple[int, Instrument | None, str] = (0, None, "")
    for inst in INSTRUMENTS:
        for alias in inst.aliases + (inst.key.replace("_", " "),):
            if re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", text) and len(alias) > best[0]:
                best = (len(alias), inst, alias)
    return best[1], best[2]


def resolve(instrument: str | None, role: str | None = None, prompt: str | None = None, density: float | None = None, style: str | None = None) -> PartSpec:
    """Map an instrument name and/or free text to a part spec. Never raises on unknown text."""
    text = " ".join(t for t in (instrument, prompt) if t).lower().replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    inst, alias = _find_alias(text)
    matched = "table"
    gm_program: int | None = None
    if inst is None:
        for i, name in enumerate(GM_NAMES):
            words = [w for w in name.split() if not w.isdigit() and w not in ("fx", "lead", "pad")]
            if words and all(re.search(r"(?<![a-z])" + re.escape(w) + r"s?(?![a-z])", text) for w in words):
                gm_program = i
                break
        if gm_program is None:
            for i, name in enumerate(GM_NAMES):
                head = name.split()[0]
                if len(head) > 3 and re.search(r"(?<![a-z])" + re.escape(head) + r"s?(?![a-z])", text):
                    gm_program = i
                    break
        matched = "gm" if gm_program is not None else "default"
    # Role: explicit > words in the text > instrument default.
    role_out = None
    if role:
        role_out = ROLE_ALIASES.get(role.lower().strip(), role.lower().strip())
        if role_out not in ROLES:
            role_out = None
    if role_out is None:
        for word in sorted(ROLE_ALIASES, key=len, reverse=True):
            if word in ("bass", "part") or (inst and word in inst.aliases):
                continue
            if re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", text.replace(alias, " ") if alias else text):
                role_out = ROLE_ALIASES[word]
                break
    style_out = style
    if style_out is None:
        for word, st in STYLE_WORDS.items():
            if re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z])", text):
                style_out = st
                break
    dens = density
    if dens is None:
        for word, d in DENSITY_WORDS.items():
            if re.search(r"(?<![a-z])" + word + r"(?![a-z])", text):
                dens = d
                break
    texture = any(w in text for w in TEXTURE_WORDS)
    if inst is not None:
        role_final = role_out or inst.default_role
        spec = PartSpec(inst.key, inst.display, inst.family, role_final, inst.renderer, inst.fallback, inst.low, inst.high,
                        inst.articulations.get(role_final, "sustain" if role_final == "pad" else "normal"), _density(dens, role_final), style_out,
                        inst.monophonic or role_final in ("bassline", "counter_melody", "lead"), text, matched, texture)
    elif gm_program is not None:
        fam = GM_FAMILY[gm_program]
        lo, hi = GM_RANGE[fam]
        role_final = role_out or GM_ROLE[fam]
        spec = PartSpec(GM_NAMES[gm_program].replace(" ", "_"), GM_NAMES[gm_program].title(), fam, role_final, f"gm:{gm_program}", None, lo, hi,
                        "normal", _density(dens, role_final), style_out, role_final in ("bassline", "counter_melody", "lead"), text, matched, texture)
    else:
        # Unknown sound: a soft synth pad keeps the symbolic path usable; suggest audio generation.
        role_final = role_out or "pad"
        preset = {"pad": "pad", "ostinato": "arp80s", "bassline": "bass", "lead": "lead", "counter_melody": "lead", "riff": "pluck", "stabs": "pluck", "chords": "pad"}[role_final]
        lo, hi = (24, 55) if role_final == "bassline" else (45, 91)
        spec = PartSpec("synth_" + preset, (prompt or instrument or "Synth").strip().title()[:40], "synth", role_final, f"synth:{preset}", None, lo, hi,
                        "normal", _density(dens, role_final), style_out, role_final in ("bassline", "counter_melody", "lead"), text, matched, True)
    if spec.style == "funk" and spec.role == "bassline":
        spec.density = max(spec.density, 0.7)
    if spec.style == "80s" and spec.family != "synth" and "synth" in text:
        pass
    return spec


def _density(d: float | None, role: str) -> float:
    if d is not None:
        return float(min(1.0, max(0.0, d)))
    return {"pad": 0.3, "counter_melody": 0.45, "ostinato": 0.6, "bassline": 0.5, "riff": 0.6, "stabs": 0.45, "chords": 0.5, "lead": 0.55}[role]


def capabilities() -> dict[str, Any]:
    return {"roles": list(ROLES), "instruments": [{"key": i.key, "display": i.display, "family": i.family, "renderer": i.renderer, "default_role": i.default_role} for i in INSTRUMENTS],
            "gm_programs": len(GM_NAMES), "free_text": True}
