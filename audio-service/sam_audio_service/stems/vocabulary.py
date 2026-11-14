"""Music stem vocabulary: canonical stem names and the wordings callers use for them.

A splitter produces a fixed set of canonical stems (``htdemucs``: vocals, drums, bass,
other; ``htdemucs_6s`` adds guitar and piano). ``instrumental`` is virtual: the sum of every
non-vocal stem the splitter produced. Caller wording ("lead vocal", "the beat", "karaoke")
is mapped onto these names in the same way ``contracts.MODEL_ALIASES`` maps prompt wording
onto the prompted model's caption vocabulary.
"""
from __future__ import annotations

from collections.abc import Iterable

from ..contracts import normalise_description

# Display order for stems in a response (a full split lists them in this order).
CANONICAL_STEMS: tuple[str, ...] = ("vocals", "drums", "bass", "guitar", "piano", "other")
INSTRUMENTAL = "instrumental"

STEM_ALIASES: dict[str, str] = {
    **{w: "vocals" for w in (
        "vocals", "vocal", "voice", "voices", "singing", "singer", "singers", "vocalist", "lead vocal", "lead vocals",
        "lead voice", "backing vocals", "acapella", "a cappella", "vocal track", "vocals track")},
    **{w: "drums" for w in ("drums", "drum", "drum kit", "drumkit", "percussion", "beat", "beats", "drum track")},
    **{w: "bass" for w in ("bass", "bassline", "bass line", "bass guitar", "electric bass", "bass track")},
    **{w: "guitar" for w in ("guitar", "guitars", "electric guitar", "acoustic guitar", "guitar track")},
    **{w: "piano" for w in ("piano", "keys", "keyboard", "keyboards", "piano track")},
    **{w: "other" for w in ("other", "other instruments")},
    **{w: INSTRUMENTAL for w in (
        "instrumental", "instrumentals", "accompaniment", "backing track", "karaoke", "karaoke track",
        "music without vocals", "no vocals", "without vocals")},
}


def canonical_stem(text: str) -> str | None:
    """Canonical stem for a caller's wording, or None when it is not a music stem.

    The text goes through the same normalisation as prompts first ("Isolate the Vocals!" ->
    "vocals"; ``MODEL_ALIASES`` turns "singer" into "singing"), so both prompt forms and
    ``options.stems`` entries resolve identically.
    """
    return STEM_ALIASES.get(normalise_description(text))


def display_name(canonical: str) -> str:
    return canonical.title()


def components(canonical: str, available: Iterable[str]) -> list[str]:
    """The splitter stems a canonical name stands for (``instrumental`` = every non-vocal stem)."""
    available = tuple(available)
    if canonical == INSTRUMENTAL:
        return [s for s in available if s != "vocals"]
    return [canonical] if canonical in available else []


def ordered(available: Iterable[str]) -> list[str]:
    """Splitter stems in display order (vocals, drums, bass, guitar, piano, other)."""
    available = tuple(available)
    return [s for s in CANONICAL_STEMS if s in available] + [s for s in available if s not in CANONICAL_STEMS]


def requestable(available: Iterable[str]) -> list[str]:
    """Names a caller may request from a splitter with these stems, in display order."""
    names = ordered(available)
    return names + ([INSTRUMENTAL] if "vocals" in names and len(names) > 1 else [])
