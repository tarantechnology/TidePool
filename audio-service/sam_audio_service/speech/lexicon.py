"""Word-level text rules on top of a transcript: normalisation, profanity, fillers, pauses.

Pure Python, no model. ``/v1/words`` returns indices into its ``words`` list, so the app can
turn them into bleep / mute ranges without re-implementing any of this.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Iterable, Sequence

# Characters stripped from both ends of a word before matching. "*" is kept so censored
# partials ("f***") survive; inner apostrophes and hyphens are kept ("don't", "mother-fucker").
_EDGE = " \t\n\"'`“”‘’«»()[]{}<>.,;:!?¡¿…–—-_/\\|"


def normalize(word: str) -> str:
    """Lower-case, surrounding punctuation removed, typographic apostrophes folded."""
    return word.strip().replace("’", "'").replace("‘", "'").strip(_EDGE).lower()


@dataclass
class ProfanityList:
    """Exact words and ``prefix:`` stems; see ``profanity_en.txt`` for the file format."""

    words: set[str] = field(default_factory=set)
    prefixes: tuple[str, ...] = ()
    source: str = "builtin:profanity_en.txt"

    @classmethod
    def parse(cls, lines: Iterable[str], source: str) -> "ProfanityList":
        words: set[str] = set()
        prefixes: list[str] = []
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("prefix:"):
                stem = normalize(line[7:])
                if stem:
                    prefixes.append(stem)
            else:
                entry = normalize(line)
                if entry:
                    words.add(entry)
        return cls(words, tuple(sorted(set(prefixes))), source)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "ProfanityList":
        if path:
            return cls.parse(Path(path).read_text(encoding="utf-8").splitlines(), str(path))
        text = resources.files("sam_audio_service.speech").joinpath("profanity_en.txt").read_text(encoding="utf-8")
        return cls.parse(text.splitlines(), "builtin:profanity_en.txt")

    def with_extra(self, extra: Sequence[str]) -> "ProfanityList":
        if not extra:
            return self
        more = ProfanityList.parse(extra, "request")
        return ProfanityList(self.words | more.words, tuple(sorted(set(self.prefixes) | set(more.prefixes))), self.source + "+request")

    def __len__(self) -> int:
        return len(self.words) + len(self.prefixes)

    def matches(self, word: str) -> bool:
        token = normalize(word)
        if not token:
            return False
        # "mother-fucker", "bull-shit": also try the joined form and each part.
        candidates = {token, token.replace("-", ""), *(p for p in token.split("-") if p)}
        # Possessives/plurals the list does not spell out ("bitch's").
        candidates |= {c[:-2] for c in list(candidates) if c.endswith("'s")}
        for c in candidates:
            if c in self.words or any(c.startswith(p) for p in self.prefixes):
                return True
        return False


# Hesitation sounds Whisper writes (with the filler-biased prompt). Kept conservative:
# "hmm"/"mm"/"ah"/"oh" are often meaningful replies, so they are not fillers.
FILLER_WORDS = frozenset({"um", "umm", "ummm", "uh", "uhh", "uhhh", "uhm", "er", "erm", "ehm", "em"})
# "like" is a filler only when set off by commas or pauses on both sides ("it was, like, huge").
_LIKE_PAUSE = 0.25


def find_fillers(words: Sequence[dict]) -> list[int]:
    """Indices of filler words. ``words`` items carry ``word`` (raw, with punctuation),
    ``start`` and ``end``."""
    out: list[int] = []
    for i, w in enumerate(words):
        token = normalize(w["word"])
        if token in FILLER_WORDS:
            out.append(i)
        elif token == "like" and _like_is_filler(words, i):
            out.append(i)
    return out


def _like_is_filler(words: Sequence[dict], i: int) -> bool:
    raw = words[i]["word"].strip()
    prev = words[i - 1] if i > 0 else None
    nxt = words[i + 1] if i + 1 < len(words) else None
    before = prev is None or prev["word"].rstrip().endswith((",", "...", "…", "—", "-")) or (words[i]["start"] - prev["end"]) >= _LIKE_PAUSE
    after = raw.endswith((",", "...", "…", "—")) or (nxt is not None and nxt["start"] - words[i]["end"] >= _LIKE_PAUSE)
    # A sentence-final "like" ("I don't feel like it" never ends on it; "stuff like." rarely)
    # is not counted: only a "like" between two pauses/commas is.
    return before and after and nxt is not None


def find_profanity(words: Sequence[dict], profanity: ProfanityList) -> list[int]:
    return [i for i, w in enumerate(words) if profanity.matches(w["word"])]


def find_pauses(words: Sequence[dict], min_pause: float, duration: float | None = None) -> list[dict]:
    """Silences of at least ``min_pause`` seconds between consecutive words (not before the
    first or after the last word: those are lead-in/out, not pauses)."""
    out = []
    for i in range(len(words) - 1):
        start, end = words[i]["end"], words[i + 1]["start"]
        if end - start >= min_pause:
            out.append({"start": round(start, 3), "end": round(end, 3), "after_word": i})
    return out


def word_tokens(text: str) -> list[str]:
    """Split free text into words (used by the mock backend)."""
    return [t for t in re.split(r"\s+", text.strip()) if t]
