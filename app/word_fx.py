"""Effects on spoken words: "bass boost the bad words", "add echo whenever he says McLovin".

The command bar plans the effect itself as usual (kind=fx) from the ask with the word clause taken out, and this
module adds ``on_words`` to that plan: ``{'profanity': bool, 'words': [...]}``. When the effect preview renders,
``resolve`` transcribes the speech locally (app/censorship.py's faster-whisper runtime, word timestamps), finds the
words and gives the effect one window per word, so the effect only plays while the word is said.

Limits, said plainly: transcription can mishear or miss words, and the profanity list is a short fixed list
(app/censorship.PROFANITY plus the curse words below).
"""
from __future__ import annotations

import re

#: Extra curse words for "bad words" beyond the censorship list (which stays exact for bleeping).
EXTRA_PROFANITY = frozenset('damn goddamn goddammit dammit hell crap piss pissed dick dicks dickhead prick douche douchebag '
                            'whore slut tits jackass dumbass badass bitchin wanker bollocks'.split())
MAX_WORDS = 10
MAX_WINDOWS = 64
PAD = .06

_PROFANITY_NOUN = r"(?:bad|swear|curse|cuss|dirty|naughty|foul|explicit)\s*[- ]?\s*words?|swears?|swearing|curses|cursing|cuss(?:es|ing)?|profanit(?:y|ies)|expletives?|f[- ]bombs?"
_SUBJECT = r"(?:he|she|they|someone|somebody|anyone|anybody|people|everyone|a character|any(?:one|body)|person\s*\d+|[a-z][\w'.-]*)"
_PROFANITY_CLAUSES = [
    # "whenever he swears", "every time someone curses"
    re.compile(rf"\b(?:whenever|every\s*time|each\s+time|any\s*time|when)\s+{_SUBJECT}\s+(?:swears|curses|cusses|says?\s+(?:a\s+)?(?:bad|swear|curse|cuss)\s*words?)\b", re.I),
    # "on the bad words", "to every swear word", "during the cursing"
    re.compile(rf"\b(?:on|to|over|for|at|during|in)?\s*(?:all\s+(?:of\s+)?)?(?:the|every|each|any|all|his|her|their)?\s*(?:{_PROFANITY_NOUN})\b", re.I),
]
# "whenever he says McLovin", "every time someone says 'hey'", "on the word McLovin", "when they say "fake ID""
#: The asked word or phrase: quoted (anything may follow) or bare (ends at punctuation, "and", "or" or the end).
_TARGET = r"(?:[\"“'‘](?P<q>[^\"“”'‘’]+)[\"”'’]|(?P<w>[^\"“”‘’',.;!?]+?)(?=\s*(?:$|[,.;!?]|\band\b)))"
_SAID = re.compile(rf"\b(?:whenever|every\s*time|each\s+time|any\s*time|when)\s+{_SUBJECT}\s+(?:says?|said|saying|yells?|shouts?)\s+"
                   r"(?:the\s+(?:word|name|phrase)\s+)?"+_TARGET+r"", re.I)
_ON_WORD = re.compile(r"(?:\b(?:on|to|over|for|at|during|in)\s+)?\b(?:(?:every|each|all|any)\s+)?(?:(?:instances?|occurrences?|mentions?|uses?|times?)\s+of\s+)?(?:the\s+|every\s+|each\s+)?(?:word|name|phrase)s?\s+"+_TARGET+r"", re.I)
# "every instance of mohammed", "each time the name McLovin comes up"
_INSTANCE = re.compile(r"\b(?:every|each|all|any)\s+(?:the\s+)?(?:instances?|occurrences?|mentions?|uses?)\s+of\s+(?:the\s+(?:word|name|phrase)\s+)?"+_TARGET+r"", re.I)

# Effect words: only an ask that names an effect is a word effect. "Add a vine boom when he says 'what'" stays a
# sound insert (app/replacement.py); "bleep/mute the swearing" stays the censorship flow.
_EFFECT = re.compile(r"\b(?:reverb\w*|echo\w*|delay|chorus|flanger|phaser|tremolo|distort\w*|overdrive|eq|bass(?:\s*[- ]?boost\w*|ier)?|"
                     r"boost\w*|treble|compress\w*|pitch\w*|deep(?:en|er)?|muffle\w*|muffled|lo-?fi|bitcrush\w*|telephone|phone|radio|"
                     r"megaphone|underwater|robot\w*|chipmunk|giant|announcer|church|cathedral|stadium|hall|bathroom|dreamy|warm(?:er)?|"
                     r"bright(?:er)?|louder|quieter|loud|quiet|deep[- ]fried|earrape|8-?bit|vinyl|wobble|stutter|slow(?:er|ed)?|effect)\b", re.I)


def norm(word):
    return re.sub(r"[^a-z0-9']", '', str(word).lower().replace('’', "'")).strip("'")


def parse(prompt):
    """``(rest, on_words)`` when the ask puts an effect on spoken words, else None.
    ``rest`` is the ask without the word clause ("bass boost"), for the normal effect planner."""
    text = str(prompt or '')
    if not _EFFECT.search(text): return None
    profanity = False; words: list[str] = []; rest = text
    # Profanity first, so "every bad word and the word X" never reads "word and the word X" as the asked word.
    for pattern in _PROFANITY_CLAUSES:
        m = pattern.search(rest)
        if m:
            profanity = True
            rest = rest[:m.start()] + ' ' + rest[m.end():]
            break
    for pattern in (_SAID, _INSTANCE, _ON_WORD):
        m = pattern.search(rest)
        if m and m.group('w'):
            # Unquoted and the clause comes first ("whenever he says McLovin add echo"): the word stops at the ask.
            cut = re.search(r"\s+(?:add|put|make|give|apply|turn|use|then)\b", m.group('w'), re.I) or _EFFECT.search(m.group('w'))
            if cut and cut.start() > 0:
                end = m.start('w') + cut.start()
                m = pattern.match(rest[:end], m.start()) or m
        if m and not re.fullmatch(rf"(?:a|an|any|the|some)?\s*(?:{_PROFANITY_NOUN})", (m.group('q') or m.group('w')).strip(), re.I):
            phrases = [m.group('q')] if m.group('q') else [p for p in re.split(r"\s*(?:,|\bor\b|/)\s*", m.group('w').strip()) if p]
            for phrase in phrases:
                key = ' '.join(norm(w) for w in phrase.split() if norm(w))
                if key and key not in words: words.append(key)
            rest = (rest[:m.start()] + ' ' + rest[m.end():])
    if not profanity and not words: return None
    rest = re.sub(r"\s+", ' ', rest).strip(' ,.;:-')
    # Joiners left behind by the removed clauses: "make (every bad word) and (the word X) bass boosted".
    for _ in range(3):
        rest = re.sub(r"^(?:and|or|plus|also)\b\s*|\s*\b(?:and|or|plus)$", '', rest, flags=re.I).strip(' ,.;:-')
        rest = re.sub(r"\b(?:and|or|plus)\s+(?:and|or|plus)\b", 'and', rest, flags=re.I)
        rest = re.sub(r"^((?:please\s+)?(?:make|turn|get|have|give))\s+(?:and|or|plus)\b\s*", r"\1 ", rest, flags=re.I)
    rest = re.sub(r"\b(?:on|to|over|for|at|during|in|with)$", '', rest, flags=re.I).strip(' ,.;:-')
    return rest, {'profanity': profanity, 'words': words[:MAX_WORDS]}


def clean(on_words):
    """Validated ``on_words`` from a request, or ValueError."""
    if not isinstance(on_words, dict): raise ValueError('Invalid word selection.')
    words = []
    for w in on_words.get('words') or []:
        key = ' '.join(norm(p) for p in str(w).split() if norm(p))[:60]
        if key and key not in words: words.append(key)
    profanity = bool(on_words.get('profanity'))
    if not profanity and not words: raise ValueError('Say which words the effect should follow, e.g. "the bad words" or "whenever he says McLovin".')
    if len(words) > MAX_WORDS: raise ValueError(f'At most {MAX_WORDS} words per effect.')
    return {'profanity': profanity, 'words': words}


def describe(on_words):
    parts = (['bad words'] if on_words.get('profanity') else []) + [f'“{w}”' for w in on_words.get('words') or []]
    return ' and '.join(parts) if len(parts) < 3 else ', '.join(parts[:-1]) + ' and ' + parts[-1]


def find(words, on_words):
    """Transcript words (``{'word','start','end'}``) that match: profanity and/or the asked words and phrases."""
    from .censorship import PROFANITY
    bad = PROFANITY | EXTRA_PROFANITY
    keys = [norm(w['word']) for w in words]
    hits = [False] * len(words)
    if on_words.get('profanity'):
        for i, k in enumerate(keys):
            if k in bad or k.replace("'", '') in bad or (k.startswith('fuck') or k.startswith('motherfuck')): hits[i] = True
    for phrase in on_words.get('words') or []:
        parts = [p.replace("'", '') for p in phrase.split()]
        n = len(parts)
        for i in range(len(keys) - n + 1):
            window = [keys[i + j].replace("'", '') for j in range(n)]
            # "mclovin" also matches "mclovin's"; transcripts often split or join names, so join-compare as well.
            if window == parts or (n == 1 and window[0].rstrip('s') == parts[0].rstrip('s')) or ''.join(window) == ''.join(parts):
                for j in range(n): hits[i + j] = True
        if n == 1:  # "McLovin" heard as "Mc Lovin": two transcript words that join into the asked word.
            for i in range(len(keys) - 1):
                if (keys[i] + keys[i + 1]).replace("'", '') == parts[0]: hits[i] = hits[i + 1] = True
    return [w for w, h in zip(words, hits) if h]


def windows(found, duration):
    out: list[list[float]] = []
    for w in sorted(found, key=lambda x: x['start']):
        a, b = max(0., w['start'] - PAD), min(duration, w['end'] + PAD)
        if b - a < .06: b = min(duration, a + .06)
        if b <= a: continue
        if out and a <= out[-1][1] + .02: out[-1][1] = max(out[-1][1], b)
        else: out.append([a, b])
    return out[:MAX_WINDOWS]


VOICE = re.compile(r'\b(?:voice|voices|vocals?|speech|speaking|dialogue|dialog|narration|narrator|speaker|people\s+talking|person\s*\d*|people)\b', re.I)


def voice_rows(rows):
    def text(r): return ' '.join(str(r.get(k) or '') for k in ('name', 'label', 'prompt'))
    return [r for r in rows if not r.get('muted') and float(r.get('volume') or 0) > 0 and r.get('role') != 'effect' and VOICE.search(text(r))]


def resolve(pid, spec, snapshot, source, new_file, stage=None):
    """Turn ``spec['on_words']`` into timeline windows on ``spec['item']['intervals']``; retarget a whole-mix effect
    to the voice layers so the music and background are untouched. Mutates and returns ``spec``."""
    from .censorship import available, transcribe, MAX_DURATION
    from .core import probe
    from .media import export
    if not available(): raise ValueError('Word timing needs local transcription (faster-whisper), which is not set up on this server.')
    on_words = clean(spec['on_words'])
    rows = snapshot['rows']
    voices = voice_rows(rows)
    duration = max([float(r.get('start_offset') or 0) + float(r['duration']) for r in rows] or [probe(source)['duration']])
    if duration > MAX_DURATION: raise ValueError('Word timing is limited to 10 minutes of audio.')
    if stage: stage('transcribing', 15)
    audio = new_file()
    if voices: export(voices, source, audio, 'wav', False)
    elif rows: export(rows, source, audio, 'wav', False)
    else:
        import shutil
        shutil.copyfile(source, audio)
    words, _ = transcribe(audio, duration)
    audio.unlink(missing_ok=True)
    found = find(words, on_words)
    if not found:
        heard = 'No bad words' if on_words['profanity'] and not on_words['words'] else f'I did not hear {describe(on_words)}'
        raise ValueError(f'{heard} in the speech. Transcription can miss words; give a time range instead (for example "from 12 to 13 seconds").')
    spans = windows(found, duration)
    item = spec['item']
    item['intervals'] = [{'start': round(a, 3), 'end': round(b, 3)} for a, b in spans]
    item['interval'] = None
    item['on_words'] = on_words
    item['label'] = f"{item['label']} on {describe(on_words)}"[:60]
    if spec['targets'] == ['mix'] and voices: spec['targets'] = [r['id'] for r in voices]
    spec['word_matches'] = [{'word': w['word'], 'start': round(w['start'], 2), 'end': round(w['end'], 2)} for w in found][:MAX_WINDOWS]
    return spec
