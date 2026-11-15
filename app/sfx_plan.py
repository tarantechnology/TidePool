"""Planner support for `kind: 'insert'` (SFX inserts from the command bar).

Self-contained so app/agent.py needs only a few touch points:
  * INSERTS_PROPERTY goes into the strict Structured Outputs schema,
  * INSERT_INSTRUCTIONS is appended to the planner instructions,
  * validate_inserts(data, body) checks the model's plan (never trusting its timestamps),
  * rule_plan(prompt) is the offline fallback parser,
  * finalize(plan, duration, features) resolves sounds and anchors into a plan or a clarify.
"""
import math
import re

MAX_INSERTS = 10
ANCHOR_TYPES = ['time', 'start', 'end', 'word', 'beat', 'pause']
RELATIONS = ['at', 'before', 'after']
GAIN_RANGE = (-24.0, 6.0)
OFFSET_RANGE = (-10.0, 10.0)
DUCK_DB = 6.0


def _obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


ANCHOR = _obj({
    'type': {'type': 'string', 'enum': ANCHOR_TYPES},
    'value': {'anyOf': [{'type': 'number'}, {'type': 'string'}, {'type': 'null'}]},
    'relation': {'type': 'string', 'enum': RELATIONS},
    'occurrence': {'type': ['integer', 'null']},
})
INSERT = _obj({
    'sfx_query': {'type': 'string'},
    'anchor': ANCHOR,
    'offset': {'type': ['number', 'null']},
    'gain_db': {'type': ['number', 'null']},
    'duck': {'type': ['boolean', 'null']},
})
# Top-level property: null unless kind=insert.
INSERTS_PROPERTY = {'anyOf': [{'type': 'array', 'items': INSERT}, {'type': 'null'}]}


def instructions(titles):
    return ('''
Use kind=insert to add one-shot sound effects (meme, comedy, editor sounds) from the TidePool sound library as NEW layers; inserts is null for every other kind and actions must be empty for insert. Available sounds include: ''' + titles + '''.
Each insert: sfx_query is a short name for the sound as the user said it (e.g. 'vine boom', 'record scratch', 'laugh track'); the server matches it to the library, so never invent a sound that was not requested.
anchor says where it plays: type=time with value in seconds only when the user typed that time (0:12 means 12, 1:05 means 65); type=start or end for "at the start/end" (value null); type=word with value the exact word or short phrase the user quoted ("when he says 'what'" -> value 'what'), occurrence 1 for "the first time", -1 for "the last time", null for every time; type=beat with value the 1-based beat number ("on beat 4"); type=pause for "during the silence/pause" (value null, occurrence null for the longest pause).
relation: at (the sound starts at the anchor), after (starts when the word ends), before (the sound ends, or lands its hit, at the anchor). offset is extra seconds the user stated ("0.2 seconds before" -> relation at, offset -0.2), else null. gain_db only if the user asks louder (+3) / quieter (-6) or gives dB, else null. duck=true only if the user asks to duck or lower the other audio under it, else null.
You cannot hear the audio: if the anchor is a moment you would have to find by listening or watching (after the joke, before the reveal, when he falls, on the drop), use kind=clarify and ask for a timestamp. At most 10 inserts per request.''')


# ---------------------------------------------------------------- times in user text

TIME = r'(?:\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?|\d+(?:\.\d+)?\s*(?:ms|milliseconds?|s|secs?|seconds?)\b)'


def parse_time(text):
    """'1:05' -> 65, '0:12.5' -> 12.5, '12s' -> 12, '200 ms' -> 0.2, '3' -> 3."""
    text = text.strip().lower()
    if ':' in text:
        parts = [float(p) for p in text.split(':')]
        value = 0.0
        for p in parts:
            value = value * 60 + p
        return value
    match = re.match(r'(\d+(?:\.\d+)?)\s*(ms|milliseconds?)?', text)
    value = float(match.group(1))
    return value / 1000 if match.group(2) else value


def supplied_times(prompt):
    """Every number the user typed, also read as m:ss and as milliseconds."""
    found = {float(n) for n in re.findall(r'(?<![\w.])\d+(?:\.\d+)?', prompt)}
    found |= {parse_time(t) for t in re.findall(r'\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?', prompt)}
    found |= {float(n) / 1000 for n in re.findall(r'(\d+(?:\.\d+)?)\s*(?:ms|milliseconds?)\b', prompt, re.I)}
    return found


def _near(value, options):
    return any(abs(value - o) < 1e-6 for o in options)


def _num(value):
    return type(value) in (int, float) and math.isfinite(value)


def _clamp(value, bounds, default):
    if value is None:
        return default
    if not _num(value):
        raise ValueError('Invalid number')
    return float(min(max(value, bounds[0]), bounds[1]))


def norm_words(text):
    return re.sub(r"[^a-z0-9' ]+", ' ', str(text).lower().replace('’', "'")).split()


# ---------------------------------------------------------------- validating a model plan

def normalize_insert(item, prompt=None, duration=None):
    """Check one insert's shape and ranges. With a prompt, also check that every time,
    offset and word came from the user's own words (the planner may not invent them)."""
    if not isinstance(item, dict):
        raise ValueError('Invalid insert')
    query = item.get('sfx_query')
    if not isinstance(query, str) or not query.strip() or len(query) > 100:
        raise ValueError('Invalid sound name')
    anchor = item.get('anchor')
    if not isinstance(anchor, dict) or anchor.get('type') not in ANCHOR_TYPES:
        raise ValueError('Invalid anchor')
    kind, value = anchor['type'], anchor.get('value')
    relation = anchor.get('relation') or 'at'
    if relation not in RELATIONS:
        raise ValueError('Invalid anchor relation')
    occurrence = anchor.get('occurrence')
    if occurrence is not None and (type(occurrence) is not int or occurrence == 0 or not -1 <= occurrence <= 50):
        raise ValueError('Invalid occurrence')
    supplied = supplied_times(prompt) if prompt is not None else None
    if kind == 'time':
        if not _num(value) or value < 0 or (duration is not None and value > duration):
            raise ValueError('Invalid insert time')
        if supplied is not None and not _near(float(value), supplied):
            raise ValueError('Unsupplied timestamp')
        value = float(value)
    elif kind == 'word':
        if not isinstance(value, str) or not norm_words(value) or len(value) > 80 or len(norm_words(value)) > 6:
            raise ValueError('Invalid word anchor')
        if prompt is not None and ' '.join(norm_words(value)) not in ' '.join(norm_words(prompt)):
            raise ValueError('Unsupplied word')
        value = ' '.join(norm_words(value))
    elif kind == 'beat':
        if value is None:
            value = 1
        if type(value) is float and value.is_integer():
            value = int(value)
        if type(value) is not int or not 1 <= value <= 10000:
            raise ValueError('Invalid beat')
    else:
        value = None
    offset = _clamp(item.get('offset'), OFFSET_RANGE, 0.0)
    if supplied is not None and offset and not _near(abs(offset), supplied):
        raise ValueError('Unsupplied offset')
    duck = item.get('duck')
    if duck is not None and type(duck) is not bool:
        raise ValueError('Invalid duck flag')
    out = {'sfx_query': query.strip(), 'anchor': {'type': kind, 'value': value, 'relation': relation, 'occurrence': occurrence},
           'offset': offset, 'gain_db': _clamp(item.get('gain_db'), GAIN_RANGE, 0.0), 'duck': bool(duck)}
    if item.get('sfx_id') is not None:
        if not isinstance(item['sfx_id'], str) or len(item['sfx_id']) > 64:
            raise ValueError('Invalid sound id')
        out['sfx_id'] = item['sfx_id']
    return out


def validate_inserts(data, body):
    """agent.py hook: returns the normalized inserts (or None for other kinds)."""
    inserts = data.get('inserts')
    if data.get('kind') != 'insert':
        if inserts:
            raise ValueError('Inserts on a non-insert plan')
        return None
    if data.get('actions') or not isinstance(inserts, list) or not 1 <= len(inserts) <= MAX_INSERTS:
        raise ValueError('Invalid insert plan')
    return [normalize_insert(i, body.prompt, body.duration) for i in inserts]


# ---------------------------------------------------------------- finalizing (resolve sounds, check anchors)

def clarify(message, **extra):
    return {**empty_plan('clarify', message), **extra}


def empty_plan(kind, message):
    return {'kind': kind, 'description': None, 'remove': False, 'actions': [], 'message': message,
            'guidance': {'anchors': []}, 'edit_scope': {'action': 'keep', 'interval': None}, 'processing_time': None,
            'separation_mode': None, 'stems': None, 'use_visual': False, 'inserts': None, 'fx': None}


def fmt_time(seconds):
    m, s = divmod(float(seconds), 60)
    return f'{int(m)}:{s:04.1f}'.replace('.0', '') if s % 1 else f'{int(m)}:{int(s):02d}'


def describe(insert):
    a = insert['anchor']
    where = {'time': lambda: f"at {fmt_time(a['value'])}", 'start': lambda: 'at the start', 'end': lambda: 'at the end',
             'pause': lambda: 'in the longest pause' if a['occurrence'] is None else f"in pause {a['occurrence']}",
             'beat': lambda: f"on beat {a['value']}",
             'word': lambda: ('when' if a['relation'] == 'at' else a['relation']) + f" \"{a['value']}\" is said"
             + ('' if a['occurrence'] is None else ' (last time)' if a['occurrence'] == -1 else f" (time {a['occurrence']})")}[a['type']]()
    if a['type'] == 'time' and a['relation'] != 'at':
        where = f"{a['relation']} {fmt_time(a['value'])}"
    if insert['offset']:
        where += f" {insert['offset']:+.2f}s"
    return f"{insert['title']} {where}"


def finalize(plan, duration, features):
    """Resolve each sfx_query against the library and check that its anchor can be found.
    Returns the plan ready for /sfx-preview, or a clarify the editor shows as is."""
    from .sfx_library import resolve
    resolved = []
    for insert in plan['inserts']:
        entry, suggestions = (None, [])
        if insert.get('sfx_id'):
            from .sfx_library import get
            try:
                entry = get(insert['sfx_id'])
            except ValueError:
                entry = None
        if entry is None:
            entry, suggestions = resolve(insert['sfx_query'])
        if entry is None:
            hint = f" Closest: {', '.join(suggestions)}." if suggestions else ''
            return clarify(f"I don't have a \"{insert['sfx_query']}\" sound in the library.{hint} Try one of those, or describe it differently.",
                           sfx_unmatched=True, sfx_query=insert['sfx_query'])
        anchor = insert['anchor']
        if anchor['type'] == 'word' and not features.get('words', {}).get('available'):
            return clarify(f"I can't find words in the speech yet (word timing isn't available). Give me a timestamp instead, "
                           f"like \"{entry['title'].split(' (')[0].lower()} at 0:12\".", sfx_needs_time=True, sfx_query=insert['sfx_query'])
        if anchor['type'] == 'beat' and not features.get('beats', {}).get('available'):
            return clarify('Beat detection is not available yet. Give me a timestamp instead, like "at 0:12".', sfx_needs_time=True, sfx_query=insert['sfx_query'])
        if anchor['type'] == 'time' and anchor['value'] > duration:
            return clarify(f"{fmt_time(anchor['value'])} is past the end of the clip ({fmt_time(duration)}).")
        resolved.append({**insert, 'sfx_id': entry['id'], 'title': entry['title']})
    plan = {**plan, 'inserts': resolved, 'actions': []}
    if not plan.get('message'):
        plan['message'] = 'Adding ' + '; '.join(describe(i) for i in resolved) + '. Preview it, then apply.'
    return plan


# ---------------------------------------------------------------- offline rule-based parser

_LEAD = re.compile(r"^(?:(?:hey|ok(?:ay)?|now|so|yo|pls|please|can\s+(?:you|we|i)|could\s+(?:you|we)|would\s+you|will\s+you|"
                   r"i\s+(?:want|need)(?:\s+you)?(?:\s+to)?|i['’]?d\s+like(?:\s+you)?(?:\s+to)?|let['’]?s|go\s+ahead\s+and|just|also)\s+)+", re.I)
_VERB = re.compile(r"^(?:add|insert|put|drop|throw(?:\s+in)?|stick|slap|play|place|include|give\s+(?:me|it|us)|have|get|use|layer|pop\s+in|chuck(?:\s+in)?|hit\s+(?:me\s+with|it\s+with))\b\s*(?:in\s+)?", re.I)
_FX_WORDS = re.compile(r'\b(?:reverb|echo|delay|chorus|flanger|phaser|tremolo|distortion|overdrive|eq|equali[sz]er|bass\s+boost|treble|'
                       r'compression|compressor|autotune|auto-tune|pitch|fade|filter|lo-?fi|bitcrush|telephone|radio|megaphone|underwater|'
                       r'nightcore|slowed|sped\s+up|normali[sz]e|loudness|denoise|noise\s+reduction|vocals?\s+effect)\b', re.I)
_REPLACE = re.compile(r'\b(?:replace|replacing|swap|substitute|instead\s+of|sound\s+like|turn\s+(?:it|him|her|the\s+\w+)\s+into)\b', re.I)
_OTHER_OPS = re.compile(r'^(?:mute|unmute|solo|isolate|separate|split|remove|delete|cut|extract|lower|raise|boost|turn|make|set|bleep|duck\s+the|normali[sz]e)\b', re.I)
# Not library sounds: the music, silence, a layer. "drop the music at 0:30" / "add silence at the start" go to the planner.
_NOT_A_SOUND = re.compile(r"^(?:the\s+|some\s+|a\s+|an\s+|my\s+|this\s+|that\s+)?(?:music|song|beat|instrumental|bgm|backing\s+track|silence|quiet|nothing|"
                          r"vocals?|voice|dialog(?:ue)?|speech|audio|sound|volume|track|layer|everything|mix|bass|drums?|guitar|piano)$", re.I)
# Profanity beeps ("bleep the swearing", "put a bleep over the curse words") belong to the censorship flow.
_CENSORSHIP = re.compile(r'\b(?:cuss(?:ing)?|swear(?:ing|s)?|swear\s+words?|profanity|curse|curses|cursing|expletives?)\b', re.I)
_CENSOR_VERB = re.compile(r'\b(?:beep|bleep|censor|mute|remove)\b', re.I)
_CENSOR_NEGATED = re.compile(r"\b(?:do\s+not|don['’]?t|never)\s+(?:beep|bleep|censor|mute|remove)\b", re.I)


def requests_censorship(text):
    """Mirror of the editor's censorshipIntent: an explicit ask to beep profanity (the censorship preview owns it)."""
    text = str(text or '')
    return bool(_CENSOR_VERB.search(text) and _CENSORSHIP.search(text) and not _CENSOR_NEGATED.search(text))


_ORD = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'last': -1, 'final': -1}
_SPEECH = r"(?:says?|said|saying|goes|yells?|yelled|shouts?|shouted|screams?|screamed|whispers?|mentions?|asks?|asked|utters?)"
_SPEAKER = r"(?:(?:he|she|they|i|you|we|someone|somebody|anyone|everyone|the\s+(?:guy|girl|man|woman|kid|host|speaker|narrator|teacher|dude))\s+)?"
_UNSUPPORTED = re.compile(r"\b(?:after|before|during|on|at|when|right\s+after|right\s+before)\s+(?:the|his|her|their|that|this|my|a)\s+"
                          r"(joke|punch\s*line|punchline|reveal|fall|trip|fail|kiss|drop|beat\s+drop|twist|plot\s+twist|scene\s+change|cut|transition|"
                          r"jump|hit|slap|laugh|line|moment|part|explosion|crash|entrance|zoom)\b", re.I)
_UNSUPPORTED_EVENT = re.compile(r"\b(?:when|as|after|before)\s+(?:he|she|they|someone|somebody|the\s+\w+)\s+(?!" + _SPEECH + r"\b)(?!is\b|'s\b)(\w+)", re.I)


_QUOTE = re.compile(r'"([^"]+)"|“([^”]+)”|‘([^’]+)’|(?:^|(?<=\s))\'(.+?)\'(?=$|[\s,.!?;])')


def _protect_quotes(text):
    """Swap quoted phrases for placeholders so "and" or "at" inside them cannot split the request."""
    quotes = []

    def keep(m):
        quotes.append(next(g for g in m.groups() if g is not None).strip())
        return f' «{len(quotes) - 1}» '
    return _QUOTE.sub(keep, text), quotes


def _unprotect(text, quotes):
    return re.sub(r'\s*«(\d+)»\s*', lambda m: ' ' + quotes[int(m.group(1))] + ' ', text).strip()


def _strip_lead(text):
    text = _LEAD.sub('', text.strip())
    stripped = _VERB.sub('', text, count=1)
    return stripped.strip(), stripped != text


def _take(pattern, text, flags=re.I):
    match = re.search(pattern, text, flags)
    if not match:
        return None, text
    return match, (text[:match.start()] + ' ' + text[match.end():]).strip()


def _modifiers(seg):
    gain, duck = None, None
    m, seg = _take(r'(?:(?:at|by|with)\s+)?([+-−]?\s*\d+(?:\.\d+)?)\s*db\b(?:\s+(louder|quieter|softer|lower|higher|down|up))?', seg)
    if m:
        value = float(m.group(1).replace('−', '-').replace(' ', ''))
        if m.group(2) and m.group(2).lower() in ('quieter', 'softer', 'lower', 'down'):
            value = -abs(value)
        gain = value
    m, seg = _take(r'\b(?:very|really|super|extra|way)\s+(quiet(?:ly)?|soft(?:ly)?|loud(?:ly)?)\b', seg)
    if m:
        gain = -12.0 if m.group(1).lower().startswith(('quiet', 'soft')) else 6.0
    m, seg = _take(r'\b(quieter|softer|quietly|softly|subtle|subtly|quiet|soft|in\s+the\s+background|at\s+low\s+volume|low\s+volume)\b', seg)
    if m and gain is None:
        gain = -6.0
    m, seg = _take(r'\b(louder|loudly|loud|at\s+full\s+volume|full\s+volume)\b', seg)
    if m and gain is None:
        gain = 3.0
    m, seg = _take(r'\b(?:no|without|don\'?t)\s+duck(?:ing)?\b', seg)
    if m:
        duck = False
    m, seg = _take(r'\b(?:with\s+)?(?:auto[- ]?)?duck(?:ing|ed|s)?\b(?:\s+(?:the|all|other|everything|music|rest|audio|layers?|tracks?|background|under(?:neath)?|it|else|of|sounds?))*', seg)
    if m:
        duck = True
    m, seg = _take(r'\b(?:and\s+)?(?:lower|turn\s+down|dip|quiet(?:en)?)\s+(?:the\s+)?(?:music|rest|others?|everything(?:\s+else)?|background|other\s+(?:layers|tracks|sounds|audio))(?:\s+(?:under(?:neath)?\s+it|a\s+bit))?\b', seg)
    if m:
        duck = True
    return seg, gain, duck


def _word_phrase(rest, quotes):
    """The quoted phrase, or up to four plain words, at the start of `rest`."""
    m = re.match(r'\s*«(\d+)»', rest)
    if m:
        return quotes[int(m.group(1))], rest[m.end():]
    words = []
    tail = rest.strip().split()
    while tail and len(words) < 4 and not re.match(r"^(?:at|and|with|then|but|in|on|louder|quieter|quietly|softly|loud|plus)$", tail[0], re.I):
        words.append(tail.pop(0))
    return ' '.join(words).strip(' ,.!?'), ' '.join(tail)


def _anchor(seg, quotes):
    """Find one anchor in a segment. Returns (anchor | {'unsupported': text} | None, offset, remaining text)."""
    offset = 0.0
    m = re.search(r'(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|secs?|seconds?)\s+(before|after|earlier\s+than|later\s+than)\b', seg, re.I)
    if m:
        amount = float(m.group(1)) / (1000 if m.group(2).lower().startswith('m') else 1)
        before = m.group(3).lower().startswith(('before', 'earlier'))
        offset = -amount if before else amount
        seg = seg[:m.start()] + (' when ' if before else ' after ') + seg[m.end():]
        seg = re.sub(r'\b(when|after)\s+(?=\d)', '', seg)  # "0.2s before 0:30" -> bare time
    # Pauses / awkward silence.
    m, rest = _take(r'\b(?:during|in|at|over|for|into)\s+(?:the\s+|that\s+|an?\s+)?(?:(first|second|third|last|longest)\s+)?'
                    r'(?:awkward\s+|long\s+|dead\s+|uncomfortable\s+|big\s+)?(?:silence|pause|gap|quiet\s+(?:bit|part|moment))\b', seg)
    if m:
        occ = None if not m.group(1) or m.group(1).lower() == 'longest' else _ORD[m.group(1).lower()]
        return {'type': 'pause', 'value': None, 'relation': 'at', 'occurrence': occ}, offset, rest
    # Words.
    m = re.search(r'\b(?:(?P<ord>the\s+(?:first|second|third|fourth|fifth|last|final)\s+time)|(?P<rel>when(?:ever)?|every\s*time|each\s+time|any\s*time|'
                  r'right\s+after|right\s+before|just\s+after|just\s+before|as\s+soon\s+as|after|before|as))\s+' + _SPEAKER + _SPEECH + r'\b(?:\s+the\s+word)?', seg, re.I)
    if not m:
        m = re.search(r'\b(?P<rel>on|at|after|before)\s+the\s+word\b', seg, re.I)
    if m:
        phrase, tail = _word_phrase(seg[m.end():], quotes)
        if phrase:
            groups = m.groupdict()
            rel_text = (groups.get('rel') or '').lower()
            occurrence = None
            if groups.get('ord'):
                occurrence = _ORD[groups['ord'].split()[1].lower()]
            relation = 'after' if rel_text.endswith('after') or rel_text == 'as soon as' else 'before' if rel_text.endswith('before') else 'at'
            return ({'type': 'word', 'value': ' '.join(norm_words(phrase)), 'relation': relation, 'occurrence': occurrence},
                    offset, (seg[:m.start()] + ' ' + tail).strip())
    m = _UNSUPPORTED.search(seg) or _UNSUPPORTED_EVENT.search(seg)
    if m:
        return {'unsupported': m.group(0).strip()}, offset, seg
    # Beats.
    m, rest = _take(r'\bon\s+(?:the\s+)?(?:beat\s+(?P<n>\d+)|(?P<n2>\d+)(?:st|nd|rd|th)\s+beat|(?P<w>first|second|third|fourth|fifth)\s+beat|'
                    r'(?P<down>(?:first\s+)?downbeat)|(?P<any>beat))\b', seg)
    if m:
        number = m.group('n') or m.group('n2')
        value = int(number) if number else _ORD.get((m.group('w') or 'first').lower(), 1)
        return {'type': 'beat', 'value': value, 'relation': 'at', 'occurrence': None}, offset, rest
    # Start / end.
    m, rest = _take(r'\b(?:at|by|on|for|to|right\s+at)\s+the\s+(?:very\s+)?end(?:ing)?(?:\s+of\s+(?:the\s+|this\s+|my\s+)?(?:video|clip|song|track|audio|it|everything))?\b|'
                    r'\bat\s+the\s+(?:outro|close)\b|\bto\s+end\s+(?:it|the\s+(?:video|clip))\b|\bas\s+(?:an?\s+|the\s+)?(?:outro|ending)\b|\bat\s+the\s+finish\b', seg)
    if m:
        return {'type': 'end', 'value': None, 'relation': 'at', 'occurrence': None}, offset, rest
    m, rest = _take(r'\b(?:at|in|on|right\s+at)\s+the\s+(?:very\s+)?(?:start|beginning|top|intro|opening)(?:\s+of\s+(?:the\s+|this\s+|my\s+)?(?:video|clip|song|track|audio|it))?\b|'
                    r'\bto\s+(?:start|open)\s+(?:it|the\s+(?:video|clip))\b|\bas\s+(?:an?\s+|the\s+)?intro\b', seg)
    if m:
        return {'type': 'start', 'value': None, 'relation': 'at', 'occurrence': None}, offset, rest
    # Explicit times.
    m, rest = _take(r'(?:\b(?P<rel>at|@|around|near|on|by|from|after|before|right\s+at|right\s+after|right\s+before)\s+)?(?:the\s+)?(?P<t>' + TIME +
                    r'|\d{1,2}:\d{2}(?:\.\d+)?)(?:\s+mark)?|\bat\s+(?P<bare>\d+(?:\.\d+)?)\b(?!\s*(?:db|%|x\b|times))', seg)
    if m:
        value = parse_time(m.group('t') or m.group('bare'))
        rel_text = (m.group('rel') or '').lower()
        relation = 'after' if rel_text.endswith('after') else 'before' if rel_text.endswith('before') else 'at'
        return {'type': 'time', 'value': value, 'relation': relation, 'occurrence': None}, offset, rest
    if re.fullmatch(r'\s*(?:and\s+)?(?:at\s+)?(' + TIME + r'|\d{1,2}:\d{2}(?:\.\d+)?)\s*', seg, re.I):
        return {'type': 'time', 'value': parse_time(seg.strip().split()[-1]), 'relation': 'at', 'occurrence': None}, offset, ''
    return None, offset, seg


def _sfx_text(text, quotes):
    text = _unprotect(text, quotes)
    text = re.sub(r'^(?:and|also|plus|then)\s+', '', text.strip(), flags=re.I)
    text, _ = _strip_lead(text)
    text = re.sub(r'^(?:(?:a|an|the|some|that|this|one|another)\s+)+', '', text, flags=re.I)
    text = re.sub(r'\b(?:sound\s+effects?|sfx|sound|effect|noise|please|for\s+me|here|in\s+there|to\s+(?:it|this|the\s+(?:video|clip|song|track)))\b', ' ', text, flags=re.I)
    text = re.sub(r'\s+(?:in|on|at|to|into)\s*$', '', ' '.join(text.split()), flags=re.I)
    return text.strip(' ,.!?-')


def rule_plan(prompt):
    """Offline parse of insert requests. Returns an insert plan, a clarify, or None when the
    prompt is not an insert request (so other planners can handle it)."""
    text = ' '.join(str(prompt).strip().split()).rstrip('.!?')
    if not text or _REPLACE.search(text) or _CENSORSHIP.search(text):
        return None
    text, quotes = _protect_quotes(text)
    body, has_verb = _strip_lead(text)
    if not has_verb and _OTHER_OPS.match(body):
        return None
    if _FX_WORDS.search(_unprotect(re.sub(r'«\d+»', ' ', body), quotes)) and not re.search(r'«\d+»', body):
        return None
    from .sfx_library import resolve
    segments = [s for s in re.split(r'\s*(?:,|;|\band\s+then\b|\bthen\b|\balso\b|\bplus\b|\band\b)\s*', body, flags=re.I) if s.strip()]
    inserts, duck_all, unsupported, anchor_found = [], None, None, False
    for seg in segments:
        seg, has = _strip_lead(seg)
        seg, gain, duck = _modifiers(seg)
        if duck is not None:
            duck_all = duck
        anchor, offset, rest = _anchor(seg, quotes)
        if anchor and 'unsupported' in anchor:
            unsupported = anchor['unsupported']
            anchor = None
            rest = re.sub(re.escape(unsupported), ' ', rest, flags=re.I)
        anchor_found = anchor_found or anchor is not None
        name = _sfx_text(rest, quotes)
        if not name:
            if anchor and inserts:
                last = inserts[-1]
                if last['anchor'] is None:
                    last['anchor'], last['offset'] = anchor, offset or last['offset']
                else:
                    inserts.append({**last, 'anchor': anchor, 'offset': offset})
            if gain is not None and inserts:
                inserts[-1]['gain_db'] = gain
            continue
        entry, suggestions = resolve(name)
        inserts.append({'sfx_query': name, 'anchor': anchor, 'offset': offset, 'gain_db': gain, 'duck': None,
                        '_entry': entry, '_suggestions': suggestions})
    if not inserts:
        return None
    matched = [i for i in inserts if i['_entry'] is not None]
    looks_like_insert = has_verb and (anchor_found or unsupported) or bool(matched)
    if not looks_like_insert:
        return None
    if not matched and not (has_verb and (anchor_found or unsupported)):
        return None
    # Sounds without an anchor share the next (or previous) one: "a boom and a rimshot at 0:12".
    for i, insert in enumerate(inserts):
        if insert['anchor'] is None:
            later = next((j['anchor'] for j in inserts[i + 1:] if j['anchor']), None)
            earlier = next((j['anchor'] for j in reversed(inserts[:i]) if j['anchor']), None)
            insert['anchor'] = later or earlier
    missing = next((i for i in inserts if i['_entry'] is None), None)
    if missing and not matched and all(_NOT_A_SOUND.match(i['sfx_query'].strip()) for i in inserts):
        return None
    if missing:
        hint = f" Closest: {', '.join(missing['_suggestions'])}." if missing['_suggestions'] else ''
        return clarify(f"I don't have a \"{missing['sfx_query']}\" sound in the library.{hint} Try one of those, or describe it differently.",
                       sfx_unmatched=True, sfx_query=missing['sfx_query'])
    if unsupported and any(i['anchor'] is None for i in inserts):
        return clarify(f"I can't tell from the audio when \"{unsupported}\" happens. Give me a timestamp, e.g. "
                       f"\"{inserts[0]['_entry']['title'].split(' (')[0].lower()} at 0:42\".", sfx_needs_time=True, sfx_query=inserts[0]['sfx_query'])
    if any(i['anchor'] is None for i in inserts):
        title = inserts[0]['_entry']['title']
        return clarify(f'When should the {title} play? Say a time ("at 0:12"), "at the start/end", or "when he says \'word\'".', sfx_needs_time=True,
                       sfx_query=inserts[0]['sfx_query'])
    if len(inserts) > MAX_INSERTS:
        return clarify(f'I can add at most {MAX_INSERTS} sounds per request.')
    out = []
    for insert in inserts:
        item = {k: insert[k] for k in ('sfx_query', 'anchor', 'offset', 'gain_db')}
        item['duck'] = duck_all
        item['sfx_id'] = insert['_entry']['id']
        out.append(normalize_insert(item))
    plan = empty_plan('insert', '')
    plan['inserts'] = out
    return plan
