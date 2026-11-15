"""Planner + client glue for chat-fx Phase E ("add violins to the song").

Self-contained so it can be wired into ``app/agent.py`` without touching the existing kinds:

* ``GENERATE`` - the strict Structured-Outputs schema fragment for ``kind: 'generate'``
  (the planner LLM only picks instrument / role / sections / density; it never writes notes).
* ``validate_generate(plan, sections)`` - clamps and checks a planner plan.
* ``rule_plan(prompt)`` - offline fallback: "add a funky bassline in the chorus" -> plan.
* ``request_generate_part(...)`` - POSTs the project's mix + stems to the audio service
  ``/v1/generate-part`` and returns the JSON (audio is a new ``music`` layer for a
  ``generate-preview`` candidate: compare / apply / undo as usual).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import httpx

ROLES = ['pad', 'counter_melody', 'ostinato', 'bassline', 'riff', 'stabs', 'chords', 'lead']
SECTIONS = ['intro', 'verse', 'chorus', 'bridge', 'outro']
ENGINES = ['auto', 'symbolic', 'amt', 'musicgen']

GENERATE = {
    'type': 'object', 'additionalProperties': False,
    'required': ['instrument', 'prompt', 'role', 'sections', 'interval', 'density', 'engine'],
    'properties': {
        'instrument': {'type': 'string'},
        'prompt': {'type': 'string'},
        'role': {'anyOf': [{'type': 'string', 'enum': ROLES}, {'type': 'null'}]},
        'sections': {'anyOf': [{'type': 'array', 'items': {'type': 'string', 'enum': SECTIONS}}, {'type': 'null'}]},
        'interval': {'anyOf': [{'type': 'object', 'additionalProperties': False, 'required': ['start', 'end'],
                                'properties': {'start': {'type': 'number'}, 'end': {'type': 'number'}}}, {'type': 'null'}]},
        'density': {'type': ['number', 'null']},
        'engine': {'type': 'string', 'enum': ENGINES},
    },
}

PLANNER_NOTE = ("Use kind=generate when the user asks to ADD a new musical part to a song (\"add violins\", \"add a piano part\", "
                "\"put a string pad under the chorus\", \"add a funky bassline\"). Set instrument to the user's instrument words, prompt to "
                "their full wording, role only if they name one, sections only if they name song sections, interval only for explicit times. "
                "Never invent notes, keys or tempos: the audio service analyses the song.")

_VERB = re.compile(r'^\s*(?:please\s+)?(?:can you\s+)?(add|put|layer|throw in|bring in)\s+', re.I)
_TAIL = re.compile(r'\s+(?:to|in|into|under|over|on|during|for|behind|throughout|at|from|after|before|starting|when)\b.*$', re.I)
# The planner only routes asks that name a playable part; everything else keeps its existing flow.
_INSTRUMENTS = re.compile(r'\b(violins?|viola|cellos?|strings?|string pad|orchestra(?:l)?|fiddle|double bass|upright bass|contrabass|pizzicato|piano|keys|rhodes|'
                          r'organ|harpsichord|celesta|glockenspiel|music box|vibraphone|marimba|xylophone|harp|guitars?|riff|bass(?:line| line| guitar)?|'
                          r'synths?|arp(?:eggio)?|pad|lead|pluck|trumpets?|trombones?|horns?|brass|tuba|sax(?:ophone)?|flutes?|clarinet|oboe|bassoon|'
                          r'choir|voices|aahs|oohs|banjo|accordion|harmonica|sitar|ukulele|mandolin|kalimba|steel drums?|countermelody|counter-melody|melody|chords|bells)\b', re.I)
# Effect / mix / sound-effect words: "add reverb to the strings", "add compression to the bass" stay fx, "add a
# vine boom" stays an insert. Kept in step with frontend/src/agent/generateIntent.ts.
_EFFECTS = re.compile(r'\b(reverb\w*|echo\w*|delay|effects?|eq|filter\w*|fade\w*|boom|sfx|sound effects?|laugh\w*|compress\w*|'
                      r'distort\w*|saturat\w*|overdrive|flanger|phaser|tremolo|auto-?tune|lo-?fi|vinyl|bitcrush\w*|duck\w*|limiter|'
                      r'riser|whoosh|swell|impact|applause|crickets|air\s*horn|rimshot|record scratch|sad trombone)\b', re.I)
# "add more bass", "add some extra low end": an EQ ask, not a bass part.
_EQ_ASK = re.compile(r'^(?:some\s+)?(?:more|less|extra)\s+(?:bass|treble|low[- ]end|lows|highs|top end)\b', re.I)


def rule_plan(prompt: str) -> dict | None:
    """Offline fallback for the common "add <instrument> ..." asks; None when it is not one."""
    text = prompt.strip()
    m = _VERB.match(text)
    if not m or _EFFECTS.search(text) or _EQ_ASK.match(text[m.end():]):
        return None
    rest = _TAIL.sub('', text[m.end():])
    # The instrument must be the thing being added, not a target: "add chorus to the guitar" is an effect.
    if not _INSTRUMENTS.search(rest):
        return None
    rest = re.sub(r'^(?:an?|some|the|more)\s+', '', rest, flags=re.I)
    rest = re.sub(r'\s+(?:part|track|layer)$', '', rest, flags=re.I).strip(' .!')
    if not rest:
        return None
    sections = [s for s in SECTIONS if re.search(r'\b' + s + r'(e?s)?\b', text, re.I)] or None
    return {'instrument': rest, 'prompt': text, 'role': None, 'sections': sections, 'interval': None, 'density': None, 'engine': 'auto'}


def validate_generate(plan: dict, duration: float) -> dict:
    if not isinstance(plan, dict) or not str(plan.get('instrument') or plan.get('prompt') or '').strip():
        raise ValueError('generate plan needs an instrument or prompt')
    out = {'instrument': str(plan.get('instrument') or '')[:80] or None, 'prompt': str(plan.get('prompt') or '')[:400] or None,
           'engine': plan.get('engine') if plan.get('engine') in ENGINES else 'auto'}
    if plan.get('role') is not None:
        if plan['role'] not in ROLES:
            raise ValueError('invalid role')
        out['role'] = plan['role']
    if plan.get('sections'):
        if any(s not in SECTIONS for s in plan['sections']):
            raise ValueError('invalid section')
        out['sections'] = list(plan['sections'])
    if plan.get('interval'):
        a, b = float(plan['interval']['start']), float(plan['interval']['end'])
        if not 0 <= a < b <= duration + 1e-6:
            raise ValueError('interval outside the timeline')
        out['interval'] = {'start': a, 'end': b}
    if plan.get('density') is not None:
        out['density'] = min(1.0, max(0.0, float(plan['density'])))
    return out


def _reference_upload(path, rate: int = 16000):
    """Mono, low-rate FLAC of the mix: enough for loudness/ducking/length once the analysis is cached."""
    import tempfile
    from contextlib import contextmanager
    from .core import run

    @contextmanager
    def ctx():
        handle, name = tempfile.mkstemp(suffix='.flac'); os.close(handle)
        out = Path(name)
        try:
            run('ffmpeg', '-v', 'error', '-y', '-i', str(path), '-vn', '-ac', '1', '-ar', str(rate), '-c:a', 'flac', '-compression_level', '1', str(out))
            with open(out, 'rb') as f:
                yield ('reference.flac', f, 'audio/flac')
        finally:
            out.unlink(missing_ok=True)
    return ctx()


def request_generate_part(mix_path: str | Path, stems: dict[str, str | Path], plan: dict, *, analysis: dict | None = None,
                          seed: int = 0, timeout: float = 600.0, output_sample_rate: int | None = None) -> dict:
    """Call the audio service. ``stems`` maps names (vocals/drums/bass/other/...) to the project's stem files.

    With a cached ``analysis`` only a mono 16 kHz FLAC reference of the mix is uploaded (no stems) and the
    service renders at ``output_sample_rate``; the part comes back as 16-bit FLAC either way."""
    url = urlsplit(os.getenv('SAM_AUDIO_SERVICE_URL', ''))
    if not url.netloc:
        raise RuntimeError('SAM_AUDIO_SERVICE_URL is required')
    meta = dict(plan, seed=seed, output_format='flac', include_analysis=analysis is None)
    if analysis is not None:
        meta['analysis'] = analysis
        if output_sample_rate:
            meta['output_sample_rate'] = int(output_sample_rate)
    # Same lossless FLAC upload as separation (app/media.py service_upload): far smaller through the tunnel.
    from contextlib import ExitStack
    from .media import service_upload
    with ExitStack() as stack:
        if analysis is not None and output_sample_rate:
            name, f, ctype = stack.enter_context(_reference_upload(mix_path))
            files = [('audio', (name, f, ctype))]
        else:
            name, f, ctype = stack.enter_context(service_upload(mix_path))
            files = [('audio', ('mix' + Path(name).suffix, f, ctype))]
            for stem, path in stems.items():
                name, f, ctype = stack.enter_context(service_upload(path))
                files.append(('stem_' + re.sub(r'[^a-z0-9_]', '_', stem.lower()), (Path(path).stem + Path(name).suffix, f, ctype)))
        response = httpx.post(urlunsplit((url.scheme, url.netloc, '/v1/generate-part', '', '')), files=files,
                              data={'metadata': json.dumps(meta)}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def analysis_key(snapshot: dict, rows: list[dict], stem_ids: list[str]) -> str:
    """Cache key for a project's music analysis: the source audio + the exact stem files it used."""
    import hashlib
    parts = [snapshot['source_sha256']] + sorted(f"{sid}:{snapshot['stem_hashes'][sid]}" for sid in stem_ids)
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:32]


CANONICAL = ('vocals', 'drums', 'bass', 'other', 'guitar', 'piano')


def _stem_name(row: dict) -> str | None:
    """Map a project layer to an analysis stem name (music-split canonical labels, else its name)."""
    for value in (row.get('label'), row.get('name')):
        v = str(value or '').lower()
        for c in CANONICAL:
            if c in v:
                return c
    return None


def build(project: dict, payload: dict, candidate_id: str) -> dict:
    """generate-preview candidate: the current mix + its stems -> audio service -> a new music layer."""
    import base64
    import shutil

    from .candidates import revisions, safe_path
    from .core import probe, project_dir, run, uid
    from .media import export
    from .visual import file_sha256

    pid = project['id']; snapshot = payload['_candidate']
    source = safe_path(pid, snapshot['source'])
    if file_sha256(source) != snapshot['source_sha256']: raise ValueError('Source changed; generate another preview.')
    rows = [dict(r) for r in snapshot['rows']]
    for row in rows:
        if file_sha256(safe_path(pid, row['path'])) != snapshot['stem_hashes'][row['id']]:
            raise ValueError('A layer changed; generate another preview.')
    directory = project_dir(pid) / 'stems'; directory.mkdir(parents=True, exist_ok=True)
    created = []
    def new_file(suffix='.wav'):
        path = directory / (uid() + suffix); created.append(path); return path
    try:
        original = new_file()
        if rows: export(rows, source, original, 'wav', False)
        else: shutil.copyfile(source, original)
        stems, stem_ids = {}, []
        for row in rows:
            name = _stem_name(row)
            if name and name not in stems and not row.get('muted'):
                stems[name] = safe_path(pid, row['path']); stem_ids.append(row['id'])
        # Analyse once per project: later asks reuse it and upload only a small mono reference.
        cache_dir = project_dir(pid) / 'analysis'; cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / (analysis_key(snapshot, rows, stem_ids) + '.json')
        analysis = None
        if cache.exists():
            try: analysis = json.loads(cache.read_text())
            except (OSError, ValueError): analysis = None
        before = probe(original)
        result = request_generate_part(original, stems, payload['plan'], seed=int(payload.get('seed') or 0), analysis=analysis,
                                       output_sample_rate=before['sample_rate'])
        if analysis is None and isinstance(result.get('analysis'), dict):
            tmp = cache.with_suffix('.tmp'); tmp.write_text(json.dumps(result['analysis'])); tmp.replace(cache)
        part = new_file()
        encoded = new_file('.' + ('flac' if 'flac' in str(result['audio'].get('format', '')) else 'wav'))
        encoded.write_bytes(base64.b64decode(result['audio']['audio_base64']))
        run('ffmpeg', '-v', 'error', '-y', '-i', str(encoded), '-ar', str(before['sample_rate']), '-ac', str(before['channels']), '-c:a', 'pcm_f32le', str(part))
        encoded.unlink(missing_ok=True)
        after = probe(part)
        if abs(before['duration'] - after['duration']) > 0.05:
            raise ValueError('The generated part does not match the song length.')
        info = result['part']
        name = str(info.get('display_name') or info.get('name') or 'New part')[:100]
        midi_path = None
        if result.get('midi_base64'):
            midi_path = new_file('.mid'); midi_path.write_bytes(base64.b64decode(result['midi_base64']))
        layer = {'id': uid(), 'project_id': pid, 'name': name, 'label': name, 'path': str(part), 'duration': after['duration'], 'start_offset': 0,
                 'volume': 1, 'muted': 0, 'solo': 0, 'automation': '[]', 'mute_ranges': '[]', 'revision': 0, 'role': 'generated',
                 'tags': json.dumps([{'label': info.get('family', 'music').title(), 'score': 1.0}, {'label': name, 'score': 1.0}])}
        base_rows = rows or [{'id': uid(), 'project_id': pid, 'name': 'Original audio', 'path': str(source), 'duration': before['duration'], 'start_offset': 0,
                              'volume': 1, 'muted': 0, 'solo': 0, 'automation': '[]', 'mute_ranges': '[]', 'revision': 0}]
        proposed = base_rows + [layer]
        mix = new_file(); export(proposed, source, mix, 'wav', False)
        warnings = list(result.get('warnings') or [])
        warnings.append('The new part is generated from the song\'s detected chords and beats; detection can be wrong, so listen before applying.')
        return {'target': {'path': str(part)}, 'mix': {'path': str(mix)}, 'original': {'path': str(original)}, 'snapshot': snapshot, 'proposed': proposed,
                'inference': {'model': {'is_mock': False, 'backend': 'generate-' + str((result.get('engine') or {}).get('name', 'symbolic'))}, 'warnings': warnings},
                'provenance': {'kind': 'generate', 'plan': payload['plan'], 'part': info, 'engine': result.get('engine'), 'renderer': result.get('renderer'),
                               'windows': result.get('windows'), 'checks': result.get('checks'), 'mix_report': result.get('mix'),
                               'midi_path': str(midi_path) if midi_path else None, 'layer_id': layer['id'],
                               'stems_used': sorted(stems), 'analysis_cached': analysis is not None, 'stem_revisions': revisions(rows), 'source_sha256': snapshot['source_sha256']}}
    except Exception:
        for path in created: path.unlink(missing_ok=True)
        raise
