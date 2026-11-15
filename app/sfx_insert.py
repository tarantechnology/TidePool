"""SFX inserts: library one-shots placed as NEW layers at a time, word, beat or pause.

Mirrors the replacement 'add' flow: a revision-checked snapshot -> candidate preview
(target = just the inserted sounds, mix, original) -> apply / undo through app.candidates.

Content anchors come from the audio service (never guessed):
  POST {service}/v1/words  multipart audio + metadata  -> {"words": [{"word", "start", "end"}]}
  POST {service}/v1/beats  multipart audio + metadata  -> {"beats": [t | {"time": t}], "downbeats": [...]}
each only when GET {service}/capabilities says words.available / beats.available. Pauses are
found here from the audio's energy (no ML).
"""
import json
import os
import re
import time
from pathlib import Path

from . import candidates
from .core import probe, project_dir, run, uid
from .media import export
from .visual import file_sha256

MAX_PLACEMENTS = 10
SPEECH_LAYER = re.compile(r'speech|speak|voice|dialog|vocal|narrat|talk|person|man|woman|host', re.I)
_features_cache = {'at': 0.0, 'value': None}


# ---------------------------------------------------------------- service capabilities and clients

def service_features():
    """{'words': {'available'}, 'beats': {...}, 'sfx_search': {...}} from the audio service, cached 30 s.
    The fake adapter has none of them, so word and beat anchors ask for a timestamp instead."""
    off = {'words': {'available': False}, 'beats': {'available': False}, 'sfx_search': {'available': False}}
    if os.getenv('INFERENCE_ADAPTER', 'fake') != 'http':
        return off
    if _features_cache['value'] is not None and time.monotonic() - _features_cache['at'] < 30:
        return _features_cache['value']
    import httpx
    from urllib.parse import urlsplit, urlunsplit
    url = urlsplit(os.getenv('SAM_AUDIO_SERVICE_URL', ''))
    try:
        response = httpx.get(urlunsplit((url.scheme, url.netloc, '/capabilities', '', '')), timeout=5)
        response.raise_for_status()
        data = response.json()
        value = {key: {'available': bool(isinstance(data.get(key), dict) and data[key].get('available')),
                       **({'model_id': data[key].get('model_id')} if isinstance(data.get(key), dict) and isinstance(data[key].get('model_id'), str) else {})}
                 for key in off}
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        value = off
    _features_cache.update(at=time.monotonic(), value=value)
    return value


def _service_post(path, audio, metadata, timeout):
    import httpx
    from urllib.parse import urlsplit, urlunsplit
    url = urlsplit(os.getenv('SAM_AUDIO_SERVICE_URL', ''))
    if not url.netloc:
        raise ValueError('The audio service is not configured.')
    try:
        with open(audio, 'rb') as f:
            response = httpx.post(urlunsplit((url.scheme, url.netloc, path, '', '')), files={'audio': ('audio.wav', f, 'audio/wav')},
                                  data={'metadata': json.dumps(metadata)}, timeout=timeout)
    except httpx.HTTPError as exc:
        raise ValueError('The audio service could not be reached for word or beat timing.') from exc
    if response.is_error:
        raise ValueError(f'The audio service could not time the audio (HTTP {response.status_code}).')
    return response.json()


def fetch_words(audio):
    """[{'word','start','end'}] for an audio file, validated and sorted."""
    data = _service_post('/v1/words', audio, {'word_timestamps': True}, float(os.getenv('WORDS_TIMEOUT_SECONDS', '600')))
    words = []
    for w in (data.get('words') if isinstance(data, dict) else None) or []:
        if isinstance(w, dict) and isinstance(w.get('word'), str) and all(type(w.get(k)) in (int, float) for k in ('start', 'end')) and 0 <= w['start'] <= w['end']:
            words.append({'word': w['word'], 'start': float(w['start']), 'end': float(w['end'])})
    return sorted(words, key=lambda w: w['start'])


def fetch_beats(audio):
    data = _service_post('/v1/beats', audio, {}, float(os.getenv('BEATS_TIMEOUT_SECONDS', '300')))
    raw = (data.get('beats') if isinstance(data, dict) else None) or []
    beats = [float(b['time'] if isinstance(b, dict) else b) for b in raw if isinstance(b, (int, float)) or (isinstance(b, dict) and type(b.get('time')) in (int, float))]
    return sorted(b for b in beats if b >= 0)


def _cached(pid, kind, audio, fetch):
    """Word and beat timings per audio file, cached by content hash under the project."""
    directory = project_dir(pid) / 'analysis'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{kind}-{file_sha256(audio)[:24]}.json'
    if path.is_file():
        return json.loads(path.read_text())
    value = fetch(audio)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value))
    tmp.replace(path)
    return value


# ---------------------------------------------------------------- anchors

def collapse(word):
    """Lowercase letters/digits with repeated letters folded, so "Whaaat?!" matches "what"."""
    return re.sub(r'(.)\1+', r'\1', re.sub(r"[^a-z0-9]+", '', str(word).lower().replace('’', "'")))


def find_phrase(words, phrase):
    """Every (start, end) where the phrase's words occur consecutively."""
    target = [collapse(w) for w in str(phrase).split() if collapse(w)]
    if not target:
        return []
    tokens = [collapse(w['word']) for w in words]
    hits = []
    for i in range(len(tokens) - len(target) + 1):
        if tokens[i:i + len(target)] == target:
            hits.append((words[i]['start'], words[i + len(target) - 1]['end']))
    return hits


def find_pauses(samples, rate, min_seconds=0.6):
    """Quiet runs (start, end) longest first: 50 ms frames below max(-50 dBFS, peak-35 dB)."""
    import numpy as np
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    frame = max(1, int(rate * 0.05))
    n = len(samples) // frame
    if not n:
        return []
    rms = np.sqrt(np.mean(samples[:n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    level = 20 * np.log10(rms)
    threshold = max(-50.0, float(np.max(level)) - 35)
    quiet = level < threshold
    runs, start = [], None
    for i, q in enumerate(list(quiet) + [False]):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * frame / rate >= min_seconds:
                runs.append((start * frame / rate, i * frame / rate))
            start = None
    return runs


class Anchors:
    """Lazily fetches what the inserts need: word timings, beats or pauses."""

    def __init__(self, pid, source, rows, duration, features=None):
        self.pid, self.source, self.rows, self.duration = pid, source, rows, duration
        self.features = features if features is not None else service_features()
        self._words = self._beats = self._pauses = None

    def speech_audio(self):
        """A separated speech/vocals layer at offset 0 if there is one (cleaner ASR), else the project audio."""
        for row in self.rows:
            text = ' '.join(str(row.get(k) or '') for k in ('name', 'label', 'prompt'))
            if row.get('role') not in ('residual', 'sfx', 'effect') and row.get('start_offset', 0) == 0 and SPEECH_LAYER.search(text):
                return candidates.safe_path(self.pid, row['path'])
        return self.source

    def words(self):
        if self._words is None:
            if not self.features.get('words', {}).get('available'):
                raise ValueError("Word timing isn't available, so I can't find spoken words. Give a timestamp instead, like \"at 0:12\".")
            self._words = _cached(self.pid, 'words', self.speech_audio(), fetch_words)
        return self._words

    def beats(self):
        if self._beats is None:
            if not self.features.get('beats', {}).get('available'):
                raise ValueError('Beat detection is not available. Give a timestamp instead, like "at 0:12".')
            self._beats = _cached(self.pid, 'beats', self.source, fetch_beats)
        return self._beats

    def pauses(self):
        if self._pauses is None:
            from .media import read_wav
            path = self.speech_audio()
            wav = Path(path)
            if wav.suffix.lower() != '.wav':
                wav = project_dir(self.pid) / 'stems' / (uid() + '-pauses.wav')
                run('ffmpeg', '-v', 'error', '-y', '-i', str(path), '-ac', '1', '-c:a', 'pcm_f32le', str(wav))
            samples, rate = read_wav(wav)
            if wav != Path(path):
                wav.unlink(missing_ok=True)
            self._pauses = find_pauses(samples, rate)
        return self._pauses

    def resolve(self, anchor):
        """Anchor -> list of (time seconds, note) points in project time."""
        kind, value, occurrence = anchor['type'], anchor['value'], anchor.get('occurrence')
        if kind == 'time':
            if not 0 <= value <= self.duration:
                raise ValueError('That time is outside the clip.')
            return [value]
        if kind == 'start':
            return [0.0]
        if kind == 'end':
            return [self.duration]
        if kind == 'word':
            hits = find_phrase(self.words(), value)
            if not hits:
                raise ValueError(f"I couldn't find \"{value}\" in the speech. Give me a timestamp instead, like \"at 0:12\".")
            hits = _pick(hits, occurrence, f'"{value}" is said')
            return [(h[1] if anchor['relation'] == 'after' else h[0]) for h in hits]
        if kind == 'beat':
            beats = self.beats()
            if not 1 <= value <= len(beats):
                raise ValueError(f'The audio has {len(beats)} detected beats; choose beat 1 to {len(beats)} or give a timestamp.')
            return [beats[value - 1]]
        if kind == 'pause':
            pauses = self.pauses()
            if not pauses:
                raise ValueError("I couldn't find a clear pause. Give me a timestamp instead, like \"crickets at 0:12\".")
            if occurrence is None:
                chosen = max(pauses, key=lambda p: p[1] - p[0])
            else:
                ordered = sorted(pauses)
                if occurrence > len(ordered):
                    raise ValueError(f'I found only {len(ordered)} pause(s).')
                chosen = ordered[occurrence - 1] if occurrence > 0 else ordered[-1]
            return [min(chosen[0] + 0.15, chosen[1])]
        raise ValueError('Unsupported anchor.')


def _pick(hits, occurrence, what):
    if occurrence is None:
        return hits
    if occurrence == -1:
        return hits[-1:]
    if occurrence > len(hits):
        raise ValueError(f'{what} only {len(hits)} time(s).')
    return [hits[occurrence - 1]]


def placement_start(point, anchor, entry, offset):
    """Where the clip's first sample goes. 'at'/'after' start the sound at the point; 'before' lands
    the sound's hit (or its end) on the point; 'end' fits the whole sound before the clip ends."""
    if anchor['type'] == 'end':
        return point - entry['duration'] + offset
    if anchor['relation'] == 'before':
        return point - (entry['hit'] if entry.get('hit') is not None else entry['duration']) + offset
    return point + offset


# ---------------------------------------------------------------- request validation (route)

def validate_request(inserts, duration, features):
    """Route-side checks before queueing: sounds exist, anchors are possible and in range."""
    from .sfx_library import get
    from .sfx_plan import normalize_insert
    out = []
    for item in inserts:
        insert = normalize_insert(item, None, duration)
        if not insert.get('sfx_id'):
            from .sfx_library import resolve
            entry, _ = resolve(insert['sfx_query'])
            if not entry:
                raise ValueError(f"No library sound matches \"{insert['sfx_query']}\".")
            insert['sfx_id'] = entry['id']
        get(insert['sfx_id'])
        kind = insert['anchor']['type']
        if kind in ('word', 'beat'):
            key = 'words' if kind == 'word' else 'beats'
            if not features.get(key, {}).get('available'):
                raise ValueError('Word timing is not available; give a timestamp instead.' if kind == 'word' else 'Beat detection is not available; give a timestamp instead.')
        out.append(insert)
    return out


# ---------------------------------------------------------------- the candidate

def duck_layer(row, output, windows, rate, depth_db):
    """Write a copy of a layer with gain dips (40 ms in, 300 ms out) under each window."""
    import numpy as np
    from .media import read_wav, write_wav
    samples, sr = read_wav(row['path'])
    if sr != rate:
        samples_path = Path(output).with_name(uid() + '-rs.wav')
        run('ffmpeg', '-v', 'error', '-y', '-i', str(row['path']), '-ar', str(rate), '-c:a', 'pcm_f32le', str(samples_path))
        samples, sr = read_wav(samples_path)
        samples_path.unlink(missing_ok=True)
    t = row.get('start_offset', 0) + np.arange(len(samples)) / rate
    depth = 1 - 10 ** (-depth_db / 20)
    amount = np.zeros(len(samples))
    for start, end in windows:
        attack, release = 0.04, 0.3
        shape = np.clip(np.minimum((t - (start - attack)) / attack, 1 - (t - end) / release), 0, 1)
        amount = np.maximum(amount, shape)
    write_wav(output, samples * (1 - depth * amount)[:, None], rate)


def place(inserts, anchors):
    """Resolve every insert's anchor to placements (project time). Returns (placements, warnings)."""
    from .sfx_library import get
    warnings, placements = [], []
    for index, insert in enumerate(inserts):
        entry = get(insert['sfx_id'])
        points = anchors.resolve(insert['anchor'])
        for point in points:
            placements.append({'id': uid(), 'insert': index, 'sfx_id': entry['id'], 'title': entry['title'], 'anchor_time': round(point, 4),
                               'start': placement_start(point, insert['anchor'], entry, insert['offset']),
                               'gain_db': insert['gain_db'], 'duck': insert['duck']})
    if len(placements) > MAX_PLACEMENTS:
        warnings.append(f'Found {len(placements)} places; only the first {MAX_PLACEMENTS} were used.')
        placements = sorted(placements, key=lambda p: p['start'])[:MAX_PLACEMENTS]
    return placements, warnings


def render_layers(pid, placements, inserts, base, rate, channels, count, created, warnings, duck_db):
    """One new sfx layer per placement, plus ducked copies of the base layers where asked.
    Returns (new_rows, proposed) where proposed is the base layers (ducked if needed) followed by the new ones."""
    from .media import read_wav, write_wav
    from .sfx_library import asset_file
    directory = project_dir(pid) / 'stems'
    directory.mkdir(parents=True, exist_ok=True)
    rendered = {}
    new_rows = []
    solo = int(any(r['solo'] for r in base))
    titles = {}
    for placement in sorted(placements, key=lambda p: p['start']):
        key = (placement['sfx_id'], placement['gain_db'])
        if key not in rendered:
            path = directory / (uid() + '-sfx-src.wav')
            created.append(path)
            run('ffmpeg', '-v', 'error', '-y', '-i', str(asset_file(placement['sfx_id'])), '-ar', str(rate), '-ac', str(channels),
                '-af', f"volume={placement['gain_db']}dB", '-c:a', 'pcm_f32le', str(path))
            rendered[key] = read_wav(path)[0]
        clip = rendered[key]
        start = round(placement['start'] * rate)
        trim = max(0, -start)
        position = max(0, start)
        end = min(len(clip), trim + count - position)
        if position >= count or end <= trim:
            raise ValueError(f"{placement['title']} would start outside the clip ({placement['start']:.2f}s).")
        if trim:
            warnings.append(f"{placement['title']} started before the clip; its beginning was trimmed.")
        if end < len(clip):
            warnings.append(f"{placement['title']} runs past the end of the clip; its tail was trimmed.")
        path = directory / (uid() + '-sfx.wav')
        created.append(path)
        write_wav(path, clip[trim:end], rate)
        titles[placement['title']] = titles.get(placement['title'], 0) + 1
        n = titles[placement['title']]
        name = placement['title'][:100] if n == 1 else f"{placement['title'][:90]} {n}"
        placement.update(start=round(position / rate, 6), end=round((position + end - trim) / rate, 6), layer_id=placement['id'])
        new_rows.append({'id': placement['id'], 'project_id': pid, 'name': name, 'label': placement['title'][:100], 'path': str(path),
                         'duration': (end - trim) / rate, 'start_offset': position / rate, 'volume': 1, 'muted': 0, 'solo': solo,
                         'automation': '[]', 'mute_ranges': '[]', 'revision': 0, 'role': 'sfx',
                         'prompt': inserts[placement['insert']]['sfx_query'][:200],
                         'tags': json.dumps([{'label': 'SFX'}])})
    windows = [(p['start'], p['end']) for p in placements if p['duck']]
    proposed = []
    for row in base:
        row = dict(row)
        if windows and not row['muted']:
            path = directory / (uid() + '-ducked.wav')
            created.append(path)
            duck_layer({**row, 'path': str(candidates.safe_path(pid, row['path']))}, path, windows, rate, duck_db)
            row['path'] = str(path)
        proposed.append(row)
    proposed.extend(new_rows)
    return new_rows, proposed


def build(project, payload, cid):
    import numpy as np
    from .media import read_wav, write_wav
    from .sfx_library import asset_file, get
    pid = project['id']
    snapshot = payload['_candidate']
    source = candidates.safe_path(pid, snapshot['source'])
    if file_sha256(source) != snapshot['source_sha256']:
        raise ValueError('Source changed; generate another preview.')
    rows = [dict(r) for r in snapshot['rows']]
    for row in rows:
        if file_sha256(candidates.safe_path(pid, row['path'])) != snapshot['stem_hashes'][row['id']]:
            raise ValueError('A layer changed; generate another preview.')
    info = probe(source)
    rate, channels, duration = info['sample_rate'], info['channels'], info['duration']
    count = round(duration * rate)
    base = rows or [{'id': uid(), 'project_id': pid, 'name': 'Original audio', 'path': str(source), 'duration': duration,
                     'start_offset': 0, 'volume': 1, 'muted': 0, 'solo': 0, 'automation': '[]', 'mute_ranges': '[]', 'revision': 0}]
    anchors = Anchors(pid, source, rows, duration, payload.get('_features'))
    placements, warnings = place(payload['inserts'], anchors)
    directory = project_dir(pid) / 'stems'
    directory.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        new_rows, proposed = render_layers(pid, placements, payload['inserts'], base, rate, channels, count, created, warnings,
                                           float(payload.get('duck_db', 6.0)))
        target, mix, original = (directory / (uid() + '.wav') for _ in range(3))
        created.extend([target, mix, original])
        export(new_rows, source, target, 'wav', False)
        export(proposed, source, mix, 'wav', False)
        if rows:
            export(rows, source, original, 'wav', False)
        else:
            run('ffmpeg', '-v', 'error', '-y', '-i', str(source), '-c:a', 'pcm_f32le', str(original))
        peak = float(np.max(np.abs(read_wav(mix)[0]))) if count else 0.0
        if peak > 1.0:
            warnings.append(f'The mix peaks at {20 * np.log10(peak):+.1f} dBFS and may clip; lower the sound (e.g. "quieter") or turn on ducking.')
        for path in [p for p in created if p.name.endswith('-sfx-src.wav')]:
            path.unlink(missing_ok=True)  # per-placement clips hold the audio now
        notes = ['Inserted sounds are new layers from the TidePool CC0 sound pack; Undo removes them.']
        if any(i['anchor']['type'] == 'word' for i in payload['inserts']):
            notes.append('Word times come from speech recognition; check each placement.')
        if any(i['anchor']['type'] == 'pause' for i in payload['inserts']):
            notes.append('The pause was found from loudness alone; check it is the moment you meant.')
        licences = {}
        for p in placements:
            e = get(p['sfx_id'])
            licences[e['id']] = {'id': e['id'], 'title': e['title'], 'licence': e['licence'], 'author': e['author'], 'source_url': e['source_url']}
        return {'target': {'path': str(target)}, 'mix': {'path': str(mix)}, 'original': {'path': str(original)}, 'snapshot': snapshot, 'proposed': proposed,
                'inference': {'model': {'backend': 'sfx-insert', 'is_mock': False}, 'warnings': notes + warnings},
                'provenance': {'kind': 'sfx_insert', 'request': payload.get('request'), 'inserts': payload['inserts'],
                               'placements': [{k: p[k] for k in ('id', 'sfx_id', 'title', 'anchor_time', 'start', 'end', 'gain_db', 'duck')} for p in sorted(placements, key=lambda p: p['start'])],
                               'licences': list(licences.values()), 'stem_revisions': candidates.revisions(rows), 'source_sha256': snapshot['source_sha256']}}
    except Exception:
        for path in created:
            Path(path).unlink(missing_ok=True)
        raise
