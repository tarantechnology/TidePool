"""The TidePool SFX pack: manifest, on-disk assets and text -> sound search.

The manifest (app/sfx_manifest.json) is in git. The audio is not: it is built into
``$STORAGE_ROOT/sfx`` by ``scripts/build_sfx_pack.py``, and any procedural sound that is
missing is synthesized on first use, so a fresh checkout works without the build step.

Search is offline: aliases, tags and fuzzy matching. ``clap_search`` is the hook for the
audio service's CLAP text->audio retrieval (``/v1/sfx/search``); it is used only when the
service advertises ``sfx_search.available`` and silently falls back otherwise.
"""
import difflib
import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

MANIFEST = Path(__file__).with_name('sfx_manifest.json')
ALLOWED_LICENCES = ('CC0-1.0',)
CONFIDENT = 0.72
_STOP = {'a', 'an', 'the', 'some', 'sound', 'sounds', 'effect', 'effects', 'sfx', 'noise', 'noises', 'of', 'that', 'this', 'one', 'quick', 'little'}


@lru_cache(maxsize=1)
def manifest():
    data = json.loads(MANIFEST.read_text())
    return {'pack': data['pack'], 'version': data['version'], 'sounds': {s['id']: s for s in data['sounds']}}


def sounds():
    return list(manifest()['sounds'].values())


def get(sid):
    entry = manifest()['sounds'].get(sid) if isinstance(sid, str) else None
    if not entry:
        raise ValueError('Unknown sound effect.')
    return entry


def public(entry):
    """What the editor and API show: no file paths."""
    return {k: entry[k] for k in ('id', 'title', 'category', 'tags', 'aliases', 'duration', 'licence', 'author', 'source', 'source_url')}


def pack_dir():
    configured = os.getenv('SOUNDCANVAS_SFX_PACK_DIR', '').strip()
    if configured:
        return Path(configured)
    from . import core
    return core.ROOT / 'sfx'


def asset_file(sid):
    """Path to the sound's FLAC, synthesizing a missing procedural sound on demand."""
    entry = get(sid)
    path = pack_dir() / entry['file']
    if path.is_file():
        return path
    if entry['source'] != 'procedural':
        raise ValueError(f"The sound '{entry['title']}' is not downloaded. Run scripts/build_sfx_pack.py.")
    write_procedural(sid, path)
    return path


def write_procedural(sid, path):
    """Synthesize and encode one sound as 16-bit FLAC, atomically."""
    from .core import run
    from .media import write_wav
    from .sfx_synth import synthesize
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = synthesize(sid)
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        wav, flac = Path(tmp) / 'x.wav', Path(tmp) / 'x.flac'
        write_wav(wav, samples, 48000)
        run('ffmpeg', '-v', 'error', '-y', '-i', str(wav), '-c:a', 'flac', '-sample_fmt', 's16', '-compression_level', '8', str(flac))
        flac.replace(path)
    return samples


# ---------------------------------------------------------------- search

# Words that may sit around a sound's name without changing which sound is meant.
MODIFIERS = {'loud', 'louder', 'quiet', 'quieter', 'soft', 'softer', 'big', 'huge', 'massive', 'small', 'short', 'long',
             'deep', 'epic', 'funny', 'classic', 'famous', 'iconic', 'cartoon', 'cartoony', 'random', 'good', 'nice',
             'single', 'another', 'extra', 'meme', 'subtle', 'dramatic', 'comedic', 'goofy', 'silly', 'please'}


def _stem(token):
    if len(token) > 5 and token.endswith('ing'):
        token = token[:-3]
        if len(token) > 2 and token[-1] == token[-2] and token[-1] not in 'lsoe':
            token = token[:-1]  # clapping -> clap, but keep "bell"/"boo"
        return token
    return token[:-1] if len(token) > 3 and token.endswith('s') and not token.endswith('ss') else token


def tokens(text):
    text = re.sub(r"[^a-z0-9 ]+", ' ', str(text).lower().replace('-', ' ').replace('_', ' ').replace("'", ''))
    return [_stem(t) for t in text.split() if t not in _STOP]


def normal(text):
    return ' '.join(tokens(text))


@lru_cache(maxsize=1)
def _index():
    rows = []
    for entry in sounds():
        names = {normal(n) for n in [entry['title'], entry['id'], *entry['aliases']]} - {''}
        tags = {t for tag in entry['tags'] for t in tokens(tag)}
        rows.append((entry, names, tags))
    return rows


def score(query, names, tags):
    q = normal(query)
    if not q:
        return 0.0
    qt = set(q.split())
    best = 0.0
    for name in names:
        if q == name:
            return 1.0
        nt = set(name.split())
        ratio = difflib.SequenceMatcher(None, q, name).ratio()
        best = max(best, ratio if ratio >= 0.85 else 0.7 * ratio)  # typos count, look-alikes do not
        if nt and nt <= qt and qt - nt <= MODIFIERS:  # "a loud vine boom please" contains "vine boom"
            best = max(best, 0.85 + 0.14 * len(nt) / len(qt))  # only an exact name scores 1.0
        overlap = len(qt & nt) / len(qt | nt)
        best = max(best, 0.9 * overlap)
    if tags and qt:
        best = max(best, 0.7 * len(qt & tags) / len(qt))
    return round(best, 4)


def search(query, limit=5):
    """Ranked [{'id','title','score'}] by offline text matching (deterministic)."""
    ranked = sorted(((score(query, names, tags), entry['id'], entry) for entry, names, tags in _index()),
                    key=lambda r: (-r[0], r[1]))
    return [{'id': e['id'], 'title': e['title'], 'score': s} for s, _, e in ranked[:limit] if s > 0]


def resolve(query):
    """Best confident match for a text query, or None with suggestions.

    Returns (entry or None, suggestions [titles])."""
    hits = search(query, 5)
    remote = clap_search(query, 3)
    if hits and hits[0]['score'] >= 0.9:
        return get(hits[0]['id']), []
    if remote:
        return get(remote[0]['id']), []
    if hits and hits[0]['score'] >= CONFIDENT:
        return get(hits[0]['id']), []
    return None, [h['title'] for h in hits[:3] if h['score'] >= 0.4]


def clap_search(query, limit=3):
    """CLAP text->audio retrieval in the audio service, when it offers one.

    Contract (POST {service}/v1/sfx/search, JSON): {"query": str, "ids": [manifest ids], "top_k": int}
    -> {"results": [{"id": str, "score": float}]} where score is cosine similarity in [-1, 1].
    The service embeds the pack once (same manifest); results outside the manifest are dropped.
    Never raises: offline text search is the fallback."""
    if os.getenv('INFERENCE_ADAPTER', 'fake') != 'http':
        return []
    try:
        from .sfx_insert import service_features
        if not service_features().get('sfx_search', {}).get('available'):
            return []
        import httpx
        from urllib.parse import urlsplit, urlunsplit
        url = urlsplit(os.getenv('SAM_AUDIO_SERVICE_URL', ''))
        response = httpx.post(urlunsplit((url.scheme, url.netloc, '/v1/sfx/search', '', '')),
                              json={'query': str(query)[:200], 'ids': list(manifest()['sounds']), 'top_k': limit}, timeout=10)
        response.raise_for_status()
        known = manifest()['sounds']
        results = [r for r in response.json().get('results', []) if isinstance(r, dict) and r.get('id') in known
                   and isinstance(r.get('score'), (int, float)) and r['score'] >= float(os.getenv('SFX_CLAP_MIN_SCORE', '0.25'))]
        return sorted(results, key=lambda r: -r['score'])[:limit]
    except Exception:  # noqa: BLE001 - optional accelerator; text search still answers
        return []
