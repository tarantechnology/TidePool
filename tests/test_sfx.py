"""SFX pack, search, anchors, insert candidates and the `insert` planner kind."""
import json
import subprocess
import time

import numpy as np
import pytest


@pytest.fixture(scope='session')
def pack_dir(tmp_path_factory):
    return tmp_path_factory.mktemp('sfx-pack')


@pytest.fixture
def client(tmp_path, monkeypatch, pack_dir):
    import app.core as core
    core.ROOT = tmp_path / 'storage'; core.ROOT.mkdir(); core.DB = core.ROOT / 'db.sqlite'
    monkeypatch.setenv('STORAGE_ROOT', str(core.ROOT)); monkeypatch.setenv('INFERENCE_ADAPTER', 'fake')
    monkeypatch.setenv('SOUNDCANVAS_SFX_PACK_DIR', str(pack_dir))
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c: yield c


def project(client, tmp_path, seconds=6.0, gap=None):
    """A project whose audio is a tone, optionally with a silent gap (start, end)."""
    p = client.post('/api/projects').json(); pid = p['project_id']; h = {'Authorization': 'Bearer ' + p['access_token']}
    from app.media import write_wav
    audio = tmp_path / 'audio.wav'
    t = np.arange(int(48000 * seconds)) / 48000
    tone = 0.3 * np.sin(2 * np.pi * 330 * t)
    if gap: tone[(t >= gap[0]) & (t < gap[1])] = 0
    write_wav(audio, np.stack([tone, tone], axis=1), 48000)
    with audio.open('rb') as f:
        assert client.post(f'/api/projects/{pid}/upload', headers=h, files={'file': ('audio.wav', f)}).status_code == 200
    return pid, h


def wait(client, cid, h):
    for _ in range(600):
        job = client.get('/api/candidates/' + cid, headers=h).json()
        if job['status'] in ('completed', 'failed'): return job
        time.sleep(.05)
    pytest.fail('timed out')


def insert(sfx_id, anchor_type='time', value=None, relation='at', occurrence=None, offset=0, gain_db=0, duck=False):
    return {'sfx_query': sfx_id.replace('_', ' '), 'sfx_id': sfx_id, 'anchor': {'type': anchor_type, 'value': value, 'relation': relation, 'occurrence': occurrence},
            'offset': offset, 'gain_db': gain_db, 'duck': duck}


# ---------------------------------------------------------------- pack and manifest

def test_manifest_is_valid_and_every_sound_is_cc0_with_provenance():
    from app.sfx_library import ALLOWED_LICENCES, MANIFEST, sounds
    from app.sfx_synth import RECIPES
    data = json.loads(MANIFEST.read_text())
    entries = data['sounds']
    ids = [s['id'] for s in entries]
    assert len(ids) == len(set(ids)) and 60 <= len(ids) <= 100
    assert set(ids) == set(RECIPES)  # every recipe is listed, nothing unlisted ships
    required = {'id', 'title', 'category', 'tags', 'aliases', 'licence', 'source', 'source_url', 'author', 'file', 'hit',
                'duration', 'channels', 'sample_rate', 'loudness_lufs', 'true_peak_dbtp'}
    for s in entries:
        assert required <= set(s), s['id']
        assert s['licence'] in ALLOWED_LICENCES and s['author'] and s['source_url'].startswith('https://')
        assert s['source'] in ('procedural', 'download') and s['file'] == s['id'] + '.flac'
        assert 0.05 < s['duration'] <= 15 and s['sample_rate'] == 48000 and s['channels'] in (1, 2)
        assert s['true_peak_dbtp'] <= -0.95 and -40 < s['loudness_lufs'] < -10
        assert s['aliases'] and s['tags'] and (s['hit'] is None or 0 <= s['hit'] <= s['duration'])
        if s['source'] == 'download': assert s.get('url') and len(s.get('sha256', '')) == 64
    must_have = {'vine_boom', 'bruh', 'record_scratch', 'sad_trombone', 'rimshot', 'laugh_track', 'applause', 'crickets', 'airhorn',
                 'whoosh', 'riser', 'boom_impact', 'ding', 'cash_register', 'wrong_buzzer', 'correct_chime', 'dramatic_sting',
                 'suspense_drone', 'boing', 'slide_whistle_up', 'cartoon_run', 'glass_shatter', 'gasp', 'oof', 'drumroll', 'tada',
                 'typing', 'camera_shutter', 'record_rewind'}
    assert must_have <= set(ids)
    assert len(sounds()) == len(ids)


def test_pack_script_check_passes():
    import sys
    result = subprocess.run([sys.executable, 'scripts/build_sfx_pack.py', '--check'], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_sound_synthesizes_peak_safe_and_matches_the_manifest():
    from app.sfx_dsp import loudness_lufs, true_peak
    from app.sfx_library import sounds
    from app.sfx_synth import synthesize
    total = 0
    for entry in sounds():
        x = synthesize(entry['id'])
        assert x.dtype == np.float32 and x.ndim == 2 and x.shape[1] == entry['channels'], entry['id']
        assert np.isfinite(x).all(), entry['id']
        assert true_peak(x) <= 10 ** (-1 / 20) + 1e-3, entry['id']
        assert abs(len(x) / 48000 - entry['duration']) < 0.005, entry['id']
        assert abs(loudness_lufs(x) - entry['loudness_lufs']) < 0.6, entry['id']
        total += x.size * 2  # 16-bit
    assert total < 32e6  # raw 16-bit PCM; the FLAC pack is about 10 MB, well under the 25 MB budget


def test_synthesis_is_deterministic():
    from app.sfx_synth import synthesize
    for sid in ('vine_boom', 'laugh_track', 'record_scratch', 'applause', 'crickets'):
        assert np.array_equal(synthesize(sid), synthesize(sid))


def test_asset_file_synthesizes_missing_flac_once(tmp_path, monkeypatch):
    from app import sfx_library
    from app.core import probe
    monkeypatch.setenv('SOUNDCANVAS_SFX_PACK_DIR', str(tmp_path))
    path = sfx_library.asset_file('ding')
    assert path == tmp_path / 'ding.flac' and path.stat().st_size > 1000
    info = probe(path)
    assert info['sample_rate'] == 48000 and abs(info['duration'] - sfx_library.get('ding')['duration']) < 0.01
    mtime = path.stat().st_mtime_ns
    assert sfx_library.asset_file('ding').stat().st_mtime_ns == mtime
    with pytest.raises(ValueError): sfx_library.asset_file('not_a_sound')


def test_loudness_meter_matches_reference_tone():
    from app.sfx_dsp import loudness_lufs
    t = np.arange(48000 * 3) / 48000
    tone = 10 ** (-20 / 20) * np.sin(2 * np.pi * 1000 * t)  # 1 kHz at -20 dBFS: -23.0 LUFS mono (BS.1770)
    assert abs(loudness_lufs(tone) - -23.0) < 0.2
    assert loudness_lufs(np.zeros(4800)) == -120.0


# ---------------------------------------------------------------- search

@pytest.mark.parametrize('query,expected', [
    ('vine boom', 'vine_boom'), ('a loud vine boom', 'vine_boom'), ('vine bom', 'vine_boom'), ('record scratch', 'record_scratch'),
    ('sad trombone', 'sad_trombone'), ('womp womp', 'sad_trombone'), ('ba dum tss', 'rimshot'), ('laughter', 'laugh_track'),
    ('laughing', 'laugh_track'), ('clapping', 'applause'), ('crickets', 'crickets'), ('air horn', 'airhorn'), ('ka-ching', 'cash_register'),
    ('dun dun dun', 'dramatic_sting'), ('glass breaking', 'glass_shatter'), ('pew pew', 'laser'), ('drum roll', 'drumroll'),
    ('bruh', 'bruh'), ('boing', 'boing'), ('rewind', 'record_rewind'),
])
def test_search_finds_the_meant_sound(query, expected):
    from app.sfx_library import resolve, search
    entry, _ = resolve(query)
    assert entry and entry['id'] == expected
    assert search(query, 1)[0]['id'] == expected


@pytest.mark.parametrize('query', ['quack', 'reverb', 'bass boost', 'dog bark', 'windows xp startup', 'car crash'])
def test_search_does_not_guess_unknown_sounds(query):
    from app.sfx_library import resolve
    entry, suggestions = resolve(query)
    assert entry is None and isinstance(suggestions, list)


def test_clap_hook_is_inert_without_the_service(monkeypatch):
    from app.sfx_library import clap_search
    monkeypatch.setenv('INFERENCE_ADAPTER', 'fake')
    assert clap_search('vine boom') == []


def test_clap_hook_uses_service_results_when_advertised(monkeypatch):
    import httpx
    from app import sfx_insert, sfx_library
    monkeypatch.setenv('INFERENCE_ADAPTER', 'http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL', 'http://svc.test/v1/separate')
    monkeypatch.setattr(sfx_insert, 'service_features', lambda: {'sfx_search': {'available': True}})
    seen = {}
    def post(url, json=None, timeout=None):
        seen.update(url=url, body=json)
        return httpx.Response(200, request=httpx.Request('POST', url), json={'results': [{'id': 'crickets', 'score': 0.61}, {'id': 'nope', 'score': 0.9}]})
    monkeypatch.setattr(httpx, 'post', post)
    assert sfx_library.clap_search('silence after a bad joke') == [{'id': 'crickets', 'score': 0.61}]
    assert seen['url'] == 'http://svc.test/v1/sfx/search' and 'crickets' in seen['body']['ids']
    entry, _ = sfx_library.resolve('silence after a bad joke')
    assert entry['id'] == 'crickets'


# ---------------------------------------------------------------- anchors

WORDS = [{'word': 'So', 'start': 0.5, 'end': 0.7}, {'word': 'what?', 'start': 1.0, 'end': 1.3}, {'word': 'no', 'start': 2.0, 'end': 2.2},
         {'word': 'way', 'start': 2.25, 'end': 2.5}, {'word': 'WHAAAT', 'start': 4.0, 'end': 4.4}]


def test_find_phrase_matches_words_and_phrases_ignoring_case_punctuation_and_stretching():
    from app.sfx_insert import collapse, find_phrase
    assert collapse('Whaaat?!') == collapse('what')
    assert find_phrase(WORDS, 'what') == [(1.0, 1.3), (4.0, 4.4)]
    assert find_phrase(WORDS, 'no way') == [(2.0, 2.5)]
    assert find_phrase(WORDS, 'nope') == []


def anchors_with(words=None, beats=None, features=None):
    from app.sfx_insert import Anchors
    a = Anchors('0' * 32, None, [], 6.0, features or {'words': {'available': True}, 'beats': {'available': True}})
    a._words, a._beats = words, beats
    return a


def test_anchor_resolution_for_words_occurrences_and_relations():
    a = anchors_with(WORDS)
    word = lambda value, relation='at', occurrence=None: {'type': 'word', 'value': value, 'relation': relation, 'occurrence': occurrence}
    assert a.resolve(word('what')) == [1.0, 4.0]
    assert a.resolve(word('what', occurrence=1)) == [1.0]
    assert a.resolve(word('what', occurrence=-1)) == [4.0]
    assert a.resolve(word('what', 'after', 1)) == [1.3]
    assert a.resolve(word('no way', 'after')) == [2.5]
    with pytest.raises(ValueError, match='couldn'): a.resolve(word('banana'))
    with pytest.raises(ValueError, match='only 2'): a.resolve(word('what', occurrence=3))


def test_anchor_resolution_for_time_start_end_beats_and_unavailable_services():
    a = anchors_with(WORDS, [0.5, 1.0, 1.5, 2.0])
    assert a.resolve({'type': 'time', 'value': 2.5, 'relation': 'at', 'occurrence': None}) == [2.5]
    assert a.resolve({'type': 'start', 'value': None, 'relation': 'at', 'occurrence': None}) == [0.0]
    assert a.resolve({'type': 'end', 'value': None, 'relation': 'at', 'occurrence': None}) == [6.0]
    assert a.resolve({'type': 'beat', 'value': 3, 'relation': 'at', 'occurrence': None}) == [1.5]
    with pytest.raises(ValueError, match='beats'): a.resolve({'type': 'beat', 'value': 9, 'relation': 'at', 'occurrence': None})
    off = anchors_with(features={'words': {'available': False}, 'beats': {'available': False}})
    with pytest.raises(ValueError, match='timestamp'): off.resolve({'type': 'word', 'value': 'what', 'relation': 'at', 'occurrence': None})
    with pytest.raises(ValueError, match='timestamp'): off.resolve({'type': 'beat', 'value': 1, 'relation': 'at', 'occurrence': None})


def test_placement_start_for_before_after_and_end():
    from app.sfx_insert import placement_start
    entry = {'duration': 2.0, 'hit': None}
    at = {'type': 'time', 'relation': 'at'}
    assert placement_start(10, at, entry, 0) == 10 and placement_start(10, at, entry, -0.2) == pytest.approx(9.8)
    assert placement_start(10, {'type': 'time', 'relation': 'before'}, entry, 0) == 8.0
    assert placement_start(10, {'type': 'time', 'relation': 'before'}, {'duration': 2.0, 'hit': 0.38}, 0) == pytest.approx(9.62)
    assert placement_start(6.0, {'type': 'end', 'relation': 'at'}, entry, 0) == 4.0


def test_find_pauses_picks_the_quiet_gap():
    from app.sfx_insert import find_pauses
    rate = 48000
    t = np.arange(rate * 6) / rate
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * ~((t >= 2.0) & (t < 3.5))
    pauses = find_pauses(x, rate)
    assert len(pauses) == 1 and abs(pauses[0][0] - 2.0) < 0.06 and abs(pauses[0][1] - 3.5) < 0.06
    assert find_pauses(0.3 * np.sin(2 * np.pi * 220 * t), rate) == []


def test_words_client_validates_service_output(monkeypatch):
    from app import sfx_insert
    monkeypatch.setattr(sfx_insert, '_service_post', lambda *a, **k: {'words': [{'word': 'b', 'start': 2, 'end': 2.5}, {'word': 'a', 'start': 1.0, 'end': 1.2},
                                                                            {'word': 'bad', 'start': 'x', 'end': 1}, {'word': 'neg', 'start': 3, 'end': 2}]})
    assert sfx_insert.fetch_words('x.wav') == [{'word': 'a', 'start': 1.0, 'end': 1.2}, {'word': 'b', 'start': 2.0, 'end': 2.5}]
    monkeypatch.setattr(sfx_insert, '_service_post', lambda *a, **k: {'beats': [0.5, {'time': 1.0}, 'x', -1]})
    assert sfx_insert.fetch_beats('x.wav') == [0.5, 1.0]


# ---------------------------------------------------------------- insert candidates

def test_insert_preview_apply_and_undo_at_a_time(client, tmp_path):
    pid, h = project(client, tmp_path)
    url = f'/api/projects/{pid}/sfx-preview'
    body = {'inserts': [insert('vine_boom', value=1.5), insert('rimshot', 'end', gain_db=-6)], 'request': 'vine boom at 0:01.5 and a rimshot at the end'}
    assert client.post(url, json=body).status_code == 401
    res = client.post(url, headers=h, json=body); assert res.status_code == 202, res.text
    cid = res.json()['candidate_id']
    job = wait(client, cid, h); assert job['status'] == 'completed', job
    prov = job['provenance']
    assert prov['kind'] == 'sfx_insert' and [p['sfx_id'] for p in prov['placements']] == ['vine_boom', 'rimshot']
    assert prov['placements'][0]['start'] == pytest.approx(1.5, abs=1e-4)
    from app.sfx_library import get
    assert prov['placements'][1]['start'] == pytest.approx(6.0 - get('rimshot')['duration'], abs=2e-3)
    assert {l['licence'] for l in prov['licences']} == {'CC0-1.0'}
    for part in ('', '?part=mix', '?part=original'):
        assert client.get(f'/api/candidates/{cid}/audio{part}', headers=h).status_code == 200
    assert client.post(f'/api/candidates/{cid}/apply', headers=h).status_code == 200
    stems = client.get(f'/api/projects/{pid}/stems', headers=h).json()
    sfx = sorted([s for s in stems if s['role'] == 'sfx'], key=lambda s: s['start_offset'])
    assert [s['name'] for s in sfx] == [get('vine_boom')['title'], get('rimshot')['title']]
    assert sfx[0]['start_offset'] == pytest.approx(1.5, abs=1e-4) and sfx[0]['tags'] == [{'label': 'SFX'}]
    assert any(s['name'] == 'Original audio' for s in stems)
    assert client.post(f'/api/candidates/{cid}/undo', headers=h).status_code == 200
    assert client.get(f'/api/projects/{pid}/stems', headers=h).json() == []


def test_insert_at_a_spoken_word_lands_on_the_word_and_ducks(client, tmp_path, monkeypatch):
    from app import sfx_insert
    from app.media import read_wav
    pid, h = project(client, tmp_path)
    features = {'words': {'available': True}, 'beats': {'available': False}, 'sfx_search': {'available': False}}
    monkeypatch.setattr(sfx_insert, 'service_features', lambda: features)
    calls = []
    def fake_words(audio):  # stands in for the audio service's /v1/words
        calls.append(audio); return WORDS
    monkeypatch.setattr(sfx_insert, 'fetch_words', fake_words)
    body = {'inserts': [insert('vine_boom', 'word', 'what', duck=True)]}
    job = wait(client, client.post(f'/api/projects/{pid}/sfx-preview', headers=h, json=body).json()['candidate_id'], h)
    assert job['status'] == 'completed', job
    starts = [p['start'] for p in job['provenance']['placements']]
    assert starts == pytest.approx([1.0, 4.0], abs=0.002)  # within 2 ms of each "what"
    assert len(calls) == 1
    import app.core as core
    output = json.loads(core.connect().execute('SELECT output FROM jobs WHERE id=?', (job['id'],)).fetchone()['output'])
    original = next(r for r in output['proposed'] if r['name'] == 'Original audio')
    ducked, rate = read_wav(original['path'])
    level = lambda a, b: float(np.sqrt(np.mean(ducked[int(a * rate):int(b * rate)] ** 2)))
    assert level(1.1, 1.3) < 0.6 * level(0.2, 0.5)  # about -6 dB under the boom
    # A second preview reuses the cached word timings.
    wait(client, client.post(f'/api/projects/{pid}/sfx-preview', headers=h, json=body).json()['candidate_id'], h)
    assert len(calls) == 1


def test_insert_in_the_awkward_silence(client, tmp_path):
    pid, h = project(client, tmp_path, 6.0, (2.0, 4.0))
    job = wait(client, client.post(f'/api/projects/{pid}/sfx-preview', headers=h, json={'inserts': [insert('crickets', 'pause')]}).json()['candidate_id'], h)
    assert job['status'] == 'completed', job
    assert job['provenance']['placements'][0]['start'] == pytest.approx(2.15, abs=0.06)


def test_insert_preview_rejects_bad_requests(client, tmp_path):
    pid, h = project(client, tmp_path)
    url = f'/api/projects/{pid}/sfx-preview'
    assert client.post(url, headers=h, json={'inserts': [insert('vine_boom', 'word', 'what')]}).status_code == 422  # no word timing
    assert client.post(url, headers=h, json={'inserts': [insert('vine_boom', 'beat', 1)]}).status_code == 422
    assert client.post(url, headers=h, json={'inserts': [{**insert('vine_boom', value=1), 'sfx_id': 'nope'}]}).status_code == 422
    assert client.post(url, headers=h, json={'inserts': [insert('vine_boom', value=60)]}).status_code == 422
    assert client.post(url, headers=h, json={'inserts': [insert('vine_boom', value=1)] * 11}).status_code == 422
    assert client.post(url, headers=h, json={'inserts': []}).status_code == 422


def test_sfx_catalog_search_and_audio_routes(client):
    catalog = client.get('/api/sfx').json()
    assert len(catalog['sounds']) >= 60 and all('file' not in s for s in catalog['sounds'])
    assert client.get('/api/sfx/search', params={'q': 'vine boom'}).json()['results'][0]['id'] == 'vine_boom'
    audio = client.get('/api/sfx/ding/audio')
    assert audio.status_code == 200 and audio.headers['content-type'] == 'audio/flac'
    assert client.get('/api/sfx/../../etc/passwd/audio').status_code == 404


def test_capabilities_report_sfx_library_and_speech_features(client, tmp_path):
    pid, h = project(client, tmp_path)
    caps = client.get(f'/api/projects/{pid}/capabilities', headers=h).json()
    assert caps['sfx_library']['available'] and caps['words'] == {'available': False} and caps['beats'] == {'available': False}


# ---------------------------------------------------------------- planner

def test_agent_route_plans_inserts_offline_and_clarifies_honestly(client, tmp_path, monkeypatch):
    from app import sfx_insert
    pid, h = project(client, tmp_path, 20.0)
    ask = lambda prompt: client.post(f'/api/projects/{pid}/agent', headers=h, json={'prompt': prompt, 'stems': [], 'tracks': [], 'duration': 20}).json()
    plan = ask('record scratch at 0:12')
    assert plan['kind'] == 'insert' and plan['inserts'][0]['sfx_id'] == 'record_scratch' and plan['inserts'][0]['anchor']['value'] == 12
    assert 'Record scratch at 0:12' in plan['message']
    plan = ask("add a vine boom when he says 'what'")
    assert plan['kind'] == 'clarify' and plan['sfx_needs_time'] is True  # no word timing in the fake adapter
    monkeypatch.setattr(sfx_insert, 'service_features', lambda: {'words': {'available': True}, 'beats': {'available': False}})
    plan = ask("add a vine boom when he says 'what'")
    assert plan['kind'] == 'insert' and plan['inserts'][0]['anchor'] == {'type': 'word', 'value': 'what', 'relation': 'at', 'occurrence': None}
    assert ask('add a quack at 3 seconds')['sfx_unmatched'] is True
    assert ask('laugh track after the joke')['kind'] == 'clarify'
    assert ask('airhorn at 0:30')['kind'] == 'clarify'  # past the end


def test_agent_route_keeps_inserts_and_effects_ahead_of_the_stale_editor_guard(client, tmp_path, monkeypatch):
    import app.main as main
    from app import sfx_insert
    monkeypatch.setattr(sfx_insert, 'service_features', lambda: {'words': {'available': True}, 'beats': {'available': False}})
    planned = []
    monkeypatch.setattr(main, 'agent_plan', lambda body: planned.append(body.prompt) or {'kind': 'fx', 'description': None, 'remove': False, 'actions': [], 'message': 'fx'})
    pid, h = project(client, tmp_path, 20.0)
    ask = lambda prompt: client.post(f'/api/projects/{pid}/agent', headers=h, json={'prompt': prompt, 'stems': [], 'tracks': [], 'duration': 20}).json()
    refresh = 'Refresh the editor to enable this sound-generation request, then submit it again.'
    plan = ask('add a vine boom at 12 seconds')
    assert plan['kind'] == 'insert' and plan['inserts'][0]['anchor']['value'] == 12
    plan = ask('add an airhorn after the word hello')
    assert plan['kind'] == 'insert' and plan['inserts'][0]['anchor']['type'] == 'word' and plan['inserts'][0]['anchor']['value'] == 'hello'
    plan = ask('add reverb to the vocals at 12 seconds')
    assert plan['kind'] == 'fx' and planned == ['add reverb to the vocals at 12 seconds']
    plan = ask('Add a gong every time we see person 5')  # main's object-appearance generation flow
    assert plan['kind'] == 'clarify' and plan['message'] == refresh
    assert ask('play a whoosh whenever car 2 appears')['message'] == refresh
    assert planned == ['add reverb to the vocals at 12 seconds']


def test_strict_schema_requires_every_property():
    from app.agent import SCHEMA
    def walk(node):
        if isinstance(node, dict):
            if node.get('type') == 'object':
                assert node['additionalProperties'] is False and set(node['required']) == set(node['properties'])
            for value in node.values(): walk(value)
        elif isinstance(node, list):
            for value in node: walk(value)
    walk(SCHEMA)
    assert 'insert' in SCHEMA['properties']['kind']['enum'] and 'inserts' in SCHEMA['required']


def model_plan(monkeypatch, output):
    import app.agent as agent
    monkeypatch.setenv('OPENAI_API_KEY', 'not-a-real-key')
    class Response:
        is_error = False
        def json(self): return {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(output)}]}]}
    captured = {}
    def post(url, **kwargs):
        captured.update(kwargs['json']); return Response()
    monkeypatch.setattr(agent.httpx, 'post', post)
    return captured


def llm_insert(**overrides):
    base = {'kind': 'insert', 'description': None, 'remove': False, 'actions': [], 'message': '', 'guidance': {'anchors': []},
            'edit_scope': {'action': 'keep', 'interval': None}, 'processing_time': None, 'separation_mode': None, 'stems': None, 'use_visual': False,
            'inserts': [{'sfx_query': 'meme boom', 'anchor': {'type': 'time', 'value': 12, 'relation': 'at', 'occurrence': None}, 'offset': None, 'gain_db': None, 'duck': None}]}
    base.update(overrides)
    return base


def test_llm_insert_plan_is_validated_and_resolved(client, tmp_path, monkeypatch):
    pid, h = project(client, tmp_path, 20.0)
    body = {'prompt': 'the 0:12 moment needs that famous internet boom', 'stems': [], 'tracks': [], 'duration': 20}
    captured = model_plan(monkeypatch, llm_insert())
    plan = client.post(f'/api/projects/{pid}/agent', headers=h, json=body).json()
    assert plan['kind'] == 'insert' and plan['inserts'][0]['sfx_id'] == 'vine_boom' and plan['inserts'][0]['gain_db'] == 0
    assert 'kind=insert' in captured['instructions'] and 'Record scratch' in captured['instructions']
    assert captured['text']['format']['schema']['properties']['inserts']


@pytest.mark.parametrize('bad', [
    {'inserts': [{'sfx_query': 'boom', 'anchor': {'type': 'time', 'value': 7, 'relation': 'at', 'occurrence': None}, 'offset': None, 'gain_db': None, 'duck': None}]},  # invented time
    {'inserts': [{'sfx_query': 'boom', 'anchor': {'type': 'word', 'value': 'hello', 'relation': 'at', 'occurrence': None}, 'offset': None, 'gain_db': None, 'duck': None}]},  # invented word
    {'inserts': [{'sfx_query': 'boom', 'anchor': {'type': 'time', 'value': 12, 'relation': 'at', 'occurrence': None}, 'offset': 0.7, 'gain_db': None, 'duck': None}]},  # invented offset
    {'inserts': []}, {'inserts': None},
    {'actions': [{'type': 'set_gain', 'stemId': 'x', 'value': 1}]},
    {'kind': 'clarify', 'message': 'Which?'},  # inserts on another kind
])
def test_llm_insert_plan_rejects_invented_or_malformed_output(client, tmp_path, monkeypatch, bad):
    pid, h = project(client, tmp_path, 20.0)
    model_plan(monkeypatch, llm_insert(**bad))
    body = {'prompt': 'please make a meme moment at 0:12', 'stems': [], 'tracks': [], 'duration': 20}
    assert client.post(f'/api/projects/{pid}/agent', headers=h, json=body).status_code == 502


def test_existing_plans_without_inserts_still_validate(client, tmp_path, monkeypatch):
    pid, h = project(client, tmp_path)
    model_plan(monkeypatch, {'kind': 'separate', 'description': 'a dog barking', 'remove': False, 'actions': [], 'message': 'ok'})
    plan = client.post(f'/api/projects/{pid}/agent', headers=h, json={'prompt': 'isolate the dog', 'stems': [], 'tracks': [], 'duration': 6}).json()
    assert plan['kind'] == 'separate' and plan['inserts'] is None


# ---------------------------------------------------------------- natural-language asks (offline fallback)

T, W, S, E, B, P = 'time', 'word', 'start', 'end', 'beat', 'pause'
ASKS = [
    ("add a vine boom when he says 'what'", [('vine_boom', W, 'what', 'at', None, 0.0)]),
    ('record scratch at 0:12', [('record_scratch', T, 12.0, 'at', None, 0.0)]),
    ('sad trombone at the end', [('sad_trombone', E, None, 'at', None, 0.0)]),
    ('crickets during the awkward silence', [('crickets', P, None, 'at', None, 0.0)]),
    ('add an airhorn at 1:05 and a rimshot at 1:10', [('airhorn', T, 65.0, 'at', None, 0.0), ('rimshot', T, 70.0, 'at', None, 0.0)]),
    ('put a ding at 3.5 seconds', [('ding', T, 3.5, 'at', None, 0.0)]),
    ('throw in a whoosh 0.2 seconds before 0:30', [('whoosh', T, 30.0, 'at', None, -0.2)]),
    ('add a dramatic sting before 0:45', [('dramatic_sting', T, 45.0, 'before', None, 0.0)]),
    ('vine boom every time she says "bruh"', [('vine_boom', W, 'bruh', 'at', None, 0.0)]),
    ('bruh sound the first time he says "no way"', [('bruh', W, 'no way', 'at', 1, 0.0)]),
    ('add applause at the start', [('applause', S, None, 'at', None, 0.0)]),
    ('tada at the very end of the video', [('tada', E, None, 'at', None, 0.0)]),
    ('play a ding on beat 4', [('ding', B, 4, 'at', None, 0.0)]),
    ('airhorn at 0:05, 0:10 and 0:15', [('airhorn', T, 5.0, 'at', None, 0.0), ('airhorn', T, 10.0, 'at', None, 0.0), ('airhorn', T, 15.0, 'at', None, 0.0)]),
    ("rimshot after he says 'and that's my wife'", [('rimshot', W, "and that's my wife", 'after', None, 0.0)]),
    ('cash register sound at 1:02.5', [('cash_register', T, 62.5, 'at', None, 0.0)]),
    ('add a whoosh right before he says "but"', [('whoosh', W, 'but', 'before', None, 0.0)]),
    ('can you add a sad trombone at the end please', [('sad_trombone', E, None, 'at', None, 0.0)]),
    ('add a laugh track 1 second after he says "joke"', [('laugh_track', W, 'joke', 'after', None, 1.0)]),
    ('add a vine boom when he says what', [('vine_boom', W, 'what', 'at', None, 0.0)]),
    ('dun dun dun at 1:00', [('dramatic_sting', T, 60.0, 'at', None, 0.0)]),
    ('add a cartoon boing on the downbeat', [('boing', B, 1, 'at', None, 0.0)]),
    ('add a dramatic zoom sound at 0:04', [('zoom_in', T, 4.0, 'at', None, 0.0)]),
    ('add a vine boom and a record scratch at 0:12', [('vine_boom', T, 12.0, 'at', None, 0.0), ('record_scratch', T, 12.0, 'at', None, 0.0)]),
    ("the last time he says 'okay' add a bruh", [('bruh', W, 'okay', 'at', -1, 0.0)]),
    ('add applause at 12', [('applause', T, 12.0, 'at', None, 0.0)]),
    ('glass shatter at 0:33', [('glass_shatter', T, 33.0, 'at', None, 0.0)]),
    ('i want a drumroll at 0:20', [('drumroll', T, 20.0, 'at', None, 0.0)]),
    ('add some laughter after 0:42', [('laugh_track', T, 42.0, 'after', None, 0.0)]),
    ('play the wrong answer buzzer at 7s', [('wrong_buzzer', T, 7.0, 'at', None, 0.0)]),
]


@pytest.mark.parametrize('prompt,expected', ASKS)
def test_natural_language_asks_map_to_insert_plans(prompt, expected):
    from app.sfx_plan import rule_plan
    plan = rule_plan(prompt)
    assert plan and plan['kind'] == 'insert', plan
    got = [(i['sfx_id'], i['anchor']['type'], i['anchor']['value'], i['anchor']['relation'], i['anchor']['occurrence'], i['offset']) for i in plan['inserts']]
    assert got == expected


@pytest.mark.parametrize('prompt,gain,duck', [
    ('add a quieter laugh track at 0:20', -6.0, False), ('put a really loud airhorn at 0:03', 6.0, False),
    ('add a boom at 0:07 -3 dB', -3.0, False), ('add a record scratch at 12s and duck the music', 0.0, True),
    ('glass shatter at 0:33 with ducking', 0.0, True),
])
def test_asks_with_gain_and_ducking(prompt, gain, duck):
    from app.sfx_plan import rule_plan
    plan = rule_plan(prompt)
    assert plan['kind'] == 'insert' and plan['inserts'][0]['gain_db'] == gain and plan['inserts'][0]['duck'] is duck


@pytest.mark.parametrize('prompt,flag', [
    ('laugh track after the joke', 'sfx_needs_time'), ('drumroll before the reveal', 'sfx_needs_time'),
    ('add a vine boom when he falls', 'sfx_needs_time'), ('sad trombone on the drop', 'sfx_needs_time'),
    ('add crickets', 'sfx_needs_time'), ('add a quack at 3 seconds', 'sfx_unmatched'),
])
def test_asks_that_need_a_clarifying_question(prompt, flag):
    from app.sfx_plan import rule_plan
    plan = rule_plan(prompt)
    assert plan['kind'] == 'clarify' and plan.get(flag) is True and plan['message']


@pytest.mark.parametrize('prompt', ['add some reverb to the vocals', 'replace the punch with a quack', 'make it louder', 'mute the music from 0:10 to 0:20',
                                    'isolate the dog', 'add an echo at 0:10', 'remove the dog barking at 0:12', 'split this song into tracks',
                                    'make the drums sound like a phone', 'turn the vocals up'])
def test_other_requests_are_left_to_the_other_planners(prompt):
    from app.sfx_plan import rule_plan
    assert rule_plan(prompt) is None
