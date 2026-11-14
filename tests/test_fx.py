"""Chat-driven layer effects: DSL clamps, rendering guards, ducking, planner schema and the API flow."""
import json
import math
import subprocess
import time

import numpy as np
import pytest

from app import fx, fx_schema as S

SR = 48000


def tone(seconds=1.5, channels=2, level=.3):
    t = np.arange(int(SR * seconds)) / SR
    x = np.stack([level * np.sin(2 * np.pi * (220 + 110 * c) * t) for c in range(channels)], axis=1).astype(np.float32)
    x[int(SR * .5):int(SR * .5) + 50] += .5  # A transient, so effects with tails have something to ring.
    return x


# ---------------------------------------------------------------- DSL

def test_effect_params_are_clamped_and_filled():
    notes = []
    e = S.clamp_effect({'type': 'reverb', 'params': {'decay': 99, 'mix': -3, 'size': None, 'frequency': 5}}, notes)
    assert e['params']['decay'] == 8 and e['params']['mix'] == 0 and e['params']['size'] == .5
    assert 'frequency' not in e['params'] and len(notes) == 2
    assert S.clamp_effect({'type': 'delay', 'params': {'feedback': 5}})['params']['feedback'] == .9
    assert S.clamp_effect({'type': 'pitch', 'params': {'semitones': -40}})['params']['semitones'] == -12
    assert S.clamp_effect({'type': 'bitcrush', 'params': {'bits': 3.7}})['params']['bits'] == 4
    assert S.clamp_effect({'type': 'limiter', 'params': {'ceiling': 0}})['params']['ceiling'] == -1


@pytest.mark.parametrize('bad', [{'type': 'nuke'}, {'type': 'reverb', 'params': {'mix': float('nan')}}, {'type': 'reverb', 'params': {'mix': 'loud'}},
                                 {'type': 'gain', 'params': {'gain': True}}, {'type': 'bandpass', 'params': {'low': 3000, 'high': 300}}, 'reverb'])
def test_invalid_effects_are_rejected(bad):
    with pytest.raises(S.FxError): S.clamp_effect(bad)


def test_chain_speed_and_interval_limits():
    with pytest.raises(S.FxError): S.clamp_chain([{'type': 'gain', 'params': {}}] * (S.MAX_CHAIN + 1))
    with pytest.raises(S.FxError): S.clamp_speed(3)
    assert S.clamp_speed(1.0) is None and S.clamp_speed(.85) == .85
    with pytest.raises(S.FxError): S.clamp_interval({'start': 5, 'end': 4}, 10)
    with pytest.raises(S.FxError): S.clamp_interval({'start': 1, 'end': 11}, 10)
    with pytest.raises(S.FxError): S.item('slowed', interval={'start': 1, 'end': 2}, duration=10)  # Tempo is whole-song.
    with pytest.raises(S.FxError): S.item()
    with pytest.raises(S.FxError): S.item('not-a-preset')


def test_presets_cover_the_plan_and_are_valid():
    for name in ['church', 'stadium', 'bathroom', 'small_room', 'plate', 'telephone', 'radio', 'walkie_talkie', 'through_the_wall', 'underwater',
                 'megaphone', 'lofi', 'vinyl', 'eight_bit', 'old_movie', 'chipmunk', 'deep_voice', 'robot', 'announcer', 'giant', 'dreamy', 'wide',
                 'mono', 'warm', 'bright', 'less_harsh', 'slowed_reverb', 'nightcore', 'deep_fried']:
        assert name in S.PRESETS
    assert len(S.PRESETS) >= 30 and len(S.EFFECTS) >= 20
    for name in S.PRESETS:
        it = S.item(name)
        assert it['chain'] == S.clamp_chain(it['chain'])  # Preset chains are already inside the clamps.
        assert bool(it['speed']) == (name in S.TEMPO_PRESETS)


def test_more_and_less_adjust_within_bounds():
    it = S.item('church')
    more = S.adjust_item(it, 'more'); less = S.adjust_item(it, 'less')
    assert more['chain'][0]['params']['mix'] > it['chain'][0]['params']['mix'] > less['chain'][0]['params']['mix']
    assert more['label'] == 'Church (more)' and S.adjust_item(more, 'more')['label'] == 'Church (more x2)'
    assert S.adjust_item(more, 'less')['label'] == 'Church'
    strongest = it
    with pytest.raises(S.FxError):
        for _ in range(40): strongest = S.adjust_item(strongest, 'more')
    assert all(v <= S.EFFECTS['reverb'][k][1] for k, v in strongest['chain'][0]['params'].items())


# ---------------------------------------------------------------- rendering

@pytest.mark.parametrize('name', S.PRESET_NAMES)
def test_every_preset_renders_and_preserves_length(name):
    x = tone(); it = S.item(name)
    y = fx.render(x, SR, [it], [])
    expected = len(x) if not it['speed'] else round(len(x) / it['speed'])
    assert y.shape == (expected, 2) and y.dtype == np.float32
    assert np.isfinite(y).all() and fx.true_peak_db(y) <= -1 + .02
    assert fx._rms(y) > 1e-4


@pytest.mark.parametrize('effect', S.EFFECT_TYPES)
def test_every_effect_type_renders_mono_and_stereo(effect):
    for channels in (1, 2):
        x = tone(.6, channels)
        y = fx.render(x, SR, [S.item(chain=[{'type': effect, 'params': {}}])], [])
        assert y.shape == x.shape and np.isfinite(y).all()


def test_rendering_is_deterministic():
    x = tone(); it = S.item('vinyl')
    assert np.array_equal(fx.render(x, SR, [it], []), fx.render(x, SR, [it], []))


def test_true_peak_guard_on_hot_signal():
    t = np.arange(SR) / SR
    hot = np.stack([1.6 * np.sin(2 * np.pi * 11025 * t + .3)] * 2, axis=1).astype(np.float32)  # Inter-sample peaks.
    y = fx.guard(hot, SR)
    assert fx.true_peak_db(y) <= -1 + .02 and len(y) == len(hot)
    quiet = tone(level=.1)
    assert np.array_equal(fx.guard(quiet, SR), quiet)  # Transparent below the ceiling.
    with pytest.raises(ValueError): fx.guard(np.full((10, 2), np.nan, np.float32), SR)


def test_interval_effect_with_crossfades():
    x = tone(3, level=.2)
    it = S.item(chain=[{'type': 'gain', 'params': {'gain': -24}}], interval={'start': 1, 'end': 2}, duration=3)
    y = fx.render(x, SR, [it], [])
    assert np.allclose(y[:SR], x[:SR]) and np.allclose(y[2 * SR:], x[2 * SR:])
    inside = slice(int(1.1 * SR), int(1.9 * SR))
    assert fx._rms(y[inside]) < fx._rms(x[inside]) * .1
    ratio = np.abs(y[SR:SR + 1440, 0]) / np.maximum(np.abs(x[SR:SR + 1440, 0]), 1e-6)
    assert ratio[10] > .9 and ratio[-10] < .2  # A ramp, not a step.


def test_four_minute_stem_renders_in_seconds():
    x = (np.random.default_rng(1).standard_normal((SR * 240, 2)) * .1).astype(np.float32)
    began = time.monotonic(); y = fx.render(x, SR, [S.item('church')], [])
    assert len(y) == len(x) and time.monotonic() - began < 15


def test_deep_fried_is_loudness_capped():
    x = tone(2, level=.1); notes = []
    y = fx.render(x, SR, [S.item('deep_fried')], notes)
    assert fx.loudness(y, SR) - fx.loudness(x, SR) <= 3.5
    assert fx.true_peak_db(y) <= -1 + .02


# ---------------------------------------------------------------- ducking

def test_ducking_automation_shape():
    voice = np.zeros((SR * 10, 1), np.float32)
    t = np.arange(SR) / SR
    for start in (2, 6):
        voice[start * SR:(start + 1) * SR, 0] = .3 * np.sin(2 * np.pi * 200 * t)
    params = S.duck_params({'depth_db': 12})
    points, regions, _ = fx.duck_keyframes(voice, SR, 0, [], 1.0, 10, [], params)
    assert len(regions) == 2 and len(points) <= 64
    times = [p['time'] for p in points]; values = [p['value'] for p in points]
    assert times == sorted(times) and len(set(times)) == len(times)
    curve = lambda when: np.interp(when, times, values)
    duck = 10 ** (-12 / 20)
    assert curve(.5) == 1 and curve(4.3) == 1 and curve(9.5) == 1
    assert abs(curve(2.5) - duck) < 1e-3 and abs(curve(6.5) - duck) < 1e-3
    # Existing automation multiplies and the budget holds with many phrases.
    many = np.zeros((SR * 60, 1), np.float32)
    for k in range(40): many[int((k * 1.5) * SR):int((k * 1.5 + .4) * SR), 0] = .3
    existing = [{'time': 0, 'value': .5}, {'time': 60, 'value': .5}]
    points, regions, _ = fx.duck_keyframes(many, SR, 0, [], 1.0, 60, existing, S.duck_params({'attack_ms': 10, 'release_ms': 50}))
    assert len(points) <= 64 and max(p['value'] for p in points) <= .5 + 1e-6


# ---------------------------------------------------------------- planner schema

def walk(schema, path='$'):
    if isinstance(schema, dict):
        if schema.get('type') == 'object' or 'properties' in schema:
            assert schema.get('additionalProperties') is False, path
            assert sorted(schema['required']) == sorted(schema['properties']), path
        for key, value in schema.items(): walk(value, f'{path}.{key}')
    elif isinstance(schema, list):
        for i, value in enumerate(schema): walk(value, f'{path}[{i}]')


def test_planner_schema_is_strict():
    from app.agent import SCHEMA, INSTRUCTIONS
    walk(SCHEMA)
    assert 'fx' in SCHEMA['required'] and {'fx', 'duck', 'loudness'} <= set(SCHEMA['properties']['kind']['enum'])
    assert 'effects applied by this tool' not in INSTRUCTIONS and 'cannot be generated' in INSTRUCTIONS
    json.dumps(SCHEMA)


def fake_planner(monkeypatch, plan):
    import app.agent as agent
    monkeypatch.setenv('OPENAI_API_KEY', 'not-a-real-key')
    sent = {}
    class Response:
        is_error = False
        def json(self): return {'status': 'completed', 'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(plan)}]}]}
    def post(url, **kwargs): sent.update(kwargs['json']); return Response()
    monkeypatch.setattr(agent.httpx, 'post', post)
    return agent, sent


def base_plan(kind, fx_plan, message='ok'):
    empty = {'targets': ['mix'], 'preset': None, 'chain': None, 'interval': None, 'adjust': None, 'speed': None,
             'duck_source': None, 'duck_depth_db': None, 'loudness_lufs': None}
    return {'kind': kind, 'guidance': {'anchors': []}, 'edit_scope': {'action': 'keep', 'interval': None}, 'description': None, 'remove': False,
            'processing_time': None, 'separation_mode': None, 'stems': None, 'actions': [], 'use_visual': False,
            'fx': {**empty, **fx_plan}, 'message': message}


STEMS = [{'id': 'v1', 'label': 'Vocals', 'canonical_label': 'vocals', 'duration': 30}, {'id': 'd1', 'label': 'Drums', 'canonical_label': 'drums', 'duration': 30}]


def test_planner_fx_plan_is_validated(monkeypatch):
    agent, sent = fake_planner(monkeypatch, base_plan('fx', {'targets': ['v1'], 'preset': 'church',
                                                              'chain': [{'type': 'delay', 'params': {k: None for k in S.PARAM_KEYS} | {'feedback': 4}}]}))
    body = agent.AgentRequest(prompt='put the vocals in a church with an echo', stems=STEMS, tracks=[], duration=30, has_project_stems=True)
    result = agent.plan(body)
    assert result['kind'] == 'fx' and result['fx']['targets'] == ['v1'] and result['fx']['preset'] == 'church'
    assert result['fx']['chain'][0]['params']['feedback'] == .9
    assert sent['text']['format']['strict'] is True and 'church' in sent['instructions']


@pytest.mark.parametrize('kind,plan,needle', [
    ('fx', {'targets': ['nope'], 'preset': 'church'}, 'existing layer'),
    ('fx', {'targets': ['v1'], 'preset': 'nightcore'}, 'whole song'),
    ('fx', {'targets': ['v1'], 'preset': 'church', 'interval': {'start': 3, 'end': 9}}, 'Unsupplied'),
    ('fx', {'targets': ['v1']}, 'Choose an effect'),
    ('duck', {'targets': ['d1'], 'duck_source': None}, 'voice'),
])
def test_planner_unrenderable_fx_becomes_clarify(monkeypatch, kind, plan, needle):
    agent, _ = fake_planner(monkeypatch, base_plan(kind, plan))
    body = agent.AgentRequest(prompt='make the vocals sound like a church', stems=STEMS, tracks=[], duration=30, has_project_stems=True)
    result = agent.plan(body)
    assert result['kind'] == 'clarify' and needle in result['message']


def test_planner_duck_and_loudness(monkeypatch):
    agent, _ = fake_planner(monkeypatch, base_plan('duck', {'targets': ['mix'], 'duck_source': 'v1'}))
    body = agent.AgentRequest(prompt='duck the drums under the vocals', stems=STEMS, tracks=[], duration=30, has_project_stems=True)
    assert agent.plan(body)['fx']['targets'] == ['d1']
    agent, _ = fake_planner(monkeypatch, base_plan('loudness', {'loudness_lufs': -14}))
    assert agent.plan(agent.AgentRequest(prompt='normalize for youtube', stems=STEMS, tracks=[], duration=30))['fx']['loudness_lufs'] == -14


def test_effect_asks_are_not_routed_to_replacement():
    assert S.mentions_effect('make the drums sound like a phone') and S.mentions_effect('turn the vocals into a robot')
    assert not S.mentions_effect('make the dog bark sound like a cat')


REPLACEMENT_ASKS = ['turn the chorus into a verse', 'make the phone ringing sound like a doorbell', 'turn the monster roar into a whisper',
                    'make the radio chatter sound like birdsong', 'turn the hall into a kitchen', 'make the room tone sound like rain',
                    'turn the wall clock into a metronome', 'change the dog bark into a monster roar', 'make the robot sound like a cat']


@pytest.mark.parametrize('text', REPLACEMENT_ASKS)
def test_replacement_asks_do_not_mention_an_effect(text):
    # Only the "sound like" / "into" target counts, and a preset word naming a sound ("monster roar") is not an effect.
    assert not S.mentions_effect(text)


@pytest.mark.parametrize('text', ['make the drums sound like a phone', 'turn the vocals into a robot', 'make it sound like an old radio',
                                  'make the voice sound like a giant', 'make the vocals sound like they are in a cathedral'])
def test_effect_asks_mention_an_effect(text):
    assert S.mentions_effect(text)


# ---------------------------------------------------------------- API end to end (fake adapter)

@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv('STORAGE_ROOT', str(tmp_path / 'storage'))
    monkeypatch.setenv('INFERENCE_ADAPTER', 'fake')
    import app.core as core
    core.ROOT = tmp_path / 'storage'; core.ROOT.mkdir(exist_ok=True); core.DB = core.ROOT / 'db.sqlite'
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c: yield c


def upload(client, tmp_path, video=False, seconds=2):
    p = client.post('/api/projects').json(); pid = p['project_id']; h = {'Authorization': 'Bearer ' + p['access_token']}
    path = tmp_path / ('clip.mp4' if video else 'clip.wav')
    if video:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'color=c=blue:s=64x64:r=10:d={seconds}', '-f', 'lavfi', '-i', f'sine=frequency=440:duration={seconds}',
                        '-c:v', 'mpeg4', '-c:a', 'aac', '-shortest', str(path)], check=True)
    else:
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'sine=frequency=440:duration={seconds}', '-ac', '2', '-ar', '48000', str(path)], check=True)
    with path.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload', headers=h, files={'file': (path.name, f)}).status_code == 200
    return pid, h


def wait(client, url, h):
    for _ in range(600):
        data = client.get(url, headers=h).json()
        if data['status'] in ('completed', 'failed'): return data
        time.sleep(.05)
    pytest.fail('job timed out')


def split(client, pid, h, prompts=('music', 'speech')):
    jid = client.post(f'/api/projects/{pid}/process', headers=h, json={'prompts': list(prompts)}).json()['job_id']
    assert wait(client, '/api/jobs/' + jid, h)['status'] == 'completed'
    return {r['name']: r for r in client.get(f'/api/projects/{pid}/stems', headers=h).json()}


def preview(client, pid, h, body):
    r = client.post(f'/api/projects/{pid}/fx-preview', headers=h, json=body)
    assert r.status_code == 202, r.text
    result = wait(client, '/api/candidates/' + r.json()['candidate_id'], h)
    assert result['status'] == 'completed', result
    return result


def audio(client, h, stem):
    return client.get(stem['audio_url'], headers=h).content


def test_fx_preview_apply_chip_remove_and_undo(client, tmp_path):
    pid, h = upload(client, tmp_path)
    stems = split(client, pid, h); music, speech = stems['music'], stems['speech']
    before, other = audio(client, h, music), audio(client, h, speech)
    url = f'/api/projects/{pid}/fx-preview'
    assert client.post(url, json={'targets': [music['id']], 'preset': 'church'}).status_code == 401
    assert client.post(url, headers=h, json={'targets': ['nope'], 'preset': 'church'}).status_code == 422
    assert client.post(url, headers=h, json={'targets': [music['id']], 'preset': 'kazoo'}).status_code == 422
    assert client.post(url, headers=h, json={'targets': [music['id']], 'preset': 'nightcore'}).status_code == 422
    result = preview(client, pid, h, {'targets': [music['id']], 'preset': 'church'})
    assert result['provenance']['kind'] == 'fx' and result['provenance']['item']['label'] == 'Church'
    assert client.get(result['mix_url'], headers=h).status_code == 200
    cid = result['id']
    assert client.post(f'/api/candidates/{cid}/apply', headers=h).status_code == 200
    rows = {r['name']: r for r in client.get(f'/api/projects/{pid}/stems', headers=h).json()}
    assert [c['label'] for c in rows['music']['fx']['items']] == ['Church'] and rows['speech']['fx'] is None
    assert 'dry_path' not in json.dumps(rows) and audio(client, h, rows['music']) != before and audio(client, h, rows['speech']) == other
    assert rows['music']['duration'] == pytest.approx(music['duration'], abs=1e-3)
    # A second chip renders from the dry audio with both effects.
    second = preview(client, pid, h, {'targets': [music['id']], 'preset': 'telephone'})
    assert client.post(f"/api/candidates/{second['id']}/apply", headers=h).status_code == 200
    rows = {r['name']: r for r in client.get(f'/api/projects/{pid}/stems', headers=h).json()}
    chips = rows['music']['fx']['items']; assert [c['label'] for c in chips] == ['Church', 'Telephone']
    # "More" on the applied chip adjusts it in place.
    more = preview(client, pid, h, {'op': 'adjust', 'targets': [music['id']], 'direction': 'more'})
    assert more['provenance']['item']['label'] == 'Telephone (more)'
    # Remove both chips: the layer is its unprocessed audio again.
    for chip in chips:
        removed = preview(client, pid, h, {'op': 'remove', 'targets': [music['id']], 'item_id': chip['id']})
        assert client.post(f"/api/candidates/{removed['id']}/apply", headers=h).status_code == 200
    rows = {r['name']: r for r in client.get(f'/api/projects/{pid}/stems', headers=h).json()}
    assert rows['music']['fx'] is None and audio(client, h, rows['music']) == before
    # Undo the last removal brings the Telephone chip back.
    assert client.post(f"/api/candidates/{removed['id']}/undo", headers=h).status_code == 200
    rows = {r['name']: r for r in client.get(f'/api/projects/{pid}/stems', headers=h).json()}
    assert [c['label'] for c in rows['music']['fx']['items']] == ['Telephone']
    assert client.get(rows['music']['audio_url'], headers=h).status_code == 200


def test_fx_on_unsplit_audio_mix_interval_and_unapplied_more(client, tmp_path):
    pid, h = upload(client, tmp_path, seconds=3)
    first = preview(client, pid, h, {'targets': ['mix'], 'chain': [{'type': 'gain', 'params': {'gain': -12}}], 'interval': {'start': 1, 'end': 2}})
    assert first['provenance']['item']['interval'] == {'start': 1, 'end': 2}
    stronger = preview(client, pid, h, {'op': 'adjust', 'targets': ['mix'], 'direction': 'more', 'base_candidate_id': first['id']})
    assert stronger['provenance']['op'] == 'chain' and stronger['provenance']['item']['chain'][0]['params']['gain'] == -18
    assert client.post(f"/api/candidates/{stronger['id']}/apply", headers=h).status_code == 200
    row = client.get(f'/api/projects/{pid}/stems', headers=h).json()[0]
    assert row['fx']['items'][0]['scope'] == 'mix' and row['duration'] == pytest.approx(3, abs=1e-3)


def test_speed_is_whole_song_and_audio_only(client, tmp_path):
    pid, h = upload(client, tmp_path, seconds=2)
    stems = split(client, pid, h)
    result = preview(client, pid, h, {'targets': ['mix'], 'preset': 'slowed_reverb'})
    assert result['provenance']['speed'] == .85 and any('speed' in w for w in result['inference']['warnings'])
    assert client.post(f"/api/candidates/{result['id']}/apply", headers=h).status_code == 200
    rows = client.get(f'/api/projects/{pid}/stems', headers=h).json()
    assert all(r['duration'] == pytest.approx(2 / .85, abs=2e-3) for r in rows) and len(rows) == len(stems)
    eid = client.post(f'/api/projects/{pid}/export', headers=h, json={'format': 'wav', 'include_video': False}).json()['export_id']
    done = wait(client, '/api/exports/' + eid, h); assert done['status'] == 'completed'
    out = tmp_path / 'slowed.wav'; out.write_bytes(client.get(done['download_url'], headers=h).content)
    from app.core import probe
    assert probe(out)['duration'] == pytest.approx(2 / .85, abs=.01)
    vid, vh = upload(client, tmp_path, video=True)
    r = client.post(f'/api/projects/{vid}/fx-preview', headers=vh, json={'targets': ['mix'], 'preset': 'nightcore'})
    assert r.status_code == 422 and 'video' in r.json()['detail']


def test_duck_and_loudness_previews(client, tmp_path):
    pid, h = upload(client, tmp_path, seconds=2)
    stems = split(client, pid, h); music, speech = stems['music'], stems['speech']
    url = f'/api/projects/{pid}/fx-preview'
    assert client.post(url, headers=h, json={'op': 'duck', 'targets': ['mix'], 'duck': {'source': 'nope'}}).status_code == 422
    result = preview(client, pid, h, {'op': 'duck', 'targets': [music['id']], 'duck': {'source': speech['id'], 'depth_db': 10}})
    assert result['provenance']['op'] == 'duck' and result['provenance']['targets'][0]['id'] == music['id']
    before = audio(client, h, music)
    assert client.post(f"/api/candidates/{result['id']}/apply", headers=h).status_code == 200
    rows = {r['name']: r for r in client.get(f'/api/projects/{pid}/stems', headers=h).json()}
    assert rows['music']['automation'] and min(p['value'] for p in rows['music']['automation']) == pytest.approx(10 ** (-.5), abs=1e-3)
    assert audio(client, h, rows['music']) == before and rows['speech']['automation'] == []  # Automation only; no render.
    loud = preview(client, pid, h, {'op': 'loudness', 'targets': ['mix'], 'loudness_lufs': -20})
    assert loud['provenance']['loudness']['target_lufs'] == -20 and loud['provenance']['loudness']['after_lufs'] == pytest.approx(-20, abs=1.5)
    assert loud['provenance']['mix_true_peak_db'] <= -.9
    layer = preview(client, pid, h, {'op': 'loudness', 'targets': [speech['id']], 'loudness_lufs': -30})
    assert layer['provenance']['label'] == 'Loudness -30 LUFS'


def test_dry_audio_survives_cleanup(client, tmp_path):
    pid, h = upload(client, tmp_path)
    stems = split(client, pid, h); music = stems['music']
    first = preview(client, pid, h, {'targets': [music['id']], 'preset': 'warm'})
    assert client.post(f"/api/candidates/{first['id']}/apply", headers=h).status_code == 200
    second = preview(client, pid, h, {'targets': [music['id']], 'preset': 'wide'})
    assert client.post(f"/api/candidates/{second['id']}/apply", headers=h).status_code == 200
    import app.core as core
    with core.connect() as db: row = dict(db.execute('SELECT * FROM stems WHERE id=?', (music['id'],)).fetchone())
    record = json.loads(row['fx'])
    from pathlib import Path
    assert Path(record['dry_path']).is_file() and Path(row['path']).is_file()


def test_agent_route_keeps_effect_asks_away_from_replacement(client, tmp_path, monkeypatch):
    pid, h = upload(client, tmp_path)
    stems = split(client, pid, h)
    agent, sent = fake_planner(monkeypatch, base_plan('fx', {'targets': [stems['music']['id']], 'preset': 'telephone'}))
    r = client.post(f'/api/projects/{pid}/agent', headers=h, json={'prompt': 'make the music sound like a phone', 'stems': [], 'tracks': [], 'duration': 2})
    assert r.status_code == 200 and r.json()['kind'] == 'fx', r.text
    payload = json.loads(sent['input'])
    assert {s['label'] for s in payload['stems']} == {'music', 'speech'} and all('fx' in s for s in payload['stems'])


@pytest.mark.parametrize('prompt', ['replace the phone ringing with a doorbell', 'replace the dog bark with a monster roar',
                                    'turn the chorus into a verse', 'make the phone ringing sound like a doorbell'])
def test_agent_route_sends_replacement_asks_to_replacement(client, tmp_path, monkeypatch, prompt):
    pid, h = upload(client, tmp_path)
    split(client, pid, h)
    _, sent = fake_planner(monkeypatch, base_plan('fx', {'targets': ['mix'], 'preset': 'telephone'}))
    r = client.post(f'/api/projects/{pid}/agent', headers=h, json={'prompt': prompt, 'stems': [], 'tracks': [], 'duration': 2})
    assert r.status_code == 200 and r.json()['kind'] == 'clarify' and 'replacement' in r.json()['message'], r.text
    assert not sent  # The effect planner never saw it.


def test_legacy_database_gains_fx_column(tmp_path, monkeypatch):
    import sqlite3
    import app.core as core
    monkeypatch.setattr(core, 'DB', tmp_path / 'legacy.sqlite')
    with sqlite3.connect(core.DB) as db:
        db.execute("CREATE TABLE projects (id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, status TEXT NOT NULL, filename TEXT, original TEXT, audio TEXT, metadata TEXT, visual TEXT)")
        db.execute("CREATE TABLE stems (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, name TEXT NOT NULL, path TEXT NOT NULL, duration REAL NOT NULL, start_offset REAL NOT NULL DEFAULT 0, volume REAL NOT NULL DEFAULT 1, muted INTEGER NOT NULL DEFAULT 0, solo INTEGER NOT NULL DEFAULT 0, automation TEXT NOT NULL DEFAULT '[]', mute_ranges TEXT NOT NULL DEFAULT '[]')")
        db.execute("INSERT INTO stems(id,project_id,name,path,duration) VALUES('s','p','x','/x.wav',1)")
    core.init()
    with core.connect() as db:
        row = dict(db.execute('SELECT * FROM stems').fetchone())
    assert row['fx'] is None and core.stem(row)['fx'] is None
