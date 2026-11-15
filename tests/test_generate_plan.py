import pytest

from app.generate_part import GENERATE, rule_plan, validate_generate


@pytest.mark.parametrize("prompt,instrument,sections", [
    ("add violins to the song", "violins", None),
    ("add a piano part", "piano", None),
    ("add a funky bassline in the chorus", "funky bassline", ["chorus"]),
    ("put a string pad under the chorus", "string pad", ["chorus"]),
    ("add an 80s synth arpeggio", "80s synth arpeggio", None),
])
def test_rule_plan(prompt, instrument, sections):
    plan = rule_plan(prompt)
    assert plan["instrument"] == instrument and plan["sections"] == sections
    assert validate_generate(plan, 60.0)["prompt"] == prompt


def test_rule_plan_ignores_effects():
    assert rule_plan("add reverb to the vocals") is None
    assert rule_plan("make it louder") is None
    assert rule_plan("add automation to the vocals") is None
    assert rule_plan("add a fade in") is None


def test_validate_rejects_bad_plans():
    with pytest.raises(ValueError):
        validate_generate({"instrument": "", "prompt": ""}, 10)
    with pytest.raises(ValueError):
        validate_generate({"instrument": "piano", "role": "solo-ish"}, 10)
    with pytest.raises(ValueError):
        validate_generate({"instrument": "piano", "interval": {"start": 5, "end": 50}}, 10)
    assert validate_generate({"instrument": "piano", "density": 3}, 10)["density"] == 1.0


def test_schema_is_strict():
    assert GENERATE["additionalProperties"] is False and set(GENERATE["required"]) == set(GENERATE["properties"])


# ---------------------------------------------------------------- generate-preview candidate (service faked)
import base64
import json
import subprocess
import time

import numpy as np


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.core as core
    core.ROOT = tmp_path/'storage'; core.ROOT.mkdir(); core.DB = core.ROOT/'db.sqlite'
    monkeypatch.setenv('STORAGE_ROOT', str(core.ROOT))
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c: yield c


def _uploaded(client, tmp_path):
    p = client.post('/api/projects').json(); pid = p['project_id']; h = {'Authorization': 'Bearer '+p['access_token']}
    path = tmp_path/'song.wav'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=frequency=220:duration=2', '-ac', '2', '-ar', '48000', str(path)], check=True)
    with path.open('rb') as f:
        assert client.post(f'/api/projects/{pid}/upload', headers=h, files={'file': ('song.wav', f)}).status_code == 200
    return pid, h


def _wait(client, cid, h):
    for _ in range(300):
        j = client.get('/api/candidates/'+cid, headers=h).json()
        if j['status'] in ('completed', 'failed'): return j
        time.sleep(.05)
    pytest.fail('preview timed out')


def test_generate_preview_adds_a_layer_apply_undo(client, tmp_path, monkeypatch):
    from app import generate_part
    from app.media import encode_wav, probe
    calls = {}

    def fake_service(mix_path, stems, plan, seed=0, **kw):
        info = probe(mix_path)
        n = round(info['duration']*48000)
        tone = (0.1*np.sin(2*np.pi*440*np.arange(n)/48000)).astype(np.float32)
        calls.update(plan=plan, stems=dict(stems))
        calls.setdefault('analysis', []).append((kw.get('analysis'), kw.get('output_sample_rate')))
        return {'part': {'name': 'Violins', 'display_name': 'Violins', 'kind': 'music', 'family': 'strings', 'role': 'counter_melody'},
                'audio': {'audio_base64': base64.b64encode(encode_wav(np.stack([tone, tone], axis=1), 48000)).decode()},
                'midi_base64': None, 'engine': {'name': 'symbolic'}, 'renderer': {'used': 'vsco:violins'}, 'windows': [[0, 2]],
                'checks': {'notes': {'chord_tone_ratio': 0.98}}, 'mix': {}, 'warnings': [], 'analysis': {'tempo': 120.0}}

    monkeypatch.setattr(generate_part, 'request_generate_part', fake_service)
    pid, h = _uploaded(client, tmp_path)
    plan = client.post(f'/api/projects/{pid}/agent', headers=h, json={'prompt': 'add violins to the song', 'stems': [], 'tracks': [], 'duration': 2}).json()
    assert plan['kind'] == 'generate' and plan['generate']['instrument'] == 'violins'
    assert client.post(f'/api/projects/{pid}/generate-preview', headers=h, json={'prompt': 'x', 'interval': {'start': 1, 'end': 9}}).status_code == 422
    r = client.post(f'/api/projects/{pid}/generate-preview', headers=h, json={k: plan['generate'][k] for k in ('instrument', 'prompt', 'sections')})
    assert r.status_code == 202, r.text
    job = _wait(client, r.json()['candidate_id'], h)
    assert job['status'] == 'completed', job
    assert job['provenance']['kind'] == 'generate' and calls['plan']['engine'] == 'symbolic'
    assert client.get(job['mix_url'], headers=h).content[:4] == b'RIFF'
    cid = r.json()['candidate_id']
    assert client.post(f'/api/candidates/{cid}/apply', headers=h).status_code == 200
    stems = client.get(f'/api/projects/{pid}/stems', headers=h).json()
    assert sorted(s['name'] for s in stems) == ['Original audio', 'Violins']
    assert client.post(f'/api/candidates/{cid}/undo', headers=h).status_code == 200
    assert client.get(f'/api/projects/{pid}/stems', headers=h).json() == []
    # Second ask on the same audio: the cached analysis is sent (no re-analysis, light upload).
    r2 = client.post(f'/api/projects/{pid}/generate-preview', headers=h, json={'instrument': 'piano', 'prompt': 'add a piano part'})
    job2 = _wait(client, r2.json()['candidate_id'], h)
    assert job2['status'] == 'completed' and job2['provenance']['analysis_cached'] is True
    assert calls['analysis'][0][0] is None and calls['analysis'][1] == ({'tempo': 120.0}, 48000)


# ---------------------------------------------------------------- agent-route precedence (matches frontend chatRoute.ts)
@pytest.mark.parametrize("prompt", ["add reverb to the vocals", "add chorus to the guitar", "add compression to the bass",
                                    "add more bass", "add a fade", "add a vine boom at 0:12", "add a laugh track",
                                    "put a riser before the drop", "add a sad trombone at the end"])
def test_rule_plan_leaves_fx_eq_and_sfx_alone(prompt):
    assert rule_plan(prompt) is None


def test_rule_plan_trims_time_tail():
    assert rule_plan("add a piano at 0:12")["instrument"] == "piano"


def test_rule_plan_keeps_timing():
    assert rule_plan("add horns at 0:05", 60)["interval"] == {"start": 5.0, "end": 60.0}
    assert rule_plan("add violins from 0:05 to 0:20", 60)["interval"] == {"start": 5.0, "end": 20.0}
    assert rule_plan("add a piano at 12 seconds", 60)["interval"] == {"start": 12.0, "end": 60.0}
    assert "past the end" in rule_plan("add violins at 1:30", 60)["clarify"]
    assert "word" in rule_plan("add a piano when he says hello", 60)["clarify"]
    assert rule_plan("add a piano at 0:05")["clarify"]  # no duration known: ask rather than guess


@pytest.mark.parametrize("prompt", ["put the keys on the table", "put the keys in the drawer", "bring in the horns of the car"])
def test_rule_plan_needs_musical_context_for_weak_verbs(prompt):
    assert rule_plan(prompt, 60) is None


@pytest.mark.parametrize("prompt", ["put a string pad under the chorus", "give me a funky bassline", "layer some strings", "play a piano part in the chorus"])
def test_rule_plan_musical_weak_verbs(prompt):
    assert rule_plan(prompt, 60) is not None


@pytest.mark.parametrize("interval", [{"start": 1}, {"start": "x", "end": 2}, {"start": 1, "end": float("nan")}, "0:05"])
def test_validate_malformed_interval_is_value_error(interval):
    with pytest.raises(ValueError, match="interval"):
        validate_generate({"instrument": "piano", "interval": interval}, 10)


def test_generate_preview_malformed_interval_422(client, tmp_path):
    pid, h = _uploaded(client, tmp_path)
    for interval in ({'start': 1}, {'start': 'x', 'end': 2}, {'end': 1}):
        r = client.post(f'/api/projects/{pid}/generate-preview', headers=h, json={'prompt': 'add a piano', 'interval': interval})
        assert r.status_code == 422 and 'interval' in r.json()['detail'], r.text


def test_multi_preview_malformed_interval_message(client, tmp_path):
    pid, h = _uploaded(client, tmp_path)
    r = client.post(f'/api/projects/{pid}/multi-preview', headers=h,
                    json={'request': 'x', 'steps': [{'kind': 'generate', 'generate': {'instrument': 'piano', 'prompt': 'p', 'interval': {'start': 0}}}]})
    assert r.status_code == 422 and r.json()['detail'].endswith('interval needs start and end (seconds)'), r.text
    assert "'start'" not in r.json()['detail']


def _fake_service(calls):
    from app.media import encode_wav, probe

    def fake(mix_path, stems, plan, seed=0, **kw):
        n = round(probe(mix_path)['duration'] * 48000)
        tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(n) / 48000)).astype(np.float32)
        calls.append({'analysis': kw.get('analysis'), 'stems': dict(stems)})
        return {'part': {'name': 'Violins', 'display_name': 'Violins', 'kind': 'music', 'family': 'strings', 'role': 'pad'},
                'audio': {'audio_base64': base64.b64encode(encode_wav(np.stack([tone, tone], axis=1), 48000)).decode()},
                'midi_base64': None, 'engine': {'name': 'symbolic'}, 'renderer': {}, 'windows': [], 'checks': {}, 'mix': {}, 'warnings': [],
                'analysis': {'tempo': 120.0, 'n': len(calls)}}
    return fake


def test_analysis_cache_invalidated_by_new_layers_and_speed_change(client, tmp_path, monkeypatch):
    from app import generate_part
    calls = []
    monkeypatch.setattr(generate_part, 'request_generate_part', _fake_service(calls))
    pid, h = _uploaded(client, tmp_path)
    def generate():
        r = client.post(f'/api/projects/{pid}/generate-preview', headers=h, json={'instrument': 'violins', 'prompt': 'add violins'})
        assert r.status_code == 202, r.text
        job = _wait(client, r.json()['candidate_id'], h); assert job['status'] == 'completed', job
        return r.json()['candidate_id'], job
    cid, job = generate(); assert job['provenance']['analysis_cached'] is False
    _, job = generate(); assert job['provenance']['analysis_cached'] is True  # same audio + layers: reused
    assert client.post(f'/api/candidates/{cid}/apply', headers=h).status_code == 200
    _, job = generate(); assert job['provenance']['analysis_cached'] is False  # a new layer: re-analysed
    _, job = generate(); assert job['provenance']['analysis_cached'] is True
    r = client.post(f'/api/projects/{pid}/fx-preview', headers=h, json={'targets': ['mix'], 'preset': 'slowed_reverb'})
    assert r.status_code == 202, r.text
    speed = _wait(client, r.json()['candidate_id'], h); assert speed['status'] == 'completed', speed
    assert client.post(f"/api/candidates/{r.json()['candidate_id']}/apply", headers=h).status_code == 200
    _, job = generate(); assert job['provenance']['analysis_cached'] is False  # speed change: re-analysed
    assert calls[-1]['analysis'] is None


def test_cached_call_still_uploads_vocal_and_bass_references(tmp_path, monkeypatch):
    from app import generate_part
    from app.media import encode_wav
    paths = {}
    for name in ('mix', 'vocals', 'bass', 'drums'):
        paths[name] = tmp_path / f'{name}.wav'
        paths[name].write_bytes(encode_wav(np.zeros((4800, 2), np.float32), 48000))
    sent = {}

    class R:
        def raise_for_status(self): pass
        def json(self): return {'ok': True}

    def post(url, files, data, timeout):
        sent['files'] = [(k, v[0]) for k, v in files]; sent['meta'] = json.loads(data['metadata']); return R()
    monkeypatch.setenv('SAM_AUDIO_SERVICE_URL', 'http://svc:1')
    monkeypatch.setattr(generate_part.httpx, 'post', post)
    stems = {k: paths[k] for k in ('vocals', 'bass', 'drums')}
    generate_part.request_generate_part(paths['mix'], stems, {'instrument': 'piano'}, analysis={'tempo': 120.0}, output_sample_rate=48000)
    keys = [k for k, _ in sent['files']]
    assert keys == ['audio', 'stem_vocals', 'stem_bass'] and sent['meta']['analysis'] == {'tempo': 120.0}
    generate_part.request_generate_part(paths['mix'], stems, {'instrument': 'piano'})
    assert sorted(k for k, _ in sent['files']) == ['audio', 'stem_bass', 'stem_drums', 'stem_vocals']


def test_generate_preview_warns_when_a_layer_is_soloed(client, tmp_path, monkeypatch):
    from app import generate_part
    calls = []
    monkeypatch.setattr(generate_part, 'request_generate_part', _fake_service(calls))
    pid, h = _uploaded(client, tmp_path)
    r = client.post(f'/api/projects/{pid}/generate-preview', headers=h, json={'instrument': 'violins', 'prompt': 'add violins'})
    cid = r.json()['candidate_id']; _wait(client, cid, h)
    assert client.post(f'/api/candidates/{cid}/apply', headers=h).status_code == 200
    from app.core import connect
    with connect() as db: db.execute("UPDATE stems SET solo=1 WHERE project_id=? AND name='Original audio'", (pid,))
    r = client.post(f'/api/projects/{pid}/generate-preview', headers=h, json={'instrument': 'piano', 'prompt': 'add a piano part'})
    assert r.status_code == 202 and r.json()['warnings']
    job = _wait(client, r.json()['candidate_id'], h)
    assert job['status'] == 'completed' and 'soloed' in json.dumps(job), job


def test_agent_route_precedence_insert_then_generate(client, tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    pid, h = _uploaded(client, tmp_path)
    ask = lambda prompt: client.post(f'/api/projects/{pid}/agent', headers=h, json={'prompt': prompt, 'stems': [], 'tracks': [], 'duration': 30}).json()
    for prompt in ("add a vine boom at 0:01", "add a sad trombone at the end"):  # the test clip is 2 s
        plan = ask(prompt); assert plan['kind'] == 'insert', (prompt, plan.get('message'))
    for prompt, instrument in (("add a piano part", "piano"), ("add violins", "violins"), ("add a funky bassline", "funky bassline"),
                               ("add an 80s synth arpeggio", "80s synth arpeggio"), ("add a piano at 0:01", "piano")):
        plan = ask(prompt)
        assert plan['kind'] == 'generate' and plan['generate']['instrument'] == instrument, prompt
    # A single timestamp is kept as an interval from there to the song end (the clip is 2 s), never dropped.
    assert ask("add horns at 0:01")['generate']['interval'] == {'start': 1.0, 'end': pytest.approx(2.0, abs=.01)}
    assert ask("add a piano at 0:12")['kind'] == 'clarify'  # past the end
    assert ask("add a piano when he says hello")['kind'] == 'clarify'
    assert ask("put the keys on the table").get('kind') != 'generate'  # falls through to the planner
    assert ask("replace the piano with violins")['kind'] == 'clarify'
    # Censorship comes first: a generate ask never pre-empts the profanity beep flow.
    for prompt in ("add violins and bleep the swearing", "censor the cuss words then add a piano part"):
        plan = ask(prompt); assert plan['kind'] == 'clarify' and ('censorship' in plan['message'] or 'profanity' in plan['message']), prompt
    # Main's object-appearance placements and stale-editor guard: generate never pre-empts an appearance ask,
    # and an instrument part runs before the refresh guard.
    assert ask("add a gong every time we see person 5")['kind'] != 'generate'
    assert ask("add a piano part")['kind'] == 'generate'
