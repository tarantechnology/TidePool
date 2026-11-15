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
        return {'part': {'name': 'Violins', 'display_name': 'Violins', 'kind': 'music', 'family': 'strings', 'role': 'counter_melody'},
                'audio': {'audio_base64': base64.b64encode(encode_wav(np.stack([tone, tone], axis=1), 48000)).decode()},
                'midi_base64': None, 'engine': {'name': 'symbolic'}, 'renderer': {'used': 'vsco:violins'}, 'windows': [[0, 2]],
                'checks': {'notes': {'chord_tone_ratio': 0.98}}, 'mix': {}, 'warnings': []}

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


# ---------------------------------------------------------------- agent-route precedence (matches frontend chatRoute.ts)
@pytest.mark.parametrize("prompt", ["add reverb to the vocals", "add chorus to the guitar", "add compression to the bass",
                                    "add more bass", "add a fade", "add a vine boom at 0:12", "add a laugh track",
                                    "put a riser before the drop", "add a sad trombone at the end"])
def test_rule_plan_leaves_fx_eq_and_sfx_alone(prompt):
    assert rule_plan(prompt) is None


def test_rule_plan_trims_time_tail():
    assert rule_plan("add a piano at 0:12")["instrument"] == "piano"


def test_agent_route_precedence_insert_then_generate(client, tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    pid, h = _uploaded(client, tmp_path)
    ask = lambda prompt: client.post(f'/api/projects/{pid}/agent', headers=h, json={'prompt': prompt, 'stems': [], 'tracks': [], 'duration': 30}).json()
    for prompt in ("add a vine boom at 0:01", "add a sad trombone at the end"):  # the test clip is 2 s
        plan = ask(prompt); assert plan['kind'] == 'insert', (prompt, plan.get('message'))
    for prompt, instrument in (("add a piano part", "piano"), ("add violins", "violins"), ("add a funky bassline", "funky bassline"),
                               ("add an 80s synth arpeggio", "80s synth arpeggio"), ("add a piano at 0:12", "piano")):
        plan = ask(prompt)
        assert plan['kind'] == 'generate' and plan['generate']['instrument'] == instrument, prompt
    assert ask("replace the piano with violins")['kind'] == 'clarify'
    # Censorship comes first: a generate ask never pre-empts the profanity beep flow.
    for prompt in ("add violins and bleep the swearing", "censor the cuss words then add a piano part"):
        plan = ask(prompt); assert plan['kind'] == 'clarify' and ('censorship' in plan['message'] or 'profanity' in plan['message']), prompt
    # Main's object-appearance placements and stale-editor guard: generate never pre-empts an appearance ask,
    # and an instrument part runs before the refresh guard.
    assert ask("add a gong every time we see person 5")['kind'] != 'generate'
    assert ask("add a piano part")['kind'] == 'generate'
