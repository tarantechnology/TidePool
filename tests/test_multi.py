"""Several edits in one chat prompt (app/multi.py): planner schema + validation, clause splitting,
the concurrent candidate build, atomic apply/undo and the API end to end with the fake adapter."""
import base64
import json
import threading
import time

import numpy as np
import pytest

from app import fx_schema as S
from app import multi
from tests.test_fx import client, fake_planner, split, upload, wait, walk  # noqa: F401  (fixtures)

NULL_FX = {'targets': ['mix'], 'preset': None, 'chain': None, 'interval': None, 'adjust': None, 'speed': None,
           'duck_source': None, 'duck_depth_db': None, 'loudness_lufs': None}
STEMS = [{'id': 'v1', 'label': 'Vocals', 'canonical_label': 'vocals', 'duration': 30},
         {'id': 'd1', 'label': 'Drums', 'canonical_label': 'drums', 'duration': 30},
         {'id': 'b1', 'label': 'Bass', 'canonical_label': 'bass', 'duration': 30}]
GEN = {'instrument': 'piano', 'prompt': 'add a piano part', 'role': None, 'sections': None, 'interval': None, 'density': None, 'engine': 'auto'}


def step(kind, label='', fx=None, inserts=None, generate=None, actions=None):
    return {'kind': kind, 'label': label, 'fx': {**NULL_FX, **fx} if fx is not None else None, 'inserts': inserts,
            'generate': generate, 'actions': actions or []}


def insert(query='vine boom', t=12.0):
    return {'sfx_query': query, 'anchor': {'type': 'time', 'value': t, 'relation': 'at', 'occurrence': None}, 'offset': None, 'gain_db': None, 'duck': None}


def plan_of(steps):
    return {'kind': 'multi', 'guidance': {'anchors': []}, 'edit_scope': {'action': 'keep', 'interval': None}, 'description': None, 'remove': False,
            'processing_time': None, 'separation_mode': None, 'stems': None, 'actions': [], 'use_visual': False, 'fx': None, 'inserts': None,
            'message': 'ok', 'steps': steps}


PROMPT = 'add reverb to the vocals, make the drums sound like a phone, pitch the bass down 2 semitones, add a piano part and put a vine boom at 0:12'
EXAMPLE = [step('fx', 'Reverb on vocals', fx={'targets': ['v1'], 'preset': 'reverb'}),
           step('fx', 'Drums like a phone', fx={'targets': ['d1'], 'preset': 'telephone'}),
           step('fx', 'Bass down 2 semitones', fx={'targets': ['b1'], 'chain': [{'type': 'pitch', 'params': {k: None for k in S.PARAM_KEYS} | {'semitones': -2}}]}),
           step('generate', 'Add a piano part', generate=GEN),
           step('insert', 'Vine boom at 0:12', inserts=[insert()])]


# ---------------------------------------------------------------- planner schema + validation

def test_schema_is_strict_and_has_steps():
    from app.agent import SCHEMA, INSTRUCTIONS
    walk(SCHEMA)
    assert 'multi' in SCHEMA['properties']['kind']['enum'] and 'steps' in SCHEMA['required']
    item = SCHEMA['properties']['steps']['anyOf'][0]['items']
    assert item['properties']['kind']['enum'] == multi.STEP_KINDS
    assert 'kind=multi' in INSTRUCTIONS and 'split the request' in INSTRUCTIONS
    assert len(json.dumps(SCHEMA)) < 60000


def test_planner_multi_is_validated_per_step(monkeypatch):
    agent, sent = fake_planner(monkeypatch, plan_of(EXAMPLE))
    body = agent.AgentRequest(prompt=PROMPT, stems=STEMS, tracks=[], duration=30, has_project_stems=True)
    result = agent.plan(body)
    assert result['kind'] == 'multi' and [s['kind'] for s in result['steps']] == ['fx', 'fx', 'fx', 'generate', 'insert']
    assert [s['label'] for s in result['steps']][:2] == ['Reverb on vocals', 'Drums like a phone']
    assert result['steps'][2]['fx']['chain'][0]['params']['semitones'] == -2
    assert result['steps'][4]['inserts'][0]['anchor']['value'] == 12
    assert sent['text']['format']['strict'] is True


@pytest.mark.parametrize('steps,needle', [
    ([EXAMPLE[0], step('fx', 'Phone on piano', fx={'targets': ['nope'], 'preset': 'telephone'})], 'Edit 2 ("Phone on piano")'),
    ([EXAMPLE[0], step('fx', 'Nightcore', fx={'targets': ['mix'], 'preset': 'nightcore'})], 'Speed changes'),
    ([EXAMPLE[0], step('fx', 'More reverb', fx={'targets': ['v1'], 'adjust': 'more'})], 'on its own'),
    ([EXAMPLE[0], step('insert', 'Boom', inserts=[insert(t=99)])], 'Edit 2'),
    ([EXAMPLE[0], step('insert', 'Boom', inserts=[insert(t=5)])], 'Edit 2'),   # 5 was never said
    ([EXAMPLE[0], step('edit', 'Follow', actions=[{'type': 'follow_intensity', 'stemId': 'd1', 'sourceStemIds': ['v1']}])], 'on its own'),
    ([EXAMPLE[0], step('generate', 'Add', generate={**GEN, 'role': 'kazoo'})], 'Edit 2'),
    ([step('fx', 'x', fx={'targets': ['v1'], 'preset': 'reverb'})] * 9, 'split the request'),
    ([step('generate', 'g', generate=GEN)] * 4, 'split the request'),
])
def test_invalid_steps_become_a_clarify_naming_the_step(monkeypatch, steps, needle):
    agent, _ = fake_planner(monkeypatch, plan_of(steps))
    result = agent.plan(agent.AgentRequest(prompt=PROMPT, stems=STEMS, tracks=[], duration=30, has_project_stems=True))
    assert result['kind'] == 'clarify' and needle in result['message'], result


def test_steps_on_a_single_kind_are_rejected(monkeypatch):
    agent, _ = fake_planner(monkeypatch, {**plan_of(EXAMPLE[:1]), 'kind': 'fx', 'fx': {**NULL_FX, 'targets': ['v1'], 'preset': 'reverb'}})
    with pytest.raises(agent.AgentError):
        agent.plan(agent.AgentRequest(prompt='add reverb', stems=STEMS, tracks=[], duration=30))


@pytest.mark.parametrize('prompt,count', [
    (PROMPT, 5),
    ('add reverb to the vocals and make the drums sound like a phone', 2),
    ('add reverb and echo to the vocals', 1),
    ('add reverb to the drums and bass', 1),
    ('slowed and reverb', 1),
    ("put a vine boom when he says 'what and add' then mute the drums", 2),
    ('duck the music under the voice; normalize for youtube', 2),
    ('make the vocals sound like a robot, then add a laugh track at the end', 2),
])
def test_split_clauses(prompt, count):
    assert len(multi.split_clauses(prompt)) == count, multi.split_clauses(prompt)


# ---------------------------------------------------------------- the candidate (fake adapter)

def multi_preview(client, pid, h, steps, expect=202):
    r = client.post(f'/api/projects/{pid}/multi-preview', headers=h, json={'steps': steps, 'request': 'test'})
    assert r.status_code == expect, r.text
    return r


def stems_by_name(client, pid, h):
    return {r['name']: r for r in client.get(f'/api/projects/{pid}/stems', headers=h).json()}


def slow_renderer(monkeypatch, delay=.4):
    real = multi.F.render_layer; calls = []; lock = threading.Lock()
    def render_layer(row, rec, items, new_file):
        began = time.monotonic(); time.sleep(delay)
        try: return real(row, rec, items, new_file)
        finally:
            with lock: calls.append({'row': row['name'], 'labels': [i['label'] for i in items], 'start': began, 'end': time.monotonic()})
    monkeypatch.setattr(multi.F, 'render_layer', render_layer)
    return calls


def test_layers_render_concurrently_and_same_layer_steps_chain_in_order(client, tmp_path, monkeypatch):
    calls = slow_renderer(monkeypatch)
    pid, h = upload(client, tmp_path)
    stems = split(client, pid, h); music, speech = stems['music'], stems['speech']
    steps = [step('fx', 'Church on music', fx={'targets': [music['id']], 'preset': 'church'}),
             step('fx', 'Phone on speech', fx={'targets': [speech['id']], 'preset': 'telephone'}),
             step('fx', 'Echo on music', fx={'targets': [music['id']], 'preset': 'echo'}),
             step('edit', 'Speech quieter', actions=[{'type': 'set_gain', 'stemId': speech['id'], 'value': .5}])]
    r = multi_preview(client, pid, h, steps)
    assert [s['label'] for s in r.json()['steps']] == ['Church on music', 'Phone on speech', 'Echo on music', 'Speech quieter']
    cid = r.json()['candidate_id']
    stages = set()
    for _ in range(400):
        job = client.get('/api/candidates/' + cid, headers=h).json()
        if job['stage']: stages.add(job['stage'])
        if job['status'] in ('completed', 'failed'): break
        time.sleep(.02)
    assert job['status'] == 'completed', job
    assert any(s.startswith('Applying ') and s.endswith(' of 4 edits…') for s in stages), stages
    # One render per layer; the music layer's two steps chain in prompt order.
    assert sorted((c['row'], tuple(c['labels'])) for c in calls) == [('music', ('Church', 'Echo')), ('speech', ('Telephone',))]
    a, b = calls
    assert a['start'] < b['end'] and b['start'] < a['end'], 'the two layers should render at the same time'
    prov = job['provenance']
    assert prov['kind'] == 'multi' and [s['label'] for s in prov['edits']][0] == 'Church on music' and prov['concurrency']['tasks'] == 2
    # Apply is atomic: both chip lists and the gain land together; undo restores everything.
    before = {k: client.get(v['audio_url'], headers=h).content for k, v in stems.items()}
    assert client.post(f'/api/candidates/{cid}/apply', headers=h).status_code == 200
    rows = stems_by_name(client, pid, h)
    assert [c['label'] for c in rows['music']['fx']['items']] == ['Church', 'Echo']
    assert [c['label'] for c in rows['speech']['fx']['items']] == ['Telephone'] and rows['speech']['volume'] == .5
    assert client.post(f'/api/candidates/{cid}/apply', headers=h).status_code == 409
    assert client.post(f'/api/candidates/{cid}/undo', headers=h).status_code == 200
    rows = stems_by_name(client, pid, h)
    assert rows['music']['fx'] is None and rows['speech']['volume'] == speech['volume']
    assert {k: client.get(v['audio_url'], headers=h).content for k, v in rows.items()} == before


def fake_generate(monkeypatch, fail=False):
    from app import generate_part
    from app.media import encode_wav, probe
    def service(mix_path, stems, plan, seed=0, **kw):
        if fail: raise RuntimeError('the music service is down')
        n = round(probe(mix_path)['duration'] * 48000)
        tone = (0.1 * np.sin(2 * np.pi * 330 * np.arange(n) / 48000)).astype(np.float32)
        return {'part': {'name': 'Piano', 'display_name': 'Piano', 'family': 'keys'}, 'engine': {'name': 'symbolic'},
                'audio': {'audio_base64': base64.b64encode(encode_wav(np.stack([tone, tone], axis=1), 48000)).decode()}, 'warnings': []}
    monkeypatch.setattr(generate_part, 'request_generate_part', service)


def test_generate_insert_edit_and_loudness_in_one_preview(client, tmp_path, monkeypatch):
    fake_generate(monkeypatch)
    pid, h = upload(client, tmp_path, seconds=3)
    stems = split(client, pid, h); music, speech = stems['music'], stems['speech']
    steps = [step('fx', 'Robot speech', fx={'targets': [speech['id']], 'preset': 'robot'}),
             step('generate', 'Add a piano part', generate=GEN),
             step('insert', 'Boom at 1s', inserts=[insert(t=1.0)]),
             step('edit', 'Mute music 2-2.5', actions=[{'type': 'mute_range', 'stemId': music['id'], 'start': 2, 'end': 2.5}]),
             step('loudness', 'TikTok loudness', fx={'loudness_lufs': -14})]
    cid = multi_preview(client, pid, h, steps).json()['candidate_id']
    job = wait(client, '/api/candidates/' + cid, h)
    assert job['status'] == 'completed', job
    assert job['provenance']['generated'][0]['name'] == 'Piano' and job['provenance']['placements'][0]['start'] == pytest.approx(1.0)
    assert client.post(f'/api/candidates/{cid}/apply', headers=h).status_code == 200
    rows = stems_by_name(client, pid, h)
    assert {'music', 'speech', 'Piano'} <= set(rows) and any(r['role'] == 'sfx' for r in rows.values())
    assert rows['music']['mute_ranges'] == [[2, 2.5]]
    assert [c['label'] for c in rows['speech']['fx']['items']] == ['Robot', 'Loudness -14 LUFS']
    assert any(c['label'] == 'Loudness -14 LUFS' for c in rows['Piano']['fx']['items'])
    assert client.post(f'/api/candidates/{cid}/undo', headers=h).status_code == 200
    assert set(stems_by_name(client, pid, h)) == {'music', 'speech'}


def test_a_failing_step_fails_the_whole_preview_and_names_it(client, tmp_path, monkeypatch):
    fake_generate(monkeypatch, fail=True)
    pid, h = upload(client, tmp_path)
    stems = split(client, pid, h)
    steps = [step('fx', 'Church on music', fx={'targets': [stems['music']['id']], 'preset': 'church'}),
             step('generate', 'Add a piano part', generate=GEN)]
    job = wait(client, '/api/candidates/' + multi_preview(client, pid, h, steps).json()['candidate_id'], h)
    assert job['status'] == 'failed' and 'Edit 2 ("Add a piano part")' in job['error'] and 'down' in job['error']
    assert all(r['fx'] is None for r in stems_by_name(client, pid, h).values())


def test_route_rejects_bad_steps_with_the_step_named(client, tmp_path):
    pid, h = upload(client, tmp_path)
    stems = split(client, pid, h)
    good = step('fx', 'Church', fx={'targets': [stems['music']['id']], 'preset': 'church'})
    cases = [([good, step('fx', 'Bad', fx={'targets': ['nope'], 'preset': 'church'})], 'Edit 2 ("Bad")'),
             ([good, step('fx', 'Fast', fx={'targets': ['mix'], 'preset': 'nightcore'})], 'Speed'),
             ([good, step('separate', 'Split')], 'on its own'),
             ([good, step('insert', 'Kazoo', inserts=[insert('xyzzy kazoo nonsense', 1)])], 'Edit 2'),
             ([good] * 9, 'split the request')]
    for steps, needle in cases:
        r = multi_preview(client, pid, h, steps, expect=422)
        assert needle in r.json()['detail'], r.text


def test_agent_route_sends_compound_asks_to_the_planner(client, tmp_path, monkeypatch):
    pid, h = upload(client, tmp_path)
    stems = split(client, pid, h); music, speech = stems['music'], stems['speech']
    fake_planner(monkeypatch, plan_of([step('fx', 'Church on music', fx={'targets': [music['id']], 'preset': 'church'}),
                                       step('insert', 'Vine boom at 1', inserts=[insert(t=1)])]))
    # Without the compound check the insert fast path would take only the vine boom.
    plan = client.post(f'/api/projects/{pid}/agent', headers=h, json={'prompt': 'add reverb to the music and put a vine boom at 1s',
                                                                       'stems': [], 'tracks': [], 'duration': 2}).json()
    assert plan['kind'] == 'multi', plan
    assert plan['steps'][1]['inserts'][0]['sfx_id'] == 'vine_boom'
    job = wait(client, '/api/candidates/' + multi_preview(client, pid, h, plan['steps']).json()['candidate_id'], h)
    assert job['status'] == 'completed', job
