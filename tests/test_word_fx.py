"""Effects on spoken words: "bass boost the bad words", "add echo whenever he says McLovin" (app/word_fx.py)."""
import json

import numpy as np
import pytest

from test_fx import audio, base_plan, client, fake_planner, preview, split, upload  # noqa: F401  (client is a fixture)
from app.word_fx import find, parse, windows


@pytest.mark.parametrize('prompt,rest,profanity,words', [
    ('bass boost the bad words', 'bass boost', True, []),
    ('bass boost every time someone swears', 'bass boost', True, []),
    ('muffle the cursing', 'muffle', True, []),
    ('make the swear words louder', 'make louder', True, []),
    ('whenever he says a bad word add echo', 'add echo', True, []),
    ('add echo whenever he says McLovin', 'add echo', False, ['mclovin']),
    ('whenever he says McLovin add echo', 'add echo', False, ['mclovin']),
    ('deepen person 2 whenever he says McLovin or Fogell', 'deepen person 2', False, ['mclovin', 'fogell']),
    ('put reverb on the word "fake ID"', 'put reverb', False, ['fake id']),
    ("make every bad word and instance of the word 'mohammed' bass boosted", 'make bass boosted', True, ['mohammed']),
    ('make all the bad words and every instance of mohammed bass boosted', 'make bass boosted', True, ['mohammed']),
    ('bass boost every time he says mohammed and the bad words', 'bass boost', True, ['mohammed']),
    ('bass boost the word mohammed', 'bass boost', False, ['mohammed']),
    ('make the bad words and every Muhammad sound like a giant', 'make sound like a giant', True, ['muhammad']),
    ('pitch the bad words and Muhammad down 12 semitones', 'pitch down 12 semitones', True, ['muhammad']),
    ('bass boost the bad words and whenever anyone says Muhammad', 'bass boost', True, ['muhammad']),
    ('make every Mohammed squeaky', 'make squeaky', False, ['mohammed']),
    ('make the bad words and person 2 louder', 'make person 2 louder', True, []),
])
def test_parse_word_clauses(prompt, rest, profanity, words):
    assert parse(prompt) == (rest, {'profanity': profanity, 'words': words})


@pytest.mark.parametrize('prompt', ['add a vine boom when he says what', 'bleep the swearing', 'add reverb to the vocals', 'bass boost from 2 to 4 seconds'])
def test_other_asks_are_not_word_effects(prompt):
    assert parse(prompt) is None


def test_find_matches_profanity_names_and_split_names():
    words = [{'word': w, 'start': i * .5, 'end': i * .5 + .4} for i, w in enumerate(" Hey, McLovin! What the fuck, Mc Lovin is a damn name. McLovin's class".split())]
    found = find(words, {'profanity': True, 'words': ['mclovin']})
    assert [w['word'] for w in found] == ['McLovin!', 'fuck,', 'Mc', 'Lovin', 'damn', "McLovin's"]
    assert windows(found, 10)[1] == [pytest.approx(1.94), pytest.approx(3.46)]  # "fuck, Mc Lovin" merge into one window
    assert find(words, {'profanity': False, 'words': ['class']})[0]['word'] == 'class'
    assert find(words, {'profanity': False, 'words': ['classy']}) == []


@pytest.mark.parametrize('heard,asked,ok', [('Muhammad!', 'mohammed', True), ('Mohamad', 'mohammed', True), ('MacLevin', 'mclovin', True),
                                            ("McLovin's", 'mclovin', True), ('mohawk', 'mohammed', False), ('hammer', 'mohammed', False),
                                            ('Michael', 'mclovin', False), ('watt', 'what', False), ('shot', 'shit', False)])
def test_spoken_spellings_match(heard, asked, ok):
    words = [{'word': 'hey', 'start': 0, 'end': .2}, {'word': heard, 'start': .3, 'end': .6}]
    assert [w['word'] for w in find(words, {'profanity': False, 'words': [asked]})] == ([heard] if ok else [])


def test_a_short_word_after_the_name_is_not_marked():
    words = [{'word': w, 'start': i * .5, 'end': i * .5 + .4} for i, w in enumerate('Muhammad is a Mohamed and Mac Levin was MacLevin'.split())]
    assert [w['word'] for w in find(words, {'profanity': False, 'words': ['mohammed']})] == ['Muhammad', 'Mohamed']
    assert [w['word'] for w in find(words, {'profanity': False, 'words': ['mclovin']})] == ['Mac', 'Levin', 'MacLevin']


def test_blend_many_only_changes_the_windows():
    from app.fx import blend_many, layer_intervals
    sr = 1000; dry = np.zeros((3000, 1), np.float32); wet = np.ones((3000, 1), np.float32)
    y = blend_many(dry, wet, sr, [{'start': .5, 'end': 1.0}, {'start': 2.0, 'end': 2.5}])
    assert y[:490].max() == 0 and y[600:900].min() == 1 and y[1100:1990].max() == 0 and y[2100:2400].min() == 1 and y[2600:].max() == 0
    assert layer_intervals([{'start': .5, 'end': 1.0}, {'start': 2.0, 'end': 2.5}], 1.0, 1.2) == [{'start': 1.0, 'end': 1.2}]


def _transcript(monkeypatch, words):
    import app.censorship as c
    monkeypatch.setattr(c, 'available', lambda: True)
    monkeypatch.setattr(c, 'transcribe', lambda path, duration: (words, 'en'))


def test_agent_plans_the_effect_and_attaches_the_words(client, tmp_path, monkeypatch):
    pid, h = upload(client, tmp_path)
    split(client, pid, h)
    _, sent = fake_planner(monkeypatch, base_plan('fx', {'targets': ['mix'], 'preset': 'bass_boost'}))
    r = client.post(f'/api/projects/{pid}/agent', headers=h, json={'prompt': 'bass boost the bad words', 'stems': [], 'tracks': [], 'duration': 2})
    assert r.status_code == 200, r.text
    plan = r.json()
    assert plan['kind'] == 'fx' and plan['fx']['preset'] == 'bass_boost' and plan['fx']['on_words'] == {'profanity': True, 'words': []}
    assert json.loads(sent['input'])['prompt'] == 'bass boost'  # The planner never sees the word clause.
    # The planner reads "add echo" as a new sound and asks: the effect word still names the effect.
    _, sent = fake_planner(monkeypatch, base_plan('clarify', {}))
    r = client.post(f'/api/projects/{pid}/agent', headers=h, json={'prompt': 'add echo whenever he says McLovin', 'stems': [], 'tracks': [], 'duration': 2})
    plan = r.json()
    assert plan['kind'] == 'fx' and plan['fx']['preset'] == 'echo' and plan['fx']['on_words'] == {'profanity': False, 'words': ['mclovin']}, plan
    assert json.loads(sent['input'])['prompt'] == 'apply echo effect'
    # A layer named in the ask is the target; louder is a gain chain.
    _, sent = fake_planner(monkeypatch, base_plan('clarify', {}))
    r = client.post(f'/api/projects/{pid}/agent', headers=h, json={'prompt': 'make speech louder every time he swears', 'stems': [], 'tracks': [], 'duration': 2})
    plan = r.json()
    stems = {s['name']: s for s in client.get(f'/api/projects/{pid}/stems', headers=h).json()}
    assert plan['kind'] == 'fx' and plan['fx']['targets'] == [stems['speech']['id']] and plan['fx']['chain'][0]['type'] == 'gain', plan
    # A pitch amount is read from the words, without the planner; down keeps the formants.
    _, sent = fake_planner(monkeypatch, base_plan('clarify', {}))
    r = client.post(f'/api/projects/{pid}/agent', headers=h, json={'prompt': 'pitch the bad words and every Mohammed down 7 semitones', 'stems': [], 'tracks': [], 'duration': 2})
    plan = r.json()
    assert plan['kind'] == 'fx' and plan['fx']['on_words'] == {'profanity': True, 'words': ['mohammed']}, plan
    assert plan['fx']['chain'][0]['type'] == 'pitch' and plan['fx']['chain'][0]['params']['semitones'] == -7 and plan['fx']['chain'][0]['params']['formant'] == 1
    assert not sent
    # With no effect named at all, it asks.
    r = client.post(f'/api/projects/{pid}/agent', headers=h, json={'prompt': 'do something cool to the bad words with an effect', 'stems': [], 'tracks': [], 'duration': 2})
    assert r.json()['kind'] == 'clarify' and 'effect' in r.json()['message']


def test_word_effect_preview_targets_voice_layers_inside_word_windows(client, tmp_path, monkeypatch):
    pid, h = upload(client, tmp_path, seconds=2)
    stems = split(client, pid, h)  # 'music' and 'speech'
    music, speech = stems['music'], stems['speech']
    _transcript(monkeypatch, [{'word': 'hey', 'start': .1, 'end': .3}, {'word': 'shit', 'start': 1.0, 'end': 1.3}])
    result = preview(client, pid, h, {'targets': ['mix'], 'chain': [{'type': 'gain', 'params': {'gain': -24}}], 'on_words': {'profanity': True, 'words': []}})
    prov = result['provenance']
    assert prov['word_matches'] == [{'word': 'shit', 'start': 1.0, 'end': 1.3}]
    assert prov['item']['intervals'] == [{'start': .94, 'end': 1.36}] and 'bad words' in prov['item']['label']
    assert [t['name'] for t in prov['targets']] == ['speech']  # The whole-mix ask lands on the voice only.
    assert client.post(f"/api/candidates/{result['id']}/apply", headers=h).status_code == 200
    rows = {r['name']: r for r in client.get(f'/api/projects/{pid}/stems', headers=h).json()}
    assert rows['music']['fx'] is None and audio(client, h, rows['music']) == audio(client, h, music)
    import app.core as core
    with core.connect() as db: row = dict(db.execute('SELECT * FROM stems WHERE id=?', (speech['id'],)).fetchone())
    from app.fx import decode
    dry, sr = decode(json.loads(row['fx'])['dry_path']); wet, _ = decode(row['path'])
    def rms(x): return float(np.sqrt(np.mean(np.square(x)) + 1e-12))
    assert rms(wet[int(1.0 * sr):int(1.3 * sr)]) < rms(dry[int(1.0 * sr):int(1.3 * sr)]) * .1
    assert np.allclose(wet[:int(.8 * sr)], dry[:int(.8 * sr)], atol=1e-4) and np.allclose(wet[int(1.5 * sr):], dry[int(1.5 * sr):], atol=1e-4)


def test_word_effect_errors(client, tmp_path, monkeypatch):
    pid, h = upload(client, tmp_path)
    url = f'/api/projects/{pid}/fx-preview'
    import app.censorship as c
    monkeypatch.setattr(c, 'available', lambda: False)
    r = client.post(url, headers=h, json={'targets': ['mix'], 'preset': 'echo', 'on_words': {'profanity': True}})
    assert r.status_code == 422 and 'transcription' in r.json()['detail']
    _transcript(monkeypatch, [{'word': 'hello', 'start': .1, 'end': .3}])
    assert client.post(url, headers=h, json={'targets': ['mix'], 'preset': 'echo', 'on_words': {'words': []}}).status_code == 422
    assert client.post(url, headers=h, json={'targets': ['mix'], 'preset': 'echo', 'interval': {'start': .1, 'end': .5}, 'on_words': {'profanity': True}}).status_code == 422
    r = client.post(url, headers=h, json={'targets': ['mix'], 'preset': 'echo', 'on_words': {'words': ['McLovin']}})
    assert r.status_code == 202, r.text
    from test_fx import wait
    job = wait(client, '/api/candidates/' + r.json()['candidate_id'], h)
    assert job['status'] == 'failed' and 'did not hear' in job['error']
    assert client.get(f'/api/projects/{pid}/stems', headers=h).json() == []
