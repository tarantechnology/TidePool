"""Cross-feature checks: chat FX (fx/duck/loudness) and library SFX inserts share one planner."""
import pytest

from tests.test_fx import STEMS, base_plan, client, fake_planner  # noqa: F401 (client is a fixture)


def test_schema_and_instructions_cover_both_features():
    from app.agent import INSTRUCTIONS, SCHEMA, _instructions
    kinds = SCHEMA['properties']['kind']['enum']
    assert {'edit', 'separate', 'cinematic', 'director', 'fx', 'duck', 'loudness', 'insert', 'clarify'} <= set(kinds)
    assert {'fx', 'inserts'} <= set(SCHEMA['required']) and set(SCHEMA['required']) == set(SCHEMA['properties'])
    text = _instructions()
    assert 'kind=fx' in text and 'kind=insert' in text and 'vine boom' in text.lower()
    assert 'effects applied by this tool' not in INSTRUCTIONS and 'sound library' in INSTRUCTIONS


def test_fx_plan_gets_null_inserts(monkeypatch):
    agent, _ = fake_planner(monkeypatch, {**base_plan('fx', {'targets': ['v1'], 'preset': 'reverb'}), 'inserts': None})
    result = agent.plan(agent.AgentRequest(prompt='add reverb to the vocals', stems=STEMS, tracks=[], duration=30, has_project_stems=True))
    assert result['kind'] == 'fx' and result['inserts'] is None and result['fx']['targets'] == ['v1']


def test_fx_plan_carrying_inserts_is_rejected(monkeypatch):
    insert = {'sfx_query': 'vine boom', 'anchor': {'type': 'time', 'value': 12, 'relation': 'at', 'occurrence': None}, 'offset': None, 'gain_db': None, 'duck': None}
    agent, _ = fake_planner(monkeypatch, {**base_plan('fx', {'targets': ['v1'], 'preset': 'reverb'}), 'inserts': [insert]})
    with pytest.raises(agent.AgentError):
        agent.plan(agent.AgentRequest(prompt='add reverb to the vocals at 12', stems=STEMS, tracks=[], duration=30, has_project_stems=True))


def test_insert_plan_gets_null_fx(monkeypatch):
    insert = {'sfx_query': 'vine boom', 'anchor': {'type': 'time', 'value': 12, 'relation': 'at', 'occurrence': None}, 'offset': None, 'gain_db': None, 'duck': None}
    plan = {**base_plan('insert', {}), 'fx': None, 'inserts': [insert]}
    agent, _ = fake_planner(monkeypatch, plan)
    result = agent.plan(agent.AgentRequest(prompt='the 0:12 moment needs a boom', stems=STEMS, tracks=[], duration=30, has_project_stems=True))
    assert result['kind'] == 'insert' and result['fx'] is None and result['inserts'][0]['sfx_query'] == 'vine boom'


@pytest.mark.parametrize('prompt', ['add reverb to the vocals', 'make the vocals sound like a phone', 'put an echo on the drums',
                                    'duck the music under the voice', 'normalize for youtube', 'add some chorus to the guitar'])
def test_effect_asks_skip_the_offline_insert_parser(prompt):
    from app.sfx_plan import rule_plan
    assert rule_plan(prompt) is None


def test_insert_asks_stay_with_the_insert_parser():
    from app.sfx_plan import rule_plan
    boom = rule_plan('add a vine boom at 0:12')
    assert boom['kind'] == 'insert' and boom['inserts'][0]['anchor']['value'] == 12
    riser = rule_plan('add a riser before the drop')
    # Owned by the insert flow: it asks for the moment rather than guessing (and is never an effect).
    assert riser['kind'] == 'clarify' and riser.get('sfx_needs_time') is True and 'riser' in riser['message']


# ---- Routing precedence after merging main's censorship feature: profanity beeps are never library inserts.

@pytest.mark.parametrize('prompt', ['bleep the swearing', 'censor the cuss words', 'beep out curse words', 'put a bleep over the swear words', 'mute the profanity'])
def test_profanity_beeps_skip_the_insert_parser(prompt):
    from app.sfx_plan import requests_censorship, rule_plan
    assert requests_censorship(prompt) and rule_plan(prompt) is None


@pytest.mark.parametrize('prompt', ['add a censor bleep at 0:12', "don't censor the swearing", 'make the beep louder'])
def test_censorship_intent_needs_an_explicit_profanity_ask(prompt):
    from app.sfx_plan import requests_censorship
    assert not requests_censorship(prompt)


def test_explicit_censor_bleep_insert_still_uses_the_library():
    from app.sfx_plan import rule_plan
    plan = rule_plan('add a censor bleep at 0:12')
    assert plan['kind'] == 'insert' and plan['inserts'][0]['sfx_id'] == 'censor_bleep'


def test_agent_route_defers_profanity_beeps_to_censorship(client, monkeypatch):
    import app.agent as agent
    called = []
    monkeypatch.setattr(agent, 'plan', lambda *a, **k: called.append(1))
    p = client.post('/api/projects').json(); pid = p['project_id']; h = {'Authorization': 'Bearer ' + p['access_token']}
    r = client.post(f'/api/projects/{pid}/agent', headers=h, json={'prompt': 'bleep the swearing', 'stems': [], 'tracks': [], 'duration': 10, 'has_project_stems': False})
    assert r.status_code == 200 and r.json()['kind'] == 'clarify' and 'censorship' in r.json()['message'] and not called


@pytest.mark.parametrize('prompt', ['drop the music at 0:30', 'add silence at the start'])
def test_non_sounds_fall_through_to_the_planner(prompt):
    from app.sfx_plan import rule_plan
    assert rule_plan(prompt) is None


@pytest.mark.parametrize('prompt,expected', [
    ('make the vocals higher pitch', 3), ('lower the pitch of the bass', -3), ('make the vocals way higher pitched', 7),
    ('lower the pitch a lot', -7), ('raise the pitch slightly', 1), ('lower the pitch a little', -1),
    ('raise the pitch by 5 semitones', 5), ('pitch the vocals down 2 semitones', -2), ('make the vocals louder', None)])
def test_pitch_defaults(prompt, expected):
    from app.fx_schema import pitch_shift
    assert pitch_shift(prompt) == expected


def test_planner_pitch_without_amount_gets_the_default():
    from app.fx_schema import FX_INSTRUCTIONS, validate_plan
    assert '+3' in FX_INSTRUCTIONS and 'never needs a clarify' in FX_INSTRUCTIONS
    base = {'targets': ['v1'], 'preset': None, 'interval': None, 'adjust': None, 'speed': None}
    out = validate_plan('fx', {**base, 'chain': [{'type': 'pitch', 'params': {'semitones': None}}]}, STEMS, 30, 'make the vocals higher pitch')
    assert out['chain'][0]['params']['semitones'] == 3
    out = validate_plan('fx', {**base, 'chain': None}, STEMS, 30, 'lower the pitch of the vocals a lot')
    assert out['chain'][0]['params']['semitones'] == -7
    out = validate_plan('fx', {**base, 'chain': [{'type': 'pitch', 'params': {'semitones': 4}}]}, STEMS, 30, 'pitch the vocals up 4 semitones')
    assert out['chain'][0]['params']['semitones'] == 4
