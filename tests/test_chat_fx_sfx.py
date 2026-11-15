"""Cross-feature checks: chat FX (fx/duck/loudness) and library SFX inserts share one planner."""
import pytest

from tests.test_fx import STEMS, base_plan, fake_planner


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
