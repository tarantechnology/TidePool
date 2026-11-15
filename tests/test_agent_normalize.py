from app.agent import AgentRequest, normalize_edit, validate_operations


def body():
    return AgentRequest(prompt='x', stems=[{'id': 'd', 'label': 'dialogue'}, {'id': 't', 'label': 'thunder'}],
                        tracks=[{'stemId': 'd', 'gain': 1.0}, {'stemId': 't', 'gain': 0.5}], duration=49.4, has_project_stems=True)


def test_ranged_gain_becomes_a_curve_over_the_range():
    data = {'kind': 'edit', 'edit_scope': {'action': 'lower', 'interval': {'start': 2, 'end': 5}}, 'processing_time': None,
            'actions': [{'type': 'set_gain', 'stemId': 'd', 'value': 0.5, 'start': 2, 'end': 5}]}
    normalize_edit(data, body())
    a = data['actions'][0]
    assert a['type'] == 'set_automation' and data['edit_scope'] == {'action': 'keep', 'interval': None}
    pts = {p['t']: p['value'] for p in a['keyframes']}
    assert pts[2.0] == 0.5 and pts[5.0] == 0.5 and pts[0.0] == 1.0 and pts[49.4] == 1.0
    validate_operations(data['actions'], body())


def test_time_point_gets_three_seconds_and_respects_the_fader():
    data = {'kind': 'edit', 'edit_scope': {'action': 'keep', 'interval': None}, 'processing_time': 15,
            'actions': [{'type': 'set_gain', 'stemId': 't', 'value': 1.5}]}
    normalize_edit(data, body())
    pts = {p['t']: p['value'] for p in data['actions'][0]['keyframes']}
    assert pts[13.5] == 3.0 and pts[16.5] == 3.0 and data['processing_time'] is None   # 0.5 fader x 3.0 curve = 1.5
    validate_operations(data['actions'], body())


def test_plain_gain_stays_plain_and_separations_are_untouched():
    data = {'kind': 'edit', 'actions': [{'type': 'set_gain', 'stemId': 'd', 'value': 2.0}]}
    normalize_edit(data, body())
    assert data['actions'] == [{'type': 'set_gain', 'stemId': 'd', 'value': 2.0}]
    sep = {'kind': 'separate', 'processing_time': 3, 'actions': []}
    normalize_edit(sep, body())
    assert sep['processing_time'] == 3


def test_range_at_the_start_of_the_clip():
    data = {'kind': 'edit', 'actions': [{'type': 'set_gain', 'stemId': 'd', 'value': 0.0, 'start': 0, 'end': 3}]}
    normalize_edit(data, body())
    ts = [p['t'] for p in data['actions'][0]['keyframes']]
    assert ts == sorted(set(ts)) and ts[0] == 0.0
    validate_operations(data['actions'], body())


def test_model_curve_is_clamped_to_the_clip_and_ordered():
    data = {'kind': 'edit', 'actions': [{'type': 'set_automation', 'stemId': 'd', 'keyframes': [
        {'t': 0, 'value': 1}, {'t': 10, 'value': 6}, {'t': 5, 'value': 0.5}, {'t': 49.5, 'value': 1}]}]}
    normalize_edit(data, body())
    assert data['actions'][0]['keyframes'] == [{'t': 0.0, 'value': 1.0}, {'t': 5.0, 'value': 0.5}, {'t': 10.0, 'value': 5.0}, {'t': 49.4, 'value': 1.0}]
    validate_operations(data['actions'], body())
