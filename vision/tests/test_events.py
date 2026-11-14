from copy import deepcopy
import numpy as np
import pytest

from soundcanvas_cv.events import associate, audio_events, motion_events, reviewed_anchors


def detections(two=False):
    data = dict(analysis_id='analysis', video_id='video', source_sha256='a'*64,
                fps=30., processed_duration_seconds=1., frames=[])
    for i, x in enumerate([.1, .2, .1]):
        objects = [dict(object_id=f'track:{tid}', track_id=tid, label='knife',
                        bbox_normalized=[x, .1, x+.1, .2], mask={'test': True},
                        detection_id=f'{i}:{tid}') for tid in ([1, 2] if two else [1])]
        data['frames'].append(dict(frame_index=i, timestamp_seconds=i/30, objects=objects))
    return data


def event():
    return dict(id='audio:0', start=0., end=.1, timestamp=1/30,
                label='unclassified transient', provenance={'component': 'test_waveform'})


def test_silence_does_not_create_object_sound():
    assert audio_events(np.zeros(48000), 48000) == []
    graph = associate(detections(), [])
    assert graph['visual_events'] and not graph['associations']


def test_actual_signal_transient_has_no_invented_semantic_label():
    x = np.zeros(48000)
    x[12000:12500] = np.random.default_rng(8).normal(0, .2, 500)
    events = audio_events(x, 48000)
    assert events and all(e['label'] == 'unclassified transient' for e in events)
    assert abs(events[0]['timestamp']-.25) < .06


def test_missing_observation_breaks_motion_evidence():
    data = detections()
    data['frames'][1]['frame_index'] = 5
    assert not motion_events(data)


def test_competing_visible_sources_abstain():
    graph = associate(detections(two=True), [event()], {'knife': ['knife chopping']})
    assert graph['associations']
    assert all(a['status'] == 'ambiguous' and a['association_confidence'] == 0 for a in graph['associations'])
    assert reviewed_anchors(graph, 'track:1', ['audio:0']) == []


def test_candidate_requires_explicit_review_and_provenance():
    graph = associate(detections(), [event()], {'knife': ['knife chopping', 'knife scraping']})
    a = graph['associations'][0]
    assert a['association_confidence'] < .5 and not a['provenance']['causality_verified']
    assert reviewed_anchors(graph, 'track:1', []) == []
    assert reviewed_anchors(graph, 'track:1', ['audio:0']) == [dict(token='+', start=0., end=.1)]
    bad = deepcopy(event()); bad['end'] = 5.
    with pytest.raises(ValueError, match='timeline'):
        associate(detections(), [bad])
