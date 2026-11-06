import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from jsonschema import validate
from soundcanvas_cv.pipeline import Detection, process_media
from soundcanvas_cv.selection import frame_at_time, select_at_point
from soundcanvas_cv.masks import encode_mask, decode_mask


class Detector:
    track = False
    def __init__(self): self.calls = self.resets = 0
    def reset(self): self.resets += 1
    def detect(self, frame):
        self.calls += 1
        return [Detection('track:1', 'person', .9, 0, [4,4,20,24], [4/64,4/48,20/64,24/48], 1)]


@pytest.fixture
def video(tmp_path):
    src = tmp_path/'input.avi'
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*'MJPG'), 29.97, (64,48))
    assert writer.isOpened()
    for _ in range(60): writer.write(np.full((48,64,3), 70, dtype=np.uint8))
    writer.release()
    return src


def check_schema(name, data):
    validate(data, json.loads((Path(__file__).resolve().parents[1]/'schemas'/f'{name}.schema.json').read_text()))


def test_fractional_sampling_and_truncation(video, tmp_path):
    detector = Detector()
    data = process_media(video, tmp_path/'out', detector, sample_fps=7, max_seconds=1)
    check_schema('detections', data)
    assert data['processed_frames'] == 30 and data['analyzed_frames'] == 7
    assert [f['frame_index'] for f in data['frames']] == [0,5,9,13,18,22,26]
    assert detector.calls == 7 and detector.resets == 1
    assert abs(data['processed_duration_seconds']-30/29.97) < 1e-8
    ids = [o['detection_id'] for f in data['frames'] for o in f['objects']]
    assert len(ids) == len(set(ids))
    for f in data['frames']:
        assert abs(f['timestamp_seconds']-f['frame_index']/data['fps']) < 1e-8
        assert f['timestamp_seconds'] < f['valid_until_seconds']
    cap = cv2.VideoCapture(str(tmp_path/'out/annotated.mp4')); count = 0
    while cap.read()[0]: count += 1
    cap.release()
    assert count == 30


def test_tracker_receives_all_frames_and_resets(video, tmp_path):
    detector = Detector(); detector.track = True
    a = process_media(video, tmp_path/'a', detector, 5, 1)
    assert a['inference_frames'] == 30 and a['analyzed_frames'] == 5
    process_media(video, tmp_path/'b', detector, 5, 1)
    assert detector.resets == 2


def test_sampling_above_native_fps(video, tmp_path):
    d = process_media(video, tmp_path/'out', Detector(), 1000, .2)
    assert d['processed_frames'] == d['analyzed_frames'] == 6


def test_freezes_selection_and_rejects_stale_times(video, tmp_path):
    d = process_media(video, tmp_path/'out', Detector(), 5, 1)
    selection = select_at_point(d, .1, .1, .2)
    check_schema('selection', selection)
    assert selection['timestamp_seconds'] == 0 and selection['clicked_at_seconds'] == .1
    assert selection['frame_width'] == 64 and selection['bbox_xyxy'] == [4,4,20,24]
    d['frames'][0]['objects'][0]['bbox_xyxy'][0] = 9
    assert selection['bbox_xyxy'][0] == 4
    assert frame_at_time(d, d['processed_duration_seconds']) is None
    assert frame_at_time(d, -1) is None and frame_at_time(d, float('nan')) is None
    assert select_at_point(d, .1, 2, .2) is None


@pytest.mark.parametrize('value', [0,-1,float('nan'),float('inf')])
def test_invalid_rates(video, tmp_path, value):
    for args in [{'sample_fps':value}, {'max_seconds':value}]:
        with pytest.raises(ValueError, match='finite and positive'):
            process_media(video, tmp_path/'out', Detector(), **args)


def test_corrupt_and_output_collision(tmp_path):
    src = tmp_path/'bad.mp4'; src.write_bytes(b'not a video')
    with pytest.raises(ValueError): process_media(src, tmp_path/'out', Detector())
    assert not (tmp_path/'out/detections.json').exists()
    src = tmp_path/'annotated.mp4'; src.write_bytes(b'x')
    with pytest.raises(ValueError, match='overwrite'): process_media(src, tmp_path, Detector())


@pytest.mark.parametrize('mask', [np.zeros((3,4)),np.ones((3,4)),np.eye(5),np.array([[1,0,1],[0,1,0]])])
def test_rle_roundtrip(mask):
    assert np.array_equal(decode_mask(encode_mask(mask)), mask)


def test_rejects_invalid_masks():
    with pytest.raises(ValueError): encode_mask([[2]])
    good = encode_mask([[1,0]])
    for patch in [{'counts':[3]}, {'counts':[-1,3]}, {'order':'F'}, {'size':[0,2]}]:
        with pytest.raises(ValueError): decode_mask({**good,**patch})
