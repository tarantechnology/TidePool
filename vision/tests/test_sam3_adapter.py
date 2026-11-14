import json
from pathlib import Path

import cv2
import jsonschema
import numpy as np
import pytest

from soundcanvas_cv.masks import decode_mask
from soundcanvas_cv.pipeline import process_media
from soundcanvas_cv.sam3_adapter import SAM3VideoDetector, extract_sam3


def output(oid=7, shape=(24, 32)):
    mask = np.zeros(shape, bool)
    mask[8:10, 5:22] = 1
    return dict(out_obj_ids=np.array([oid]), out_binary_masks=mask[None], out_probs=np.array([.9]))


def test_native_thin_mask_and_identity():
    d = extract_sam3(output(), (24, 32), 'knife')[0]
    assert d.bbox_xyxy == [5, 8, 22, 10]
    assert decode_mask(d.mask).sum() == 34
    assert d.object_id == 'track:7' and d.track_id == 7
    with pytest.raises(ValueError, match='dimensions'):
        extract_sam3(output(), (12, 16), 'knife')
    with pytest.raises(ValueError, match='duplicate'):
        extract_sam3(dict(out_obj_ids=[7, 7], out_binary_masks=np.ones((2, 24, 32)), out_probs=[.9, .8]), (24, 32), 'knife')
    removed = output(); removed['out_probs'] = [-1e4]
    assert extract_sam3(removed, (24, 32), 'knife') == []
    removed['out_probs'] = [float('nan')]
    with pytest.raises(ValueError, match='invalid score'):
        extract_sam3(removed, (24, 32), 'knife')


class Predictor:
    def __init__(self, reorder=False):
        self.requests = []
        self.reorder = reorder

    def handle_request(self, request):
        self.requests.append(request)
        return dict(session_id='s')

    def handle_stream_request(self, request):
        self.requests.append(request)
        for i in range(3):
            # A real analyzed absence must stay empty, with no reused previous mask.
            result = output() if i != 1 else dict(out_obj_ids=[], out_binary_masks=[], out_probs=[])
            yield dict(frame_index=i+int(self.reorder), outputs=result)


def video(tmp_path):
    path = tmp_path/'test.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 24))
    for _ in range(3):
        writer.write(np.zeros((24, 32, 3), np.uint8))
    writer.release()
    checkpoint = tmp_path/'weights.pt'
    checkpoint.write_bytes(b'fake-for-adapter-contract-only')
    return path, checkpoint


def test_video_contract_absence_and_cleanup(tmp_path):
    path, checkpoint = video(tmp_path)
    predictor = Predictor()
    detector = SAM3VideoDetector(path, checkpoint, 'knife', predictor=predictor)
    try:
        result = process_media(path, tmp_path/'out', detector, sample_fps=1000)
    finally:
        detector.close()
    schema = json.loads((Path(__file__).parents[1]/'schemas/detections.schema.json').read_text())
    jsonschema.validate(result, schema)
    assert result['frames'][1]['objects'] == []
    assert result['frames'][2]['objects'][0]['track_id'] == 7
    assert predictor.requests[-1]['type'] == 'close_session'
    start = next(r for r in predictor.requests if r['type'] == 'start_session')
    assert start['offload_video_to_cpu'] and not start['offload_state_to_cpu']


def test_reordered_outputs_fail_instead_of_shifting_masks(tmp_path):
    path, checkpoint = video(tmp_path)
    predictor = Predictor(reorder=True)
    detector = SAM3VideoDetector(path, checkpoint, 'knife', predictor=predictor)
    try:
        with pytest.raises(ValueError, match='shifted masks'):
            process_media(path, tmp_path/'out', detector)
    finally:
        detector.close()
    assert not (tmp_path/'out/detections.json').exists()
    assert predictor.requests[-1]['type'] == 'close_session'


def test_no_implicit_checkpoint_download(tmp_path):
    with pytest.raises(FileNotFoundError, match='approved'):
        SAM3VideoDetector('video.mp4', tmp_path/'missing.pt', 'knife')
