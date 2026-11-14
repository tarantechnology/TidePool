import pytest

from soundcanvas_cv.sam2_adapter import SAM2VideoDetector


def test_sam2_does_not_invent_text_detections(tmp_path):
    checkpoint = tmp_path/'weights.pt'
    checkpoint.write_bytes(b'contract-only')
    with pytest.raises(ValueError, match='user seeds'):
        SAM2VideoDetector('clip.mp4', checkpoint, [])
    seed = dict(object_id=1, label='knife', frame_index=0, points=[[.5, .5]], point_labels=[1])
    detector = SAM2VideoDetector('clip.mp4', checkpoint, [seed])
    assert detector.describe()['label_provenance'] == 'user seed; no automatic object detection'
    seed['points'] = [[float('nan'), .5]]
    with pytest.raises(ValueError, match='normalized points'):
        SAM2VideoDetector('clip.mp4', checkpoint, [seed])
