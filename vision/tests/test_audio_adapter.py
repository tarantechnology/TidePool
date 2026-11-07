import numpy as np
import pytest
import torch
from soundcanvas_cv.audio_adapter import build_processor_batch, prepare_visual_inputs, PreparedVisual, mocked_handoff
from soundcanvas_cv.masks import encode_mask
from soundcanvas_cv.pipeline import process_media
from soundcanvas_cv.selection import select_at_point
from test_contract import Detector, video


class FakeProcessor:
    """MOCK: matches inspected source mask polarity, not model inference."""
    audio_sampling_rate = 48000
    def mask_videos(self, videos, masks):
        self.masks = masks
        return [v*m.eq(0) for v,m in zip(videos,masks)]
    def __call__(self, **kwargs): return kwargs


class MaskDetector(Detector):
    def detect(self, frame):
        objects = super().detect(frame)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8); mask[4:24,4:20] = 1
        objects[0].mask = encode_mask(mask)
        return objects


def selection(d):
    s = select_at_point(d, 0, .1, .2)
    s['interval']['end_seconds'] = d['processed_duration_seconds']
    return s


def test_video_decode_to_mock_processor(video, tmp_path):
    data = process_media(video, tmp_path/'out', MaskDetector(), 1000, .2)
    selected = selection(data)
    prepared = prepare_visual_inputs(video, data, selected)
    assert prepared.frames_rgb.shape == (6,3,48,64)
    assert prepared.foreground_masks.shape == (6,1,48,64)
    processor = FakeProcessor()
    audio = torch.zeros(2, round(prepared.end_seconds*48000))
    batch = build_processor_batch(prepared, audio, processor)
    assert batch['descriptions'] == ['']
    masked = batch['masked_videos'][0]
    assert masked[:,:,0,0].eq(0).all() and masked[:,:,10,10].gt(0).all()
    assert processor.masks[0][0,0,10,10] == 0
    assert mocked_handoff(selected)['inference_performed'] is False
    assert mocked_handoff(selected)['stems'] == []


def test_box_only_and_sparse_metadata_rejected(video, tmp_path):
    data = process_media(video, tmp_path/'out', Detector(), 1000, .2)
    selected = selection(data)
    with pytest.raises(ValueError, match='Instance mask'): prepare_visual_inputs(video, data, selected)
    assert prepare_visual_inputs(video, data, selected, allow_box_masks=True).mask_kind == 'coarse_bbox_experiment'
    data = process_media(video, tmp_path/'out', MaskDetector(), 5, .2)
    with pytest.raises(ValueError, match='Sparse metadata'): prepare_visual_inputs(video, data, selection(data))


def test_identity_dimensions_and_audio_alignment(video, tmp_path):
    data = process_media(video, tmp_path/'out', MaskDetector(), 1000, .2)
    selected = selection(data); selected['video_id'] = 'wrong'
    with pytest.raises(ValueError, match='another video'): prepare_visual_inputs(video, data, selected)
    prepared = PreparedVisual(np.ones((2,3,4,4),np.uint8), np.ones((2,1,4,4),np.uint8), 10, 0, .2, 'instance_segmentation')
    with pytest.raises(ValueError, match='cropped'): build_processor_batch(prepared, torch.zeros(2,100), FakeProcessor())
    selected = selection(data); selected['frame_width'] = 999
    with pytest.raises(ValueError, match='dimensions'): prepare_visual_inputs(video, data, selected)
    selected = selection(data); video.write_bytes(b'changed')
    with pytest.raises(ValueError, match='Source bytes'): prepare_visual_inputs(video, data, selected)


def test_rejects_other_analysis_and_tampered_detection(video, tmp_path):
    data=process_media(video,tmp_path/'out',MaskDetector(),1000,.2)
    selected=selection(data);selected['analysis_id']='different-run'
    with pytest.raises(ValueError,match='another video'):prepare_visual_inputs(video,data,selected)
    selected=selection(data);selected['bbox_xyxy']=[0,0,30,30]
    with pytest.raises(ValueError,match='observed detection'):prepare_visual_inputs(video,data,selected)
