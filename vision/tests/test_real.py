"""Opt-in actual pretrained inference, not fake detectors."""
import ast
import json
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from soundcanvas_cv.pipeline import process_media
from soundcanvas_cv.yolo_adapter import YOLODetector
from soundcanvas_cv.masks import decode_mask
from soundcanvas_cv.selection import select_at_point
from soundcanvas_cv.audio_adapter import prepare_visual_inputs
from test_contract import check_schema

ROOT=Path(__file__).resolve().parents[1]
pytestmark=[pytest.mark.real,pytest.mark.skipif(os.getenv('SC_CV_REAL_TESTS')!='1',reason='Set SC_CV_REAL_TESTS=1 with demo and weights available')]


@pytest.mark.parametrize('weights,segmentation',[('yolo26n.pt',False),('yolo26n-seg.pt',True)])
def test_pretrained_yolo_mp4_and_selection(tmp_path,weights,segmentation):
    model=YOLODetector(str(ROOT/weights),device='cpu')
    video=ROOT/'examples/media/pedestrians.mp4'
    data=process_media(video,tmp_path/'out',model,10,1)
    check_schema('detections',data)
    assert data['inference_frames']==data['analyzed_frames']==data['processed_frames']==10
    assert all(f['objects'] for f in data['frames'])
    person=next(o for o in data['frames'][0]['objects'] if o['label']=='person')
    assert person['track_id'] is not None
    x1,y1,x2,y2=person['bbox_normalized']
    selected=select_at_point(data,0,(x1+x2)/2,(y1+y2)/2)
    check_schema('selection',selected)
    assert selected['detection_id']==person['detection_id']
    if segmentation:
        assert decode_mask(person['mask']).shape==(576,768)
        prepared=prepare_visual_inputs(video,data,selected)
        assert prepared.frames_rgb.shape==(1,3,576,768)
        assert prepared.foreground_masks.any()
    else: assert person['mask'] is None
    # Reusing one detector for a new video resets the tracker; IDs start afresh.
    again=process_media(video,tmp_path/'again',model,10,.2)
    assert again['frames'][0]['objects'][0]['track_id']==data['frames'][0]['objects'][0]['track_id']


def test_source_polarity_against_official_processor_if_downloaded():
    path=ROOT/'.research/sam-audio-sam_audio_processor.py'
    if not path.exists():pytest.skip('Optional inspected official source not downloaded; see docs/SAM.md')
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Processor')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='mask_videos')
    namespace={'torch':torch,'List':list}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),namespace)
    # Execute the actual upstream mask method with tensors. No model or fake processor.
    rgb=torch.full((2,3,4,5),255,dtype=torch.uint8)
    foreground=torch.zeros((2,1,4,5),dtype=torch.uint8);foreground[:,:,1:3,1:3]=1
    masked=namespace['mask_videos'](None,[rgb],[1-foreground])[0]
    assert masked[:,:,1:3,1:3].eq(255).all() and masked[:,:,0,0].eq(0).all()
