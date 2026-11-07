import json
import cv2
import numpy as np
import pytest
from soundcanvas_cv.pipeline import Detection, process_media, pick_object_at_point

class FakeDetector:
    def __init__(self): self.calls=0
    def detect(self, frame):
        self.calls+=1
        h,w=frame.shape[:2]
        return [Detection('track:42','truck',.92,7,[20,20,110,140],
                          [20/w,20/h,110/w,140/h],42)]

def test_video_pipeline_and_export(tmp_path):
    path=tmp_path/'video.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(320,180))
    assert writer.isOpened()
    for _ in range(10): writer.write(np.full((180,320,3),70,np.uint8))
    writer.release()
    fake=FakeDetector()
    output=process_media(path,tmp_path/'output',fake,sample_fps=5,max_seconds=3)
    assert output['processed_frames']==10
    assert output['analyzed_frames']==5 and fake.calls==5
    assert output['frames'][1]['timestamp_seconds']==.2
    assert output['frames'][0]['objects'][0]['object_id']=='track:42'
    assert json.loads((tmp_path/'output/detections.json').read_text())==output
    cap=cv2.VideoCapture(str(tmp_path/'output/annotated.mp4'))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==10
    cap.release()

def test_image_pipeline(tmp_path):
    p=tmp_path/'image.png'
    assert cv2.imwrite(str(p),np.zeros((180,320,3),np.uint8))
    output=process_media(p,tmp_path/'output',FakeDetector())
    assert output['media_type']=='image'
    assert (tmp_path/'output/annotated.jpg').is_file()
    assert len(output['frames'][0]['objects'])==1

def test_hit_testing():
    item=Detection('track:42','truck',.92,7,[20,20,110,140],[.1,.1,.55,.7],42)
    assert pick_object_at_point([item],.2,.3)['object_id']=='track:42'
    assert pick_object_at_point([item],.9,.9) is None
    assert pick_object_at_point([item],-.1,.3) is None

def test_prefers_smaller_overlap():
    large={'object_id':'truck','bbox_normalized':[.1,.1,.9,.9],'confidence':.99}
    small={'object_id':'person','bbox_normalized':[.2,.2,.3,.3],'confidence':.7}
    assert pick_object_at_point([large,small],.25,.25)['object_id']=='person'

def test_validation(tmp_path):
    with pytest.raises(FileNotFoundError):
        process_media(tmp_path/'no.mp4',tmp_path/'out',FakeDetector())
    p=tmp_path/'image.png';cv2.imwrite(str(p),np.zeros((4,4,3),np.uint8))
    with pytest.raises(ValueError,match='positive'):
        process_media(p,tmp_path/'out',FakeDetector(),sample_fps=0)
