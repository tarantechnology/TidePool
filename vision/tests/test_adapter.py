import numpy as np
from soundcanvas_cv.yolo_adapter import extract_detections
class Tensor:
    def __init__(self,x): self.data=np.array(x)
    def cpu(self): return self
    def int(self): return Tensor(self.data.astype(int))
    def tolist(self): return self.data.tolist()
class Boxes:
    xyxy=Tensor([[10,20,50,80]])
    conf=Tensor([.8741])
    cls=Tensor([7])
    id=Tensor([4])
    def __len__(self): return 1
class Result:
    boxes=Boxes();orig_shape=(100,200);names={7:'truck'}

def test_boxes_extract():
    obj=extract_detections(Result())[0]
    assert obj.label=='truck' and obj.track_id==4
    assert obj.object_id=='track:4'
    assert obj.bbox_xyxy==[10,20,50,80]
    assert obj.bbox_normalized==[.05,.2,.25,.8]
def test_empty_boxes():
    result=Result();result.boxes=None
    assert extract_detections(result)==[]
