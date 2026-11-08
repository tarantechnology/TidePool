"""Ultralytics adapter. Each process_media call resets per-video tracker state."""
import math
from pathlib import Path
from .pipeline import Detection, file_sha256
from .masks import encode_mask


class YOLODetector:
    def __init__(self, model_path='yolo26n.pt', confidence=.30, device='auto', track=True,
                 tracker='bytetrack.yaml'):
        if not math.isfinite(confidence) or not 0 < confidence <= 1:
            raise ValueError('confidence must be in (0,1]')
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.model_path = str(model_path)
        self.confidence = confidence
        self.device = device
        self.track = track
        self.tracker = tracker

    def reset(self):
        for tracker in getattr(self.model.predictor, 'trackers', []):
            tracker.reset()

    def describe(self):
        import ultralytics
        return {"name": "ultralytics", "version": ultralytics.__version__,
                "model": Path(self.model_path).name,
                "weights_sha256": file_sha256(self.model_path) if Path(self.model_path).is_file() else None, "task": self.model.task,
                "device": str(getattr(self.model.predictor, 'device', self.device)),
                "confidence_threshold": self.confidence,
                "tracking": self.track, "tracker": self.tracker if self.track else None,
                "tracking_cadence": "every decoded frame" if self.track else None}

    def detect(self, frame):
        kwargs = {'conf': self.confidence, 'verbose': False}
        if self.device not in (None, 'auto'):
            kwargs['device'] = self.device
        if self.model.task == 'segment':
            kwargs['retina_masks'] = True  # native decoded size, avoids letterbox distortion
        if self.track:
            result = self.model.track(frame, persist=True, tracker=self.tracker, **kwargs)[0]
        else:
            result = self.model.predict(frame, **kwargs)[0]
        return extract_detections(result)


def extract_detections(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    h, w = result.orig_shape
    coords = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    classes = boxes.cls.cpu().tolist()
    tids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None]*len(coords)
    masks = getattr(result, 'masks', None)
    mask_data = masks.data.cpu().numpy() if masks is not None else None
    objects = []
    for index, (xyxy, conf, cls, tid) in enumerate(zip(coords, confs, classes, tids)):
        if not all(math.isfinite(v) for v in [*xyxy, conf, cls]) or not 0 <= conf <= 1:
            continue
        x1, y1, x2, y2 = xyxy
        x1, x2 = max(0., min(float(w), x1)), max(0., min(float(w), x2))
        y1, y2 = max(0., min(float(h), y1)), max(0., min(float(h), y2))
        pixels = [round(x1), round(y1), round(x2), round(y2)]
        if pixels[0] >= pixels[2] or pixels[1] >= pixels[3]:
            continue
        mask = None
        if mask_data is not None:
            raw = mask_data[index]
            if raw.shape != (h, w):
                raise ValueError('Segmentation requires retina_masks=True at original resolution')
            mask = encode_mask(raw > .5)
        objects.append(Detection(
            object_id=f'track:{tid}' if tid is not None else f'detection:{index}',
            label=result.names[int(cls)], confidence=round(float(conf), 4), class_id=int(cls),
            bbox_xyxy=pixels,
            bbox_normalized=[round(pixels[0]/w, 6), round(pixels[1]/h, 6),
                             round(pixels[2]/w, 6), round(pixels[3]/h, 6)],
            track_id=int(tid) if tid is not None else None, mask=mask))
    return objects
