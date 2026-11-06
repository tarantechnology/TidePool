"""CFR video/image processing and versioned frontend/backend detection metadata."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
import hashlib
import json
import math
import os
import platform
import tempfile
import time
import uuid
import cv2

IMAGES = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


@dataclass
class Detection:
    object_id: str
    label: str
    confidence: float
    class_id: int
    bbox_xyxy: list[int]
    bbox_normalized: list[float]
    track_id: int | None = None
    mask: dict | None = None


class Detector(Protocol):
    def detect(self, frame) -> list[Detection]: ...


def pick_object_at_point(detections, x: float, y: float):
    """Normalized content coordinates. Prefer smallest area then highest confidence."""
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return None
    hits = []
    for detection in detections:
        o = asdict(detection) if isinstance(detection, Detection) else detection
        x1, y1, x2, y2 = o['bbox_normalized']
        if x1 <= x <= x2 and y1 <= y <= y2:
            hits.append(((x2-x1)*(y2-y1), -o['confidence'], o))
    return min(hits, key=lambda item: (item[0], item[1]))[2] if hits else None


def draw_detections(frame, objects):
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    for o in objects:
        x1, y1, x2, y2 = o.bbox_xyxy
        color = (60, 210, 80)
        cv2.rectangle(canvas, (x1, y1), (min(w-1, x2), min(h-1, y2)), color, 2)
        label = f'{o.label} {o.confidence:.2f}' + (f' #{o.track_id}' if o.track_id is not None else '')
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
        top = max(0, y1-th-8)
        cv2.rectangle(canvas, (x1, top), (min(w-1, x1+tw+8), min(h-1, top+th+8)), color, -1)
        cv2.putText(canvas, label, (x1+3, min(h-1, top+th+3)), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _record(index, timestamp, objects, width, height):
    exported = []
    for i, obj in enumerate(objects):
        value = asdict(obj)
        x1, y1, x2, y2 = obj.bbox_xyxy
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and
                math.isfinite(obj.confidence) and 0 <= obj.confidence <= 1):
            raise ValueError('Detector produced invalid coordinates or confidence')
        value.update(detection_id=f'frame:{index}:det:{i}', class_name=obj.label)
        exported.append(value)
    return {'frame_index': index, 'timestamp_seconds': round(timestamp, 9), 'objects': exported}


def process_media(input_path: str | Path, output_dir: str | Path, detector: Detector,
                  sample_fps: float = 5., max_seconds: float = 30., video_id: str | None = None):
    """Process CFR media. Track every decoded frame; export at requested sample rate.

    Time = decoded index / nominal FPS, relative to first decoded frame, in seconds.
    Boxes use decoded, orientation-corrected pixels. Normalize VFR/rotation with FFmpeg
    before upload if video/audio PTS differ from this CFR timeline. Preview is silent.
    """
    src, dest = Path(input_path), Path(output_dir)
    if not src.is_file():
        raise FileNotFoundError(f'Input does not exist: {src}')
    if not all(math.isfinite(v) and v > 0 for v in (sample_fps, max_seconds)):
        raise ValueError('sample_fps and max_seconds must be finite and positive')
    if src.resolve() in [(dest/name).resolve() for name in ('annotated.mp4', 'annotated.jpg', 'preview.jpg', 'detections.json')]:
        raise ValueError('Input must not overwrite an output artifact')
    if video_id is not None and (not isinstance(video_id, str) or not video_id.strip()):
        raise ValueError('video_id must be a nonempty string')
    start = time.perf_counter()
    source_hash = file_sha256(src)
    if hasattr(detector, 'reset'):
        detector.reset()
    dest.mkdir(parents=True, exist_ok=True)
    # Publish artifacts only after the entire run succeeds.
    with tempfile.TemporaryDirectory(prefix='.cv-', dir=dest) as temporary:
        work = Path(temporary)
        calls = 0
        if src.suffix.lower() in IMAGES:
            frame = cv2.imread(str(src))
            if frame is None:
                raise ValueError(f'Could not read image: {src}')
            height, width = frame.shape[:2]
            objects = detector.detect(frame)
            calls = total = 1
            frames = [_record(0, 0., objects, width, height)]
            annotated_name = 'annotated.jpg'
            if not cv2.imwrite(str(work/annotated_name), draw_detections(frame, objects)):
                raise IOError('Could not write annotated image')
            media = {'media_type': 'image', 'fps': None, 'duration_seconds': None,
                     'processed_duration_seconds': None, 'sampling_fps': None}
        else:
            cap = cv2.VideoCapture(str(src))
            writer = None
            try:
                if not cap.isOpened():
                    raise ValueError(f'Could not read video: {src}')
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                if not math.isfinite(fps) or not 0 < fps <= 1000:
                    raise ValueError('Invalid video FPS')
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f'Video has no decoded frames: {src}')
                height, width = frame.shape[:2]
                annotated_name = 'annotated.mp4'
                writer = cv2.VideoWriter(str(work/annotated_name), cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError('Cannot encode MP4; install OpenCV with mp4v support')
                frames, last, total, sample_index = [], [], 0, 0
                effective_fps = min(fps, sample_fps)
                track_all = bool(getattr(detector, 'track', False))
                while ok and total/fps < max_seconds - 1e-10:
                    if frame.shape[:2] != (height, width):
                        raise ValueError('Video changed decoded dimensions')
                    sample = total/fps + 1e-10 >= sample_index/effective_fps
                    if track_all or sample:
                        last = detector.detect(frame)
                        calls += 1
                    if sample:
                        frames.append(_record(total, total/fps, last, width, height))
                        sample_index += 1
                    annotated = draw_detections(frame, last)
                    if total == 0 and not cv2.imwrite(str(work/'preview.jpg'), annotated):
                        raise IOError('Could not write preview')
                    writer.write(annotated)
                    total += 1
                    ok, frame = cap.read() if total/fps < max_seconds - 1e-10 else (False, None)
                duration = total/fps
                for i, record in enumerate(frames):
                    record['valid_until_seconds'] = frames[i+1]['timestamp_seconds'] if i+1 < len(frames) else round(duration, 9)
                media = {'media_type': 'video', 'fps': fps, 'sampling_fps': effective_fps,
                         'requested_sampling_fps': sample_fps,
                         'duration_seconds': round(cap.get(cv2.CAP_PROP_FRAME_COUNT)/fps, 9),
                         'processed_duration_seconds': round(duration, 9)}
            finally:
                cap.release()
                if writer is not None:
                    writer.release()
        result = {'schema_version': '1.0', 'analysis_id': uuid.uuid4().hex, 'video_id': video_id or f'sha256:{source_hash}',
                  'source_sha256': source_hash, 'source': src.name,
                  'width': width, 'height': height, 'frame_width': width, 'frame_height': height,
                  **media, 'time_reference': 'seconds from first decoded frame; frame_index / nominal fps (CFR)',
                  'coordinate_system': 'xyxy pixels, origin top-left, exclusive right/bottom; normalized by decoded width/height',
                  'processed_frames': total, 'analyzed_frames': len(frames), 'inference_frames': calls,
                  'annotated_media': annotated_name, 'frames': frames,
                  'detector': detector.describe() if hasattr(detector, 'describe') else {'name': type(detector).__name__},
                  'runtime': {'seconds': round(time.perf_counter()-start, 4),
                              'platform': platform.platform(), 'machine': platform.machine(),
                              'python': platform.python_version(), 'opencv': cv2.__version__}}
        (work/'detections.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        for path in work.iterdir():
            os.replace(path, dest/path.name)
    return result
