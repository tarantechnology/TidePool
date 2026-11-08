"""Run the real OpenCV pipeline with simulated box detections."""
from pathlib import Path
from soundcanvas_cv.pipeline import Detection, process_media

class FakeDetector:
    def detect(self, frame):
        h,w=frame.shape[:2]
        return [Detection('track:7','demo-object',.99,0,[30,30,115,135],
                          [30/w,30/h,115/w,135/h],7)]

root=Path(__file__).resolve().parents[1]
data=process_media(root/'examples/synthetic_input.avi',
    root/'outputs/offline_smoke_test',FakeDetector(),sample_fps=5,max_seconds=2)
print(f"OK: {data['processed_frames']} frames, {data['analyzed_frames']} samples")
print('Output: outputs/offline_smoke_test/annotated.mp4')
