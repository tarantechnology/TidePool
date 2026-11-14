"""CLI: python -m soundcanvas_cv --input clip.mp4"""
import argparse
import json
from collections import Counter
from pathlib import Path
from .pipeline import process_media
from .yolo_adapter import YOLODetector


def main(argv=None):
    p = argparse.ArgumentParser(description='SoundCanvas CV detection/tracking')
    p.add_argument('--input', required=True)
    p.add_argument('--output-dir', default='outputs/first_run')
    p.add_argument('--model', default='yolo26n.pt')
    p.add_argument('--conf', type=float, default=.30)
    p.add_argument('--sample-fps', type=float, default=5.)
    p.add_argument('--max-seconds', type=float, default=30.)
    p.add_argument('--device', default='auto')
    p.add_argument('--video-id', default=None, help='Opaque backend media ID; defaults to SHA256 identity')
    p.add_argument('--tracker', default='bytetrack.yaml')
    p.add_argument('--no-track', action='store_true')
    p.add_argument('--image-size', type=int, default=None, help='YOLO inference size (e.g. 1280 for small objects)')
    p.add_argument('--adapter', choices=['yolo', 'sam3', 'sam3.1', 'sam2.1'], default='yolo')
    p.add_argument('--concept', default='knife', help='SAM concept; all matching instances are tracked')
    p.add_argument('--point-prompt', type=Path, help='JSON with frame_index, normalized points, point_labels, obj_id')
    p.add_argument('--seed-objects', type=Path, help='SAM 2 JSON list of {object_id,label,frame_index,points,point_labels}')
    a = p.parse_args(argv)
    detector = None
    try:
        if not Path(a.input).is_file():
            raise FileNotFoundError(f'Input does not exist: {a.input}')
        if a.adapter == 'yolo':
            detector = YOLODetector(a.model, a.conf, a.device, not a.no_track, a.tracker, a.image_size)
        elif a.adapter == 'sam2.1':
            from .sam2_adapter import SAM2VideoDetector
            if not a.seed_objects:
                raise ValueError('SAM 2 requires --seed-objects; it has no text detector')
            detector = SAM2VideoDetector(a.input, a.model, json.loads(a.seed_objects.read_text()), a.max_seconds)
        else:
            from .sam3_adapter import SAM3VideoDetector
            detector = SAM3VideoDetector(a.input, a.model, a.concept, a.adapter, a.conf,
                                         a.max_seconds, json.loads(a.point_prompt.read_text()) if a.point_prompt else None)
        data = process_media(a.input, a.output_dir, detector, a.sample_fps, a.max_seconds, a.video_id)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        p.exit(2, f'CV processing failed: {exc}\n')
    finally:
        if detector is not None and hasattr(detector, 'close'):
            detector.close()
    classes = Counter(o['label'] for f in data['frames'] for o in f['objects'])
    print(f"Exported {data['analyzed_frames']} samples / {data['processed_frames']} frames; {data['inference_frames']} inference calls")
    print('Detection observations:', dict(classes))
    print(f"Runtime: {data['runtime']['seconds']:.2f}s; device: {data['detector'].get('device')}")
    print('Preview:', Path(a.output_dir)/data['annotated_media'])
    print('Metadata:', Path(a.output_dir)/'detections.json')


if __name__ == '__main__':
    main()
