"""CLI: python -m soundcanvas_cv --input clip.mp4"""
import argparse
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
    a = p.parse_args(argv)
    try:
        if not Path(a.input).is_file():
            raise FileNotFoundError(f'Input does not exist: {a.input}')
        detector = YOLODetector(a.model, a.conf, a.device, not a.no_track, a.tracker)
        data = process_media(a.input, a.output_dir, detector, a.sample_fps, a.max_seconds, a.video_id)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        p.exit(2, f'CV processing failed: {exc}\n')
    classes = Counter(o['label'] for f in data['frames'] for o in f['objects'])
    print(f"Exported {data['analyzed_frames']} samples / {data['processed_frames']} frames; {data['inference_frames']} inference calls")
    print('Detection observations:', dict(classes))
    print(f"Runtime: {data['runtime']['seconds']:.2f}s; device: {data['detector'].get('device')}")
    print('Preview:', Path(a.output_dir)/data['annotated_media'])
    print('Metadata:', Path(a.output_dir)/'detections.json')


if __name__ == '__main__':
    main()
