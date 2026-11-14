#!/usr/bin/env python3
"""One model per process; native masks, measured runtime/VRAM, unlabelled proxies.

Run YOLO nano, YOLO X and SAM in separate invocations. This script never downloads
SAM weights and refuses to start if another GPU compute process is present.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from soundcanvas_cv.pipeline import process_media
from soundcanvas_cv.masks import decode_mask
from soundcanvas_cv.selection import select_at_point
from soundcanvas_cv.gpu_guard import acquire_gpu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--adapter', choices=['yolo', 'sam3', 'sam3.1', 'sam2.1'], default='yolo')
    p.add_argument('--seed-objects', type=Path)
    p.add_argument('--concept', default='knife')
    p.add_argument('--image-size', type=int, default=640)
    p.add_argument('--seconds', type=float, default=12.)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output directory already exists; choose a fresh run directory')
    gpu_lease = acquire_gpu()
    if not torch.cuda.is_available():
        p.error('CUDA is required for measured GPU experiments')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    detector = None
    try:
        if args.adapter == 'yolo':
            from soundcanvas_cv.yolo_adapter import YOLODetector
            detector = YOLODetector(args.model, .25, 'cuda:0', image_size=args.image_size)
        elif args.adapter == 'sam2.1':
            from soundcanvas_cv.sam2_adapter import SAM2VideoDetector
            if not args.seed_objects:
                p.error('SAM 2 requires --seed-objects')
            detector = SAM2VideoDetector(args.input, args.model, json.loads(args.seed_objects.read_text()), args.seconds)
        else:
            from soundcanvas_cv.sam3_adapter import SAM3VideoDetector
            detector = SAM3VideoDetector(args.input, args.model, args.concept, args.adapter,
                                         max_seconds=args.seconds)
        data = process_media(args.input, args.output, detector, sample_fps=1000, max_seconds=args.seconds)
        torch.cuda.synchronize()
        stats = dict(elapsed_seconds=time.perf_counter()-started,
                     cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                     gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        failure = dict(error_type=type(exc).__name__, elapsed_seconds=time.perf_counter()-started,
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                       inference_verified=False)
        (args.output/'failure.json').write_text(json.dumps(failure, indent=2))
        raise
    finally:
        if detector is not None and hasattr(detector, 'close'):
            detector.close()
    counts = Counter(o['label'] for f in data['frames'] for o in f['objects'])
    tracks = defaultdict(list)
    for frame in data['frames']:
        for o in frame['objects']:
            if o['label'] == args.concept:
                tracks[o['object_id']].append((frame, o))
    stats.update(observations=dict(counts), target_tracks={k: len(v) for k, v in tracks.items()},
                 target_coverage=sum(any(o['label'] == args.concept for o in f['objects']) for f in data['frames'])/len(data['frames']),
                 quality_metrics='Coverage and track lengths are proxies; no ground-truth accuracy or ID-switch claim')
    if tracks:
        oid, entries = max(tracks.items(), key=lambda item: len(item[1]))
        frame, obj = entries[0]
        mask = decode_mask(obj['mask'])
        yy, xx = np.nonzero(mask)
        # Select an actual foreground pixel for the exported picker-compatible selection.
        point = len(xx)//2
        selection = select_at_point(data, frame['timestamp_seconds'], (int(xx[point])+.5)/data['width'], (int(yy[point])+.5)/data['height'])
        # Overlapping bounding boxes can select another object: require the intended ID.
        if selection and selection['object_id'] == oid:
            selection['interval'] = {'start_seconds': 0., 'end_seconds': data['processed_duration_seconds']}
            (args.output/'selection.json').write_text(json.dumps(selection, indent=2))
        (args.output/'frame-masks.json').write_text(json.dumps({str(f['frame_index']): o['mask'] for f, o in entries}))
        cv2.imwrite(str(args.output/'mask.png'), mask.astype(np.uint8)*255)
        stats['selected_object_id'] = oid
    (args.output/'gpu-report.json').write_text(json.dumps(stats, indent=2, allow_nan=False))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
