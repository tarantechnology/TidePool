#!/usr/bin/env python3
"""Render native RLE masks with source audio and sample frames for human review."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from soundcanvas_cv.masks import decode_mask
from soundcanvas_cv.pipeline import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--label', default='knife')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Choose a fresh output directory')
    data = json.loads(args.detections.read_text())
    if file_sha256(args.video) != data['source_sha256']:
        p.error('Source identity mismatch')
    records = {f['frame_index']: f for f in data['frames']}
    if set(records) != set(range(data['processed_frames'])):
        p.error('Rendering requires complete native-frame metadata')
    args.output.mkdir(parents=True)
    cap = cv2.VideoCapture(str(args.video))
    writer = cv2.VideoWriter(str(args.output/'masks-silent.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                             data['fps'], (data['width'], data['height']))
    if not cap.isOpened() or not writer.isOpened():
        raise RuntimeError('Could not open source or output codec')
    sample_indices = set(np.linspace(0, data['processed_frames']-1, 6).astype(int).tolist())
    tiles, previous, ious, areas = [], {}, [], []
    try:
        for index in range(data['processed_frames']):
            ok, frame = cap.read()
            if not ok:
                raise ValueError('Source ends before detection timeline')
            overlay = frame.copy()
            current = {}
            for o in records[index]['objects']:
                if o['label'] != args.label or o.get('mask') is None:
                    continue
                mask = decode_mask(o['mask']).astype(bool)
                if mask.shape != frame.shape[:2]:
                    raise ValueError('Native mask geometry mismatch')
                overlay[mask] = (40, 210, 250)
                current[o['object_id']] = mask
                areas.append(int(mask.sum()))
                if o['object_id'] in previous:
                    union = np.count_nonzero(mask | previous[o['object_id']])
                    if union:
                        ious.append(np.count_nonzero(mask & previous[o['object_id']])/union)
                x1, y1, _, _ = o['bbox_xyxy']
                cv2.putText(overlay, f"{o['label']} #{o['track_id']}", (x1, max(20, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (30, 30, 255), 2)
            previous = current
            image = cv2.addWeighted(frame, .55, overlay, .45, 0)
            cv2.putText(image, f'{index/data["fps"]:.2f}s', (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(image)
            if index in sample_indices:
                tiles.append(cv2.resize(image, (640, 360)))
    finally:
        cap.release()
        writer.release()
    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(args.output/'masks-silent.mp4'),
                    '-i', str(args.video), '-map', '0:v:0', '-map', '1:a:0?',
                    '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'copy',
                    '-t', str(data['processed_duration_seconds']), str(args.output/'masks-audible.mp4')], check=True)
    cv2.imwrite(str(args.output/'contact-sheet.jpg'), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
    report = dict(label=args.label, source_sha256=data['source_sha256'],
                  adjacent_unwarped_mask_iou_mean=float(np.mean(ious)) if ious else None,
                  nonempty_mask_observations=len(areas), mean_mask_area_pixels=float(np.mean(areas)) if areas else None,
                  interpretation='Unwarped IoU changes with real motion; it measures neither segmentation accuracy nor correct identity')
    (args.output/'mask-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
