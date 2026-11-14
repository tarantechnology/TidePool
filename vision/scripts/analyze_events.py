#!/usr/bin/env python3
"""Generate an evidence sidecar without modifying detections or asserting causality."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np

from soundcanvas_cv.events import associate, audio_events
from soundcanvas_cv.pipeline import file_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--hypotheses', type=Path, help='JSON label -> candidate sound labels; never treated as observations')
    p.add_argument('--audio-events', type=Path, help='Optional independent annotations: {source_sha256, events:[{id,start,end,timestamp,label,provenance}]}')
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists; choose a fresh sidecar path')
    data = json.loads(args.detections.read_text())
    if file_sha256(args.video) != data['source_sha256']:
        p.error('Video bytes differ from analyzed source')
    if args.audio_events:
        external = json.loads(args.audio_events.read_text())
        if external['source_sha256'] != data['source_sha256']:
            p.error('Audio annotations belong to a different source')
        acoustic = external['events']
    else:
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(args.video),
                                       '-t', str(data['processed_duration_seconds']), '-vn',
                                       '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'])
        acoustic = audio_events(np.frombuffer(raw, dtype='<f4'), 16000)
    hypotheses = json.loads(args.hypotheses.read_text()) if args.hypotheses else {}
    graph = associate(data, acoustic, hypotheses)
    graph['object_observations']['reference'] = str(args.detections.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=2, allow_nan=False))
    print(json.dumps({k: len(graph[k]) for k in ['visual_events', 'audio_events', 'associations']}))


if __name__ == '__main__':
    main()
