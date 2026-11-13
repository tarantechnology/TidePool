#!/usr/bin/env python3
"""Isolated real-model backend test using existing upload/edit/export example.

Owns two temporary localhost processes, shared experiment storage and a GPU lease.
Always stops its own services. Never modifies an existing service or logs a token.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx


def wait_ready(url, process, deadline):
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'Test service exited with status {process.returncode}; see its log')
        try:
            r = httpx.get(url, timeout=2)
            if r.status_code == 200:
                return r.json()
        except httpx.HTTPError:
            pass
        time.sleep(.5)
    raise TimeoutError('Test service did not become ready')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--detections', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prompt', default='knife chopping and scraping')
    p.add_argument('--audio-port', type=int, default=18101)
    p.add_argument('--backend-port', type=int, default=18100)
    args = p.parse_args()
    if args.output.exists() or args.audio_port == args.backend_port:
        p.error('Use a fresh output directory and distinct unused ports')
    for port in (args.audio_port, args.backend_port):
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', port))
    from soundcanvas_cv.gpu_guard import acquire_gpu
    gpu_lease = acquire_gpu()
    root = Path(__file__).resolve().parents[1]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True)
    env = dict(os.environ)
    env.update(STORAGE_ROOT=str(args.output/'storage'), AUDIO_MEDIA_ROOT=str(args.output/'storage'),
               AUDIO_BACKEND='sam', AUDIO_MODEL_ID='facebook/sam-audio-small-tv', AUDIO_DEVICE='cuda',
               AUDIO_DTYPE='float32', AUDIO_LOAD_RANKERS='false', AUDIO_LOAD_SPAN_PREDICTOR='false',
               AUDIO_PREDICT_SPANS='false', AUDIO_RERANKING_CANDIDATES='1', AUDIO_VISUAL_BATCH_SIZE='8',
               AUDIO_CHUNK_BATCH='1', AUDIO_CHUNK_SECONDS='4', AUDIO_CHUNK_OVERLAP_SECONDS='1', AUDIO_HOST='127.0.0.1',
               AUDIO_PORT=str(args.audio_port), AUDIO_LOAD_ON_STARTUP='true', INFERENCE_ADAPTER='http',
               SAM_AUDIO_SERVICE_URL=f'http://127.0.0.1:{args.audio_port}/separate',
               INFERENCE_TIMEOUT_SECONDS='1200', FFMPEG_TIMEOUT='600')
    processes, logs = [], []
    try:
        for name, command in [
            ('audio', [sys.executable, '-m', 'sam_audio_service']),
            ('backend', [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', str(args.backend_port)])]:
            log = (args.output/f'{name}.log').open('w')
            logs.append(log)
            processes.append(subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT))
        # The backend has no /health route; its existing OpenAPI route proves HTTP
        # startup without creating a project or changing the public application API.
        wait_ready(f'http://127.0.0.1:{args.backend_port}/openapi.json', processes[1], time.monotonic()+60)
        ready = wait_ready(f'http://127.0.0.1:{args.audio_port}/ready', processes[0], time.monotonic()+600)
        if ready['model'].get('is_mock') is not False or ready['model'].get('loaded') is not True:
            raise RuntimeError('GPU integration test requires a loaded, real model')
        (args.output/'ready.json').write_text(json.dumps(ready, indent=2))
        command = [sys.executable, str(root/'vision/examples/visual_audio_flow.py'),
                   '--url', f'http://127.0.0.1:{args.backend_port}', '--video', str(args.video.resolve()),
                   '--detections', str(args.detections.resolve()), '--selection', str(args.selection.resolve()),
                   '--prompt', args.prompt, '--muted', '--volume', '0', '--output-dir', str(args.output/'result')]
        subprocess.run(command, cwd=root, env=env, check=True, timeout=1200)
        report = json.loads((args.output/'result/report.json').read_text())
        if report['inference']['model'].get('is_mock') is not False:
            raise RuntimeError('Backend result did not identify genuine model inference')
        probe = subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json',
                                         str(args.output/'result/edited.mp4')])
        (args.output/'result/export-probe.json').write_bytes(probe)
        print(args.output/'result/report.json')
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        for log in logs:
            log.close()
        gpu_lease.close()


if __name__ == '__main__':
    main()
