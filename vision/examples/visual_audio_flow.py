"""Submit a viewer selection to running SoundCanvas services, edit and export it.

Run from the repository root; see vision/docs/ISSUE4.md. Detection and selection
documents must come from the exact uploaded bytes and the same native-FPS analysis.
The backend, not this client, chooses the audio service's media references.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import httpx


def checked(response):
    response.raise_for_status()
    return response.json()


def wait(client, url, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = checked(client.get(url))
        if job['status'] == 'failed':
            raise RuntimeError(job.get('error') or 'Processing failed')
        if job['status'] == 'completed':
            return job
        time.sleep(.25)
    raise TimeoutError(f'Job did not finish within {timeout:g}s')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--prompt', required=True, help='Sound hypothesis, e.g. person speaking')
    parser.add_argument('--volume', type=float, default=.25)
    parser.add_argument('--muted', action='store_true')
    parser.add_argument('--solo', action='store_true')
    parser.add_argument('--timeout', type=float, default=3600)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.volume <= 2:
        parser.error('--volume must be between 0 and 2')
    detections = json.loads(args.detections.read_text())
    selection = json.loads(args.selection.read_text())
    with args.video.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    if any(document.get('source_sha256') != sha for document in (detections, selection)):
        parser.error('Video, detections and selection must identify the same original bytes')
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(base_url=args.url, timeout=600) as client:
        project = checked(client.post('/api/projects'))
        pid = project['project_id']
        # Keep the project credential in memory; reports and logs contain no token.
        client.headers['Authorization'] = 'Bearer ' + project['access_token']
        with args.video.open('rb') as stream:
            uploaded = checked(client.post(f'/api/projects/{pid}/upload', files={'file': (args.video.name, stream, 'video/mp4')}))
        assert uploaded['metadata']['source_sha256'] == sha
        checked(client.put(f'/api/projects/{pid}/visual-metadata', json=detections))
        queued = checked(client.post(f'/api/projects/{pid}/process', json={'prompts': [args.prompt], 'visual_prompt': selection}))
        job = wait(client, '/api/jobs/' + queued['job_id'], args.timeout)
        inference = job.get('inference') or {}
        model = inference.get('model') or {}
        status = 'MOCK: no source isolation' if model.get('is_mock') else 'REAL MODEL: evaluate the downloaded stems' if model.get('is_mock') is False else 'Model identity unreported: separation unverified'
        print(status)
        for warning in inference.get('warnings', []):
            print('Warning:', warning)
        if len(job['stems']) != 2:
            raise RuntimeError('This example requires one target and one residual stem from the audio service')
        for index, stem in enumerate(job['stems']):
            response = client.get(stem['audio_url'])
            response.raise_for_status()
            (args.output_dir / f'stem-{index}.wav').write_bytes(response.content)
        target = job['stems'][0]
        checked(client.patch(f"/api/projects/{pid}/stems/{target['id']}", json={'volume': args.volume, 'muted': args.muted, 'solo': args.solo}))
        export = checked(client.post(f'/api/projects/{pid}/export', json={'format': 'mp4', 'include_video': True}))
        exported = wait(client, '/api/exports/' + export['export_id'], args.timeout)
        response = client.get(exported['download_url'])
        response.raise_for_status()
        (args.output_dir / 'edited.mp4').write_bytes(response.content)
        report = {'project_id': pid, 'job_id': queued['job_id'], 'source_sha256': sha,
                  'video_id': selection['video_id'], 'analysis_id': selection['analysis_id'],
                  'detection_id': selection['detection_id'], 'track_id': selection['track_id'],
                  'interval': selection['interval'], 'inference': inference,
                  'stems': job['stems'], 'edit': {'volume': args.volume, 'muted': args.muted, 'solo': args.solo},
                  'audio_association': 'unverified'}
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f'Wrote stems, edited.mp4 and report.json to {args.output_dir}')


if __name__ == '__main__':
    main()
