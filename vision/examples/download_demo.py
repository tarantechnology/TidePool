"""Download pinned real pedestrian footage and produce a 12s browser-playable MP4.

The added AAC track is SILENCE for backend upload compatibility, not original sound.
Requires FFmpeg on PATH. Does not commit/download weights or media to git.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import requests

REVISION = '5728f796f6860246134066b4338c31c612adde77'
URL = f'https://raw.githubusercontent.com/opencv/opencv/{REVISION}/samples/data/vtest.avi'
SHA256 = '45cddc9490be69345cbdab64ca583be65987e864ca408038e648db99e10516cf'
ROOT = Path(__file__).resolve().parent/'media'


def main():
    if not shutil.which('ffmpeg'):
        raise SystemExit('Install FFmpeg first (macOS: brew install ffmpeg; Ubuntu: sudo apt-get install ffmpeg)')
    ROOT.mkdir(exist_ok=True)
    source = ROOT/'vtest.avi'
    if not source.exists():
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        source.write_bytes(response.content)
    if hashlib.sha256(source.read_bytes()).hexdigest() != SHA256:
        raise ValueError('Source checksum mismatch; remove examples/media/vtest.avi and retry')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-f', 'lavfi', '-i',
                    'anullsrc=r=48000:cl=stereo', '-t', '12', '-map', '0:v:0', '-map', '1:a:0',
                    '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-r', '10', '-c:a', 'aac',
                    '-movflags', '+faststart', str(ROOT/'pedestrians.mp4')], check=True)
    (ROOT/'source.json').write_text(json.dumps({'url': URL, 'revision': REVISION, 'sha256': SHA256,
        'output': 'pedestrians.mp4', 'seconds': 12, 'fps': 10,
        'audio': 'added silence; original vtest.avi has no audio',
        'provenance': 'OpenCV sample vtest.avi (pedestrian surveillance footage); no new media license asserted'}, indent=2))
    print(ROOT/'pedestrians.mp4')


if __name__ == '__main__':
    main()
