#!/usr/bin/env python3
"""Download one CC BY-SA cooking-related clip and normalize 12 seconds for testing.

Obihoja, Cutting Vegitable In A Market (2025), CC BY-SA 4.0.
https://commons.wikimedia.org/wiki/File:Cutting_Vegitable_In_A_Market.webm
Derivative media must retain attribution and CC BY-SA 4.0 terms.
No isolated reference stems or guaranteed music/speech mixture are provided.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

URL = 'https://upload.wikimedia.org/wikipedia/commons/7/79/Cutting_Vegitable_In_A_Market.webm'
SHA1 = 'fff39ddc436b0517615634087c0a2f13a8f9df52'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Choose a fresh output directory; existing footage is never overwritten')
    a.output.mkdir(parents=True)
    source = a.output/'source.webm'
    urllib.request.urlretrieve(URL, source)
    if hashlib.sha1(source.read_bytes()).hexdigest() != SHA1:
        raise ValueError('Downloaded source checksum differs from Wikimedia original')
    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(source), '-t', '12',
                    '-vf', 'fps=30,scale=1280:720,setsar=1,setpts=PTS-STARTPTS',
                    '-af', 'aresample=48000,asetpts=PTS-STARTPTS', '-c:v', 'libx264',
                    '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', str(a.output/'cooking.mp4')], check=True)
    (a.output/'source.json').write_text(json.dumps(dict(
        url=URL, source_sha1=SHA1, author='Obihoja', license='CC BY-SA 4.0',
        license_url='https://creativecommons.org/licenses/by-sa/4.0/',
        changes='First 12 seconds; normalized CFR 30 FPS, 1280x720, 48kHz audio',
        reference_stems_available=False), indent=2))
    print(a.output/'cooking.mp4')


if __name__ == '__main__':
    main()
