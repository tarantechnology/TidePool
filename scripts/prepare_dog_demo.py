#!/usr/bin/env python3
"""Build a disclosed 12-second controlled demo; retain clean references.

Downloaded footage contains the visible dog and two people. Authored synthetic
conversation and an independently recorded dog replace its audio; this is not a
natural synchronized recording. Media is written outside Git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.media import read_wav, write_wav

RATE = 48000
SECONDS = 12
FOOTAGE = 'https://videos.pexels.com/video-files/4265042/4265042-hd_1280_720_30fps.mp4'
BARK = 'https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-100032-A-0.wav'
LICENSE = 'https://raw.githubusercontent.com/karolpiczak/ESC-50/master/LICENSE'
LINES = [
    ('speaker_a', 'af_heart', .3, 'Did you bring the camera? We should record the demo before lunch.'),
    ('speaker_b', 'bm_george', 6., "Yes, it is ready. Let's keep talking while our dog says hello."),
]


def run(*args):
    subprocess.run(list(map(str, args)), check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--speech-dir', type=Path, required=True, help='Kokoro speaker_a.wav, speaker_b.wav and speech-provenance.json')
    args = p.parse_args()
    root = args.output.resolve()
    source, refs = root/'sources', root/'references'
    source.mkdir(parents=True, exist_ok=True)
    refs.mkdir(exist_ok=True)
    if (root/'original.mp4').exists():
        p.error('original.mp4 exists; choose a fresh output directory')
    for url, name in [(FOOTAGE, 'people-dog.mp4'), (BARK, 'barking-cc0.wav'), (LICENSE, 'ESC-50-LICENSE.txt')]:
        dest = source/name
        if not dest.exists():
            run('curl', '-fL', '--retry', 2, url, '-o', dest)
    if 'rose_bark.wav' not in (source/'ESC-50-LICENSE.txt').read_text():
        raise RuntimeError('Source attribution changed; inspect licensing before proceeding')
    signals = {}
    turns = []
    for name, voice, offset, line in LINES:
        wav = source/f'{name}.wav'
        run('ffmpeg', '-v', 'error', '-n', '-i', args.speech_dir/f'{name}.wav', '-ac', 1, '-ar', RATE, '-c:a', 'pcm_f32le', wav)
        data, rate = read_wav(wav)
        assert rate == RATE
        data = data[:, 0].astype(np.float32)
        first = round(offset*RATE)
        if first+len(data) > RATE*SECONDS:
            raise ValueError('Spoken line exceeds demo duration')
        signal = np.zeros(RATE*SECONDS, np.float32)
        signal[first:first+len(data)] = data/max(float(np.sqrt(np.mean(data**2))), 1e-8)*.1
        signals[name] = signal
        turns.append(dict(speaker=name, voice=voice, start=offset, end=(first+len(data))/RATE, text=line))
    bark48 = source/'barking-48k.wav'
    run('ffmpeg', '-v', 'error', '-n', '-i', source/'barking-cc0.wav', '-ac', 1, '-ar', RATE, '-c:a', 'pcm_f32le', bark48)
    bark, _ = read_wav(bark48)
    barking = np.tile(bark[:, 0], 3)[:RATE*SECONDS].astype(np.float32)
    speech = signals['speaker_a']+signals['speaker_b']
    barking *= float(np.sqrt(np.mean(speech**2)))/max(float(np.sqrt(np.mean(barking**2))), 1e-8)*10**(2/20)
    signals['barking'] = barking
    gain = .85/max(float(np.max(np.abs(speech+barking))), 1e-8)
    signals = {k: v*gain for k, v in signals.items()}
    signals['speech'] = signals['speaker_a']+signals['speaker_b']
    for name, data in signals.items():
        write_wav(refs/f'{name}.wav', data[:, None], RATE)
    mix = signals['speech']+signals['barking']
    write_wav(root/'mixture.wav', mix[:, None], RATE)
    # ALAC retains exact PCM timeline in an ordinary MP4 container. AAC is used for edited videos.
    run('ffmpeg', '-v', 'error', '-n', '-i', source/'people-dog.mp4', '-i', root/'mixture.wav',
        '-map', '0:v:0', '-map', '1:a:0', '-vf', 'fps=15,scale=960:540', '-t', SECONDS,
        '-c:v', 'libx264', '-crf', 20, '-pix_fmt', 'yuv420p', '-c:a', 'alac', '-movflags', '+faststart', root/'original.mp4')
    provenance = {
        'constructed_mixture': True, 'natural_recording': False, 'duration_seconds': SECONDS,
        'sample_rate': RATE, 'samples': len(mix), 'interval': [0, SECONDS],
        'footage': {'author':'Gustavo Fring', 'page':'https://www.pexels.com/video/people-playing-with-the-dog-4265042/',
                    'download':FOOTAGE, 'license':'Pexels License', 'license_url':'https://www.pexels.com/license/'},
        'barking': {'author':'nfrae', 'title':'rose_bark.wav', 'original':'https://freesound.org/people/nfrae/sounds/100032/',
                    'download':BARK, 'license':'CC0', 'license_evidence':'sources/ESC-50-LICENSE.txt, 1-100032-A.ogg'},
        'speech': {'type':'authored dialogue synthesized locally with Kokoro on CPU', 'turns':turns,
                   **json.loads((args.speech_dir/'speech-provenance.json').read_text())},
        'bark_to_speech_rms_db': 2, 'common_peak_gain':gain,
        'edits':'Footage trimmed/resized to 12 s, 15 FPS, 960x540. Original audio replaced by controlled mixture.',
        'caveat':'The dog is visible, but added barking is not synchronized to its mouth. Voices are synthetic, not the depicted people. AudioSep receives audio and text only.',
        'sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
                  for f in [root/'original.mp4', root/'mixture.wav', *refs.glob('*.wav')]}}
    (root/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(root)


if __name__ == '__main__':
    main()
