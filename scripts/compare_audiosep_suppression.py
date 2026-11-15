#!/usr/bin/env python3
"""Audition full subtraction and conservative AudioSep post-processing locally.

Requires original.mp4, mixture.wav, and audiosep/isolated-barking.wav. Reuses the
real saved model estimate; no inference, external service, or clean source is
used for processing. Outputs a fresh folder, full-length WAVs and HQ AAC videos.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'audio-service')]
from app.media import read_wav, write_wav
from sam_audio_service.suppression import suppress_target


def mux(original, wav, destination, duration):
    subprocess.run(['ffmpeg','-v','error','-n','-i',str(original),'-i',str(wav),
                    '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k',
                    '-t',str(duration),'-movflags','+faststart',str(destination)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output-name', default='voice-preserving')
    a = p.parse_args()
    root = a.directory.resolve()
    output = root/a.output_name
    if Path(a.output_name).name != a.output_name or output.exists():
        p.error('Use a fresh simple output folder name')
    inference = json.loads((root/'audiosep/report.json').read_text())
    digest = hashlib.sha256((root/'mixture.wav').read_bytes()).hexdigest()
    if inference.get('backend') != 'audiosep' or inference.get('is_mock') is not False or inference.get('inference_verified') is not True:
        p.error('A successful real AudioSep inference report is required')
    if inference.get('input_sha256') != digest:
        p.error('Mixture differs from the input used for inference')
    mixture, rate = read_wav(root/'mixture.wav')
    target, target_rate = read_wav(root/'audiosep/isolated-barking.wav')
    if rate != 48000 or rate != target_rate or mixture.shape != target.shape:
        p.error('48 kHz mixture and target timelines/channels must match')
    source, estimate = mixture.T.astype(np.float32), target.T.astype(np.float32)
    duration = len(mixture)/rate
    output.mkdir()
    variants = [
        ('full-subtraction', 'Full subtraction', dict(strength=1., processing='raw')),
        ('voice-first', 'Voice first — gentler', dict(strength=.9, processing='source_mask')),
        ('dog-muted', 'Stronger dog cleanup', dict(strength=1., processing='source_mask')),
    ]
    report = {'model':'AudioSep, saved real estimate', 'prompt':'dog barking',
              'additional_gpu_inference':False, 'clean_references_used_for_processing':False,
              'sample_rate':rate, 'sample_count':len(mixture), 'channels':mixture.shape[1],
              'duration_seconds':duration, 'aac_bitrate_requested':'192k',
              'input_sha256':digest,
              'target_sha256':hashlib.sha256((root/'audiosep/isolated-barking.wav').read_bytes()).hexdigest(),
              'source_mask_settings':dict(mask_threshold=.85,activity_low_db=-36.,activity_high_db=-24.,n_fft=2048),
              'listening':'Candidates await human review. Prior synthetic-clip full subtraction was reported by the user to modify the voice.',
              'limitations':'Target and activity are AudioSep estimates. Actor imitations/vocalizations and other effects may be mistaken for barking. No source-specific accuracy or voice-preservation score without references.',
              'variants':[]}
    for name, label, options in variants:
        started = time.perf_counter()
        residual = source-estimate if options['processing']=='raw' else suppress_target(source, estimate, strength=options['strength'])
        removed = source-residual
        write_wav(output/f'{name}.wav', residual.T, rate)
        write_wav(output/f'{name}-removed.wav', removed.T, rate)
        mux(root/'original.mp4', output/f'{name}.wav', output/f'{name}.mp4', duration)
        probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(output/f'{name}.mp4')]))
        report['variants'].append(dict(name=name,label=label,options=options,
            processing_and_export_seconds=time.perf_counter()-started,
            reconstruction_max_abs=float(np.max(np.abs(removed+residual-source))),
            peak=float(np.max(np.abs(residual))),export_probe=probe))
    if not (root/'original-preview.mp4').exists():
        mux(root/'original.mp4', root/'mixture.wav', root/'original-preview.mp4', duration)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quiet the dog. Keep the actor.</title><style>
:root{color-scheme:dark;font:17px system-ui;background:#11191d;color:#edf5f4}body{max-width:1050px;margin:36px auto;padding:0 24px}h1{font-size:clamp(30px,5vw,52px);letter-spacing:-1.7px}p{color:#b8cccf;line-height:1.6}a{color:#9ce4cf}select{font:inherit;padding:12px;background:#243e3a;color:white;border:1px solid #5b867d;border-radius:8px;max-width:100%}video{width:100%;margin:20px 0;border-radius:12px}audio{width:100%}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.note{background:#1e2e35;padding:18px;border-radius:10px}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style>
<body><p>AudioSep · real recorded clip · text prompt: dog barking</p><h1>Quiet the dog.<br>Keep the actor.</h1>
<label>Listen to <select id="mode"><option value="../original-preview.mp4">Original recording</option><option value="full-subtraction.mp4">Full subtraction · HQ encode</option><option value="voice-first.mp4">Voice first · gentler reduction</option><option value="dog-muted.mp4" selected>Stronger dog cleanup</option></select></label>
<video id="video" controls playsinline preload="metadata" src="dog-muted.mp4"></video>
<p class="note">Compare the actor's voice, breaths and vocalizations, then the barking and other sound effects. “Voice first” retains some barking to ease processing. Stronger cleanup applies the full refined removal. Both reuse the original recording's phase and limit processing where the barking estimate is weak. Neither is guaranteed artifact-free.</p>
<p><a href="dog-muted.mp4">Stronger cleanup video</a> · <a href="voice-first.mp4">Gentler video</a> · <a href="../original-preview.mp4">Original video</a> · <a href="REPORT.md">Report</a></p>
<div class="grid"><div><h2>Removed by stronger cleanup</h2><audio controls src="dog-muted-removed.wav"></audio><p>Speech here indicates unwanted removal.</p></div><div><h2>Voice-first lossless audio</h2><audio controls src="voice-first.wav"></audio><p>Use this to check the result before AAC video encoding.</p></div></div>
<p>No additional model run for these alternatives. No isolated source references were used to alter the audio. Listening is pending; no automatic quality winner is selected.</p>
<script>const video=document.querySelector('#video');document.querySelector('#mode').onchange=e=>{const t=video.currentTime,playing=!video.paused;video.pause();video.src=e.target.value;video.addEventListener('loadedmetadata',()=>{video.currentTime=Math.min(t,video.duration);if(playing)video.play().catch(()=>{})},{once:true});video.load()};document.querySelectorAll('audio').forEach(a=>a.onplay=()=>{video.pause();document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()})});video.onplay=()=>document.querySelectorAll('audio').forEach(a=>a.pause());</script></body></html>'''
    (output/'comparison.html').write_text(page)
    (output/'REPORT.md').write_text(f'''# Quiet the dog. Keep the actor.

This comparison reuses the same real AudioSep `dog barking` estimate. No further
GPU inference or reference-assisted audio cleanup was performed.

- `full-subtraction.mp4`: previous removal method, re-encoded with HQ AAC.
- `voice-first.mp4`: original-phase ratio mask, threshold .85; target-derived
  activity transition -36 to -24 dB relative to peak; removal strength .9.
- `dog-muted.mp4`: same mask/activity, strength 1.

The ratio threshold is a heuristic, not calibrated model confidence. The activity
envelope comes from the saved target, not hand-selected or reference annotations.
The gentler candidate retains 10% amplitude in fully target-dominated regions.
When dog and actor overlap in the same frequencies, either can still be affected.
Actor vocalizations resembling barking require particular listening attention.

All outputs have {len(mixture)} samples per channel at {rate} Hz on the same
{duration:.9f}-second timeline. Video is copied from the normalized input;
new exports request AAC 192 kb/s. WAVs are lossless float32. File names ending
`-removed.wav` contain exactly what that candidate subtracted from the source.

No isolated references exist for this movie clip; no barking reduction,
speech preservation or perceptual quality score is claimed. Human listening is
pending. Check `../audiosep/report.json` for the real model execution facts,
`report.json` for export probes and `../provenance.json` for the source.
''')
    manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in output.iterdir() if f.is_file()}
    (output/'manifest-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(output/'comparison.html')


if __name__ == '__main__':
    main()
