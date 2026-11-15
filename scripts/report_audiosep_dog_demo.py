#!/usr/bin/env python3
"""Create a portable before/after listening report from real experiment results."""
import argparse
import html
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    a = p.parse_args()
    root = a.directory
    facts = json.loads((root/'audiosep/report.json').read_text())
    provenance = json.loads((root/'provenance.json').read_text())
    if facts.get('inference_verified') is not True or facts.get('is_mock') is not False:
        p.error('A successful real-inference report is required')
    # Chromium does not decode the lossless ALAC soundtrack in original.mp4.
    # Use the same AAC encoder as the editor for a portable playback baseline.
    if not (root/'original-preview.mp4').exists():
        subprocess.run(['ffmpeg','-v','error','-n','-i',str(root/'original.mp4'),
                        '-i',str(root/'mixture.wav'),'-map','0:v:0','-map','1:a:0',
                        '-c:v','copy','-c:a','aac','-t','12','-movflags','+faststart',
                        str(root/'original-preview.mp4')], check=True)
    m = facts['metrics']
    gib = facts['cuda_peak_allocated_bytes']/2**30
    rows = [
        ('Barking reduction, waveform projection', '0 dB', f"{m['barking_reduction_db_projection']:.2f} dB"),
        ('Voice A retained gain', '0 dB', f"{m['speaker_a_gain_db']:+.2f} dB"),
        ('Voice B retained gain', '0 dB', f"{m['speaker_b_gain_db']:+.2f} dB"),
        ('Combined speech SDR', f"{m['baseline_speech_sdr_db']:.2f} dB", f"{m['speech_sdr_db']:.2f} dB"),
        ('Speech SI-SDR improvement', '0 dB', f"{m['speech_si_sdri_db']:+.2f} dB"),
        ('Inference time, warm model', '—', f"{facts['inference_seconds']:.3f} s"),
        ('Model load time', '—', f"{facts['load_seconds']:.2f} s"),
        ('Peak allocated VRAM during inference', '—', f'{gib:.3f} GiB'),
    ]
    turns = '\n'.join(f"- {t['speaker']} ({t['start']:.1f}–{t['end']:.2f} s): {t['text']}"
                      for t in provenance['speech']['turns'])
    table = '\n'.join('| '+' | '.join(row)+' |' for row in rows)
    md = f'''# Keep the conversation. Quiet the dog.

Real AudioSep CUDA inference on NVIDIA L4. Text prompt: **dog barking**.
SAM experiments were cancelled at the user's request. No SAM inference was run.
AudioSep uses audio and text only; there is no visual guidance.

| Measurement | Original | AudioSep residual |
|---|---:|---:|
{table}

Open `comparison.html` for synchronized before/after playback. The edited video is
`audiosep/edited.mp4`; lossless stems are `audiosep/isolated-barking.wav` and
`audiosep/conversation-residual.wav`. `original.mp4` is the controlled input video;
`original-preview.mp4` is its AAC browser playback copy. `mixture.wav` remains the
authoritative lossless input for all measurements.

## What this establishes

The official AudioSep base 4M checkpoint and CLAP encoder ran on CUDA through the
SoundCanvas adapter and separation engine. The existing editor export code muted
the isolated target and retained the mixture-derived residual. Same full interval
[0,12), 576,000 samples at 48 kHz, one pass, float32, seed 42, no refinement,
no gain tuning and no reference-assisted edits. Checkpoints and source revisions
are recorded in the runtime evidence alongside `audiosep/report.json`.

Target + residual reconstruct the source with maximum absolute error
{facts['reconstruction_max_abs']:.3g}. Export timing is recorded in report.json.
Runtime is one measured clip, including first-call GPU/kernel work, not a latency benchmark.

## Listen before deciding

Listening and perceptual sound-quality evaluation are pending human review.
No best-model claim is made. Compare at the same volume: confirm both voices stay
intelligible, listen for remaining barking, and check for warbling or missing syllables.
Also audition the isolated barking for stolen speech. Waveform projection estimates
retained source gains; distorted/delayed remnants can evade it. Aggregate speech SDR
includes both barking and processing distortion. These metrics cannot replace listening.

## Controlled scene and provenance

This is a constructed demo, not a natural conversation recording. Footage shows
two people with a visible dog; the original audio was replaced by authored synthetic
dialogue and a separate real barking recording. Dog-mouth synchronization is not
claimed. Voices are synthetic, not the depicted people. Barking RMS is 2 dB above
combined speech RMS; one common gain keeps the mixture peak at 0.85.

{turns}

- Footage: [Gustavo Fring, Pexels 4265042]({provenance['footage']['page']}),
  [Pexels License]({provenance['footage']['license_url']}); trimmed/resized to 12 s, 960×540, 15 FPS.
- Barking: [nfrae, rose_bark.wav]({provenance['barking']['original']}), CC0;
  ESC-50 clip 1-100032-A-0.wav. Original attribution ledger is included in sources/.
- Speech: authored lines rendered locally on CPU by [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), Apache-2.0 weights, voices af_heart and bm_george; kokoro-onnx runtime (MIT).
- Clean source references, source hashes and exact transformations: `references/` and `provenance.json`.

Only shareable media/reports are bundled. Credentials, checkpoints, virtual environments,
and application databases are excluded. Synthetic voices and one barking source limit
generalization to real recordings. Use a natural recorded conversation as the next validation.
'''
    (root/'REPORT.md').write_text(md)
    trs = ''.join('<tr>'+''.join(f'<td>{html.escape(c)}</td>' for c in row)+'</tr>' for row in rows)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Keep the conversation. Quiet the dog.</title>
<style>
:root{color-scheme:dark;font:16px system-ui;background:#11171b;color:#eef2f3}
body{max-width:1150px;margin:40px auto;padding:0 24px}h1{font-size:clamp(32px,5vw,56px);letter-spacing:-2px;margin-bottom:10px}
p{line-height:1.6;color:#b8c8ce}a{color:#9fe7d7}button,select{font:inherit;border:1px solid #42605d;background:#1f3833;color:white;border-radius:8px;padding:10px 16px;cursor:pointer}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}video{width:100%;border-radius:12px;background:black}
.bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:20px 0}.badge{color:#91dcc8;font-size:13px;letter-spacing:1px;text-transform:uppercase}
table{width:100%;border-collapse:collapse;margin:28px 0}th,td{text-align:left;padding:12px;border-bottom:1px solid #2b3b40}th{color:#9fe7d7}audio{width:100%}
.note{padding:18px 22px;background:#1a262b;border-radius:12px}.small{font-size:14px}input{flex:1;min-width:130px}
@media(max-width:700px){.grid{grid-template-columns:1fr}body{margin:25px auto}td,th{padding:9px;font-size:13px}}
</style><body><span class="badge">AudioSep · Real L4 GPU inference · Text only</span>
<h1>Keep the conversation.<br>Quiet the dog.</h1>
<p>Same 12 seconds. Two synthetic voices and real barking. Prompt: <strong>dog barking</strong>.</p>
<div class="bar"><button id="play">Play both</button><button id="reset">Restart</button>
<label>Listen to <select id="which"><option value="original">Original</option><option value="edited">AudioSep edited</option></select></label>
<input aria-label="Shared timeline" id="seek" type="range" min="0" max="12" step="0.01" value="0"><span id="time">0.0 / 12 s</span></div>
<div class="grid"><div><h2>Original mixture</h2><video id="original" src="original-preview.mp4" playsinline preload="metadata"></video><p class="small"><a href="original-preview.mp4">Open original video (AAC)</a></p></div>
<div><h2>AudioSep edited</h2><video id="edited" src="audiosep/edited.mp4" playsinline preload="metadata" muted></video><p class="small"><a href="audiosep/edited.mp4">Open edited video</a></p></div></div>
<p class="note">Constructed demo: stock footage with replaced audio. Dialogue is synthetic and barking is not mouth-synchronized. AudioSep receives no frames or masks. Listen to both turns, remaining barking, and any lost syllables before judging quality.</p>
<table><thead><tr><th>Measurement</th><th>Original</th><th>AudioSep residual</th></tr></thead><tbody>ROWS</tbody></table>
<p class="small">Projection metrics estimate source gains and can miss distorted remnants. Listening is pending; no perceptual quality or best-model claim is made.</p>
<div class="grid"><div><h3>Isolated barking</h3><audio controls src="audiosep/isolated-barking.wav"></audio><p class="small">Check for speech leaking into the removed target.</p></div>
<div><h3>Conversation residual</h3><audio controls src="audiosep/conversation-residual.wav"></audio><p class="small">Lossless edited audio, before MP4 encoding.</p></div></div>
<p><a href="REPORT.md">Full comparison report</a> · <a href="audiosep/report.json">Raw inference results</a> · <a href="provenance.json">Sources and licenses</a> · <a href="references/speech.wav">Clean conversation reference</a></p>
<script>
const original=document.querySelector('#original'),edited=document.querySelector('#edited'),play=document.querySelector('#play'),seek=document.querySelector('#seek'),which=document.querySelector('#which');
function choose(){original.muted=which.value!=='original';edited.muted=which.value!=='edited'}
function pause(){original.pause();edited.pause();play.textContent='Play both'}
which.onchange=choose;choose();
play.onclick=async()=>{if(!original.paused){pause();return}if(original.currentTime>=11.95)original.currentTime=0;edited.currentTime=original.currentTime;try{await Promise.all([original.play(),edited.play()]);play.textContent='Pause'}catch(e){pause();alert('Open the individual video links to play in your media player.')}};
document.querySelector('#reset').onclick=()=>{pause();original.currentTime=edited.currentTime=0;seek.value=0};
seek.oninput=()=>{original.currentTime=edited.currentTime=+seek.value};
original.ontimeupdate=()=>{seek.value=original.currentTime;document.querySelector('#time').textContent=original.currentTime.toFixed(1)+' / 12 s';if(!original.paused&&Math.abs(original.currentTime-edited.currentTime)>.15)edited.currentTime=original.currentTime};original.onended=pause;
document.querySelectorAll('audio').forEach(a=>a.onplay=()=>{pause();document.querySelectorAll('audio').forEach(b=>{if(a!==b)b.pause()})});
</script></body></html>'''.replace('ROWS', trs)
    (root/'comparison.html').write_text(page)
    print(root/'comparison.html')


if __name__ == '__main__':
    main()
