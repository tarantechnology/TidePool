"""Local word-timed transcription and reversible profanity-beep previews."""
import importlib.util
import importlib.machinery
import sys
import subprocess
import json
import math
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path

from .candidates import revisions, safe_path
from .core import probe, project_dir, run, uid
from .media import export
from .visual import file_sha256

# Deliberately exact English words, not substring matching. This is not exhaustive.
PROFANITY = frozenset('fuck fucking fucked fucker fuckers shit shitty bullshit bitch bitches asshole assholes motherfucker motherfuckers cunt cunts bastard bastards'.split())
MAX_DURATION = 600
MAX_WORDS = 200


def available():
    model = os.getenv('SOUNDCANVAS_TRANSCRIPTION_MODEL', '')
    extra = [p for p in os.getenv('SOUNDCANVAS_TRANSCRIPTION_PYTHONPATH', '').split(os.pathsep) if p]
    executable = os.getenv('SOUNDCANVAS_TRANSCRIPTION_PYTHON', '')
    found = (Path(executable).is_file() and os.access(executable, os.X_OK)) if executable else (importlib.util.find_spec('faster_whisper') or importlib.machinery.PathFinder.find_spec('faster_whisper', extra))
    return bool(model and Path(model).is_absolute() and Path(model).is_dir() and found)


@lru_cache(maxsize=1)
def model_for(path, device, compute):
    for extra in os.getenv('SOUNDCANVAS_TRANSCRIPTION_PYTHONPATH', '').split(os.pathsep):
        if extra and extra not in sys.path: sys.path.append(extra)
    from faster_whisper import WhisperModel
    return WhisperModel(path, device=device, compute_type=compute, local_files_only=True)


def transcribe(path, duration):
    if not available():
        raise ValueError('Local speech transcription is not configured. Install faster-whisper and configure a local model.')
    executable = os.getenv('SOUNDCANVAS_TRANSCRIPTION_PYTHON', '')
    model_path = os.environ['SOUNDCANVAS_TRANSCRIPTION_MODEL']
    device = os.getenv('SOUNDCANVAS_TRANSCRIPTION_DEVICE', 'cpu')
    compute = os.getenv('SOUNDCANVAS_TRANSCRIPTION_COMPUTE', 'int8')
    if executable:
        try:
            result = subprocess.run([executable, str(Path(__file__).with_name('transcription_runner.py'))],
                input=json.dumps({'path':str(path),'model':model_path,'device':device,'compute':compute}),
                text=True, capture_output=True, timeout=900, check=True)
            data=json.loads(result.stdout); raw=data['words']; language=data.get('language')
        except (OSError, subprocess.SubprocessError, ValueError, KeyError) as exc:
            raise ValueError('Local transcription failed. Check the configured transcription runtime and model.') from exc
    else:
        model = model_for(model_path, device, compute)
        segments, info = model.transcribe(str(path), word_timestamps=True, vad_filter=True, beam_size=5, condition_on_previous_text=False)
        raw = [{'word':str(w.word).strip(),'start':w.start,'end':w.end} for segment in segments for w in (segment.words or [])]
        language = getattr(info, 'language', None)
    words = []
    for word in raw:
        try: a, b = float(word['start']), float(word['end'])
        except (KeyError, ValueError, TypeError): continue
        if math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= duration + .05:
            words.append({'word': str(word.get('word','')).strip(), 'start': a, 'end': min(b, duration)})
    return words, language



def matches(words):
    return [w for w in words if re.sub(r"[^a-z]", '', w['word'].lower()) in PROFANITY]


def windows(words, duration):
    intervals = []
    for w in sorted(words, key=lambda x: x['start']):
        a, b = max(0., w['start'] - .08), min(duration, w['end'] + .08)
        if b <= a: continue
        if intervals and a <= intervals[-1][1]: intervals[-1][1] = max(b, intervals[-1][1])
        else: intervals.append([a, b])
    return intervals


def build(project, payload, candidate_id):
    pid = project['id']; snapshot = payload['_candidate']; source = safe_path(pid, snapshot['source']); rows = snapshot['rows']
    if file_sha256(source) != snapshot['source_sha256']: raise ValueError('Source changed; generate another preview.')
    for row in rows:
        if file_sha256(safe_path(pid, row['path'])) != snapshot['stem_hashes'][row['id']]: raise ValueError('A layer changed; generate another preview.')
    duration = probe(source)['duration']
    if duration > MAX_DURATION: raise ValueError('Transcription is limited to 10 minutes per preview.')
    solo = any(r['solo'] for r in rows)
    active = [r for r in rows if not r['muted'] and r['volume'] > 0 and (not solo or r['solo'])]
    voices = [r for r in active if r.get('role') == 'target' and re.search(r'\b(voice|vocals|vocal|speech|speaking|dialogue|dialog|narration|speaker)\b', ' '.join(str(r.get(k) or '') for k in ('name', 'label', 'prompt')), re.I)]
    selected = voices if len(voices) == 1 else active
    if rows and not selected: raise ValueError('There are no audible layers to transcribe.')
    directory = project_dir(pid) / 'stems'; directory.mkdir(parents=True, exist_ok=True)
    original = directory / (uid()+'.wav'); input_audio = directory / (uid()+'.wav'); mix = directory / (uid()+'.wav'); tone = directory / (uid()+'.wav')
    try:
        if rows:
            export(rows, source, original, 'wav', False)
            export(selected, source, input_audio, 'wav', False)
        else:
            shutil.copyfile(source, original); shutil.copyfile(source, input_audio)
        words, language = transcribe(input_audio, duration)
        flagged = matches(words)
        if not flagged: raise ValueError('No listed English profanity was found. Transcription may miss words; listen and review manually.')
        if len(flagged) > MAX_WORDS: raise ValueError('Too many matches for one preview; use a shorter clip.')
        intervals = windows(flagged, duration)
        proposed = [dict(r) for r in rows]
        if not proposed:
            proposed = [dict(id=uid(), project_id=pid, name='Original audio', path=str(source), duration=duration, start_offset=0, volume=1, muted=0, solo=0, automation='[]', mute_ranges='[]', revision=0)]
            selected = proposed
        import json
        selected_ids = {r['id'] for r in selected}
        for row in proposed:
            if row['id'] in selected_ids:
                existing = json.loads(row.get('mute_ranges') or '[]')
                row['mute_ranges'] = json.dumps(existing + intervals)
        # A short fade on the generated tone avoids abrupt tone edges; the mute
        # interval is padded so the original word does not leak through the fade.
        terms = [f'if(between(t,{a:.6f},{b:.6f}),min(1,min((t-{a:.6f})/0.01,({b:.6f}-t)/0.01)),0)' for a,b in intervals]
        envelope = 'min(1,' + '+'.join(terms) + ')'
        expression = f'0.18*sin(2*PI*1000*t)*({envelope})'
        run('ffmpeg','-v','error','-y','-f','lavfi','-i',f"aevalsrc='{expression}|{expression}':s=48000",'-t',str(duration),'-c:a','pcm_f32le',str(tone))
        proposed.append(dict(id=uid(), project_id=pid, name='Censor beeps', label='Censor beeps', role='effect', path=str(tone), duration=duration, start_offset=0, volume=1, muted=0, solo=int(solo), automation='[]', mute_ranges='[]', revision=0))
        export(proposed, source, mix, 'wav', False)
        warning = 'Speech was transcribed locally. Word timings and profanity detection can miss or mishear words; review the preview before applying.'
        scope_warning = 'Other layers may still contain the same speech.' if len(voices)==1 else 'All audible layers are muted during each beep, so background sounds and other speakers may also be affected.'
        return {'target':{'path':str(tone)},'mix':{'path':str(mix)},'original':{'path':str(original)},'snapshot':snapshot,'proposed':proposed,
                'inference':{'model':{'backend':'faster-whisper','is_mock':False},'warnings':[warning,scope_warning]},
                'provenance':{'kind':'censorship','matches':flagged,'language':language,'intervals':[{'start':a,'end':b} for a,b in intervals], 'stem_revisions':revisions(rows),'source_sha256':snapshot['source_sha256']}}
    except Exception:
        for path in (original, mix, tone): path.unlink(missing_ok=True)
        raise
    finally:
        input_audio.unlink(missing_ok=True)
