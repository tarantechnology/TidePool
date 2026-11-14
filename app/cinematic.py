"""Reversible, bounded whole-project audio treatment; no generated content."""
from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path

from .candidates import revisions, safe_path
from .core import probe, project_dir, run, uid
from .media import export
from .visual import file_sha256

# Explicit, bounded DSP recipes. The agent never supplies filter expressions.
STYLES = {
    'subtle': {'bass': 0.5, 'treble': 0.3, 'ratio': 1.1},
    'cinematic': {'bass': 1.0, 'treble': 0.6, 'ratio': 1.3},
    'dramatic': {'bass': 1.6, 'treble': 0.8, 'ratio': 1.5},
}


def analyze(path: Path) -> dict:
    """Measure the mix; these numbers do not classify sound sources."""
    result = subprocess.run(['ffmpeg', '-hide_banner', '-nostats', '-i', str(path), '-af', 'volumedetect', '-f', 'null', '-'], capture_output=True, text=True, timeout=600)
    if result.returncode:
        raise ValueError('Could not analyze the current mix.')
    def level(key):
        match = re.search(rf'{key}_volume:\s*(-?(?:\d+(?:\.\d+)?|inf))\s*dB', result.stderr)
        if not match: raise ValueError('Could not measure the current mix.')
        value = float(match.group(1))
        return round(value, 1) if math.isfinite(value) else None
    return {'mean_dbfs': level('mean'), 'peak_dbfs': level('max')}


def build(project: dict, payload: dict, candidate_id: str) -> dict:
    pid = project['id']; snapshot = payload['_candidate']; style = payload['style']
    if style not in STYLES: raise ValueError('Unsupported cinematic style.')
    source = safe_path(pid, snapshot['source'])
    if file_sha256(source) != snapshot['source_sha256']: raise ValueError('Source changed; generate another preview.')
    rows = snapshot['rows']
    for row in rows:
        if file_sha256(safe_path(pid, row['path'])) != snapshot['stem_hashes'][row['id']]:
            raise ValueError('A layer changed; generate another preview.')
    directory = project_dir(pid) / 'stems'; directory.mkdir(parents=True, exist_ok=True)
    created = []
    def new_file():
        path = directory / (uid() + '.wav'); created.append(path); return path
    try:
        original = new_file()
        if rows: export(rows, source, original, 'wav', False)
        else: shutil.copyfile(source, original)
        analysis = analyze(original)
        if analysis['mean_dbfs'] is None or analysis['mean_dbfs'] < -70:
            raise ValueError('The current mix is silent or nearly silent. Unmute a layer or choose an audible recording.')
        params = STYLES[style]
        # Peak-aware headroom is conservative, not loudness normalization.
        headroom = 0.80 if analysis['peak_dbfs'] is not None and analysis['peak_dbfs'] >= -1.0 else 0.85
        filters = (f"highpass=f=45,bass=g={params['bass']}:f=120:w=0.6,"
                   f"treble=g={params['treble']}:f=4000:w=0.7,"
                   f"acompressor=threshold=0.25:ratio={params['ratio']}:attack=20:release=250:makeup=1,volume={headroom}")
        proposed = []
        sources = rows or [{'id':uid(), 'project_id':pid, 'name':'Cinematic mix', 'path':str(source),
                            'duration':probe(source)['duration'], 'start_offset':0, 'volume':1, 'muted':0, 'solo':0,
                            'automation':'[]', 'mute_ranges':'[]', 'revision':0}]
        for row in sources:
            new_path = new_file()
            run('ffmpeg', '-v', 'error', '-y', '-i', str(safe_path(pid, row['path'])), '-af', filters, '-c:a', 'pcm_f32le', str(new_path))
            before, after = probe(row['path']), probe(new_path)
            if (before['sample_rate'], before['channels']) != (after['sample_rate'], after['channels']) or abs(before['duration']-after['duration']) > 1/before['sample_rate']:
                raise ValueError('Processed audio no longer matches the source timeline.')
            proposed.append({**row, 'path':str(new_path)})
        mix = new_file(); export(proposed, source, mix, 'wav', False)
        after_analysis = analyze(mix)
        warnings = ['This is a whole-mix treatment, not a dialogue, music, or sound-effects separation.',
                    'EQ and compression can affect any sound in the treated layers.']
        if analysis['peak_dbfs'] is not None and analysis['peak_dbfs'] >= -0.1:
            warnings.append('The source reaches full scale; lost detail cannot be restored.')
        if after_analysis['peak_dbfs'] is not None and after_analysis['peak_dbfs'] >= -0.1:
            warnings.append('The preview reaches full scale; listen for clipping and reduce levels if needed.')
        return {'target':{'path':str(mix)}, 'mix':{'path':str(mix)}, 'original':{'path':str(original)},
                'snapshot':snapshot, 'proposed':proposed,
                'inference':{'model':{'is_mock':False, 'backend':'cinematic-dsp'}, 'warnings':warnings},
                'provenance':{'kind':'cinematic', 'style':style, 'analysis':analysis, 'preview_analysis':after_analysis,
                              'stem_revisions':revisions(rows), 'source_sha256':snapshot['source_sha256'],
                              'steps':['Remove sub-bass below 45 Hz', 'Gentle low and high EQ', 'Light compression', 'Reserve headroom'], 'headroom_gain':headroom}}
    except Exception:
        for path in created: path.unlink(missing_ok=True)
        raise
