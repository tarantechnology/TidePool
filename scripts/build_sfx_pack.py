#!/usr/bin/env python3
"""Build the TidePool comedy/meme SFX pack into $STORAGE_ROOT/sfx (or --out).

    python scripts/build_sfx_pack.py                   # synthesize/download every sound (skips existing files)
    python scripts/build_sfx_pack.py --force           # rebuild everything
    python scripts/build_sfx_pack.py --write-manifest  # re-measure the procedural sounds into app/sfx_manifest.json
    python scripts/build_sfx_pack.py --check           # verify manifest <-> recipes <-> licences, build nothing

Procedural sounds (app/sfx_synth.py) are ours and CC0-1.0. Downloaded sounds must be listed in
the manifest with source "download", a direct `url`, its `sha256`, and a licence in
ALLOWED_LICENCES (CC0-1.0 only); anything else is refused. Audio never goes into git.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AUTHOR = 'TidePool contributors (procedural synthesis)'
SOURCE_URL = 'https://github.com/tarantechnology/TidePool/blob/main/app/sfx_synth.py'


def measure(samples):
    from app.sfx_dsp import db, loudness_lufs, true_peak
    return {'duration': round(len(samples) / 48000, 3), 'channels': int(samples.shape[1]), 'sample_rate': 48000,
            'loudness_lufs': round(loudness_lufs(samples), 1), 'true_peak_dbtp': round(db(true_peak(samples)), 1)}


def procedural_entry(recipe, samples):
    return {'id': recipe['id'], 'title': recipe['title'], 'category': recipe['category'], 'tags': recipe['tags'],
            'aliases': recipe['aliases'], 'licence': 'CC0-1.0', 'source': 'procedural',
            'source_url': f"{SOURCE_URL}#{recipe['id']}", 'author': AUTHOR, 'file': recipe['id'] + '.flac',
            'hit': recipe['hit'], **measure(samples)}


def write_manifest():
    from app.sfx_library import MANIFEST
    from app.sfx_synth import RECIPES, synthesize
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {'sounds': []}
    kept = [s for s in old['sounds'] if s.get('source') != 'procedural']
    entries = [procedural_entry(r, synthesize(sid)) for sid, r in RECIPES.items()] + kept
    data = {'pack': 'tidepool-meme-sfx', 'version': 1, 'licence_policy': 'Only CC0-1.0 assets; procedural sounds are synthesized by app/sfx_synth.py.',
            'sounds': sorted(entries, key=lambda s: (s['category'], s['id']))}
    MANIFEST.write_text(json.dumps(data, indent=1) + '\n')
    print(f'wrote {MANIFEST.relative_to(ROOT)}: {len(entries)} sounds')


def check():
    from app.sfx_library import ALLOWED_LICENCES, sounds
    from app.sfx_synth import RECIPES
    problems = []
    ids = [s['id'] for s in sounds()]
    if len(ids) != len(set(ids)):
        problems.append('duplicate ids')
    for s in sounds():
        if s['licence'] not in ALLOWED_LICENCES:
            problems.append(f"{s['id']}: licence {s['licence']} is not allowed")
        if s['source'] == 'procedural' and s['id'] not in RECIPES:
            problems.append(f"{s['id']}: no recipe")
        elif s['source'] == 'procedural' and any(s[k] != RECIPES[s['id']][k] for k in ('title', 'category', 'tags', 'aliases', 'hit')):
            problems.append(f"{s['id']}: manifest metadata differs from its recipe (run --write-manifest)")
        if s['source'] == 'download' and not (s.get('url') and s.get('sha256')):
            problems.append(f"{s['id']}: download needs url and sha256")
    missing = set(RECIPES) - set(ids)
    if missing:
        problems.append(f'recipes missing from the manifest: {sorted(missing)} (run --write-manifest)')
    for p in problems:
        print('PROBLEM', p)
    return not problems


def download(entry, path):
    from app.core import run
    from app.sfx_library import ALLOWED_LICENCES
    if entry['licence'] not in ALLOWED_LICENCES:
        raise SystemExit(f"refusing {entry['id']}: licence {entry['licence']}")
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / 'download'
        with urllib.request.urlopen(entry['url'], timeout=60) as response:
            raw.write_bytes(response.read(20 * 1024 * 1024))
        digest = hashlib.sha256(raw.read_bytes()).hexdigest()
        if digest != entry['sha256']:
            raise SystemExit(f"{entry['id']}: sha256 mismatch ({digest})")
        out = Path(tmp) / 'out.flac'
        run('ffmpeg', '-v', 'error', '-y', '-i', str(raw), '-t', '15', '-ar', '48000', '-c:a', 'flac', '-sample_fmt', 's16', str(out))
        out.replace(path)


def build(out, force):
    from app.sfx_library import sounds, write_procedural
    out.mkdir(parents=True, exist_ok=True)
    built = 0
    for entry in sounds():
        path = out / entry['file']
        if path.is_file() and not force:
            continue
        if entry['source'] == 'procedural':
            write_procedural(entry['id'], path)
        else:
            download(entry, path)
        built += 1
    size = sum(p.stat().st_size for p in out.glob('*.flac'))
    print(f'{out}: built {built}, {len(list(out.glob("*.flac")))} files, {size / 1e6:.1f} MB')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, default=None, help='pack directory (default $STORAGE_ROOT/sfx)')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--write-manifest', action='store_true')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
    if not check():
        raise SystemExit(1)
    if args.check:
        return
    if args.out:
        os.environ['SOUNDCANVAS_SFX_PACK_DIR'] = str(args.out)
    from app.sfx_library import pack_dir
    build(pack_dir(), args.force)


if __name__ == '__main__':
    main()
