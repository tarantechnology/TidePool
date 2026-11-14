import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import uuid
from pathlib import Path

ROOT = Path(os.getenv('STORAGE_ROOT', 'storage')).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
DB = ROOT / 'soundcanvas.sqlite3'
MAX_UPLOAD = int(os.getenv('MAX_UPLOAD_BYTES', str(500 * 1024 * 1024)))
# Highest layer level, 5 = 500%. Exports end in a limiter so loud layers do not clip.
MAX_GAIN = 5.0
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('soundcanvas')

def event(name, **data):
    log.info(json.dumps({'event': name, **data}))

def uid(): return uuid.uuid4().hex

def connect():
    db = sqlite3.connect(DB, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('PRAGMA busy_timeout=30000')
    return db

# Columns added after the first release. The audio service reports the stem fields; they
# stay NULL for the fake adapter and for stems created before it returned them.
STEM_COLUMNS = (('revision', 'INTEGER NOT NULL DEFAULT 0'), ('role', 'TEXT'), ('source_stem_id', 'TEXT'),
                ('prompt', 'TEXT'), ('label', 'TEXT'), ('sample_rate', 'INTEGER'), ('channels', 'INTEGER'),
                ('format', 'TEXT'), ('spans', 'TEXT'), ('refinement', 'TEXT'), ('tags', 'TEXT'))
JOB_COLUMNS = (('error_code', 'TEXT'), ('retryable', 'INTEGER'), ('stage', 'TEXT'))

def init():
    with connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, status TEXT NOT NULL, filename TEXT, original TEXT, audio TEXT, metadata TEXT, visual TEXT);
        CREATE TABLE IF NOT EXISTS stems (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), name TEXT NOT NULL, path TEXT NOT NULL, duration REAL NOT NULL, start_offset REAL NOT NULL DEFAULT 0, volume REAL NOT NULL DEFAULT 1, muted INTEGER NOT NULL DEFAULT 0, solo INTEGER NOT NULL DEFAULT 0, automation TEXT NOT NULL DEFAULT '[]', mute_ranges TEXT NOT NULL DEFAULT '[]');
        CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), kind TEXT NOT NULL, status TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0, error TEXT, payload TEXT NOT NULL DEFAULT '{}', output TEXT);
        CREATE UNIQUE INDEX IF NOT EXISTS one_active ON jobs(project_id,kind) WHERE status IN ('queued','processing');
        CREATE TABLE IF NOT EXISTS messages (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        ''')
        for table, columns in (('stems', STEM_COLUMNS), ('jobs', JOB_COLUMNS)):
            present={row[1] for row in db.execute(f'PRAGMA table_info({table})')}
            for column, declaration in columns:
                if column not in present: db.execute(f'ALTER TABLE {table} ADD COLUMN {column} {declaration}')
        if "uploaded_original" not in {row[1] for row in db.execute("PRAGMA table_info(projects)")}:
            db.execute("ALTER TABLE projects ADD COLUMN uploaded_original TEXT")
            db.execute("UPDATE projects SET uploaded_original=original")
        db.execute("CREATE TABLE IF NOT EXISTS candidate_history (candidate_id TEXT PRIMARY KEY REFERENCES jobs(id), previous_rows TEXT NOT NULL, applied_revisions TEXT NOT NULL, undone INTEGER NOT NULL DEFAULT 0)")
        # Staged (vision) jobs restart from the queued stage; others keep their old 0 progress.
        db.execute("UPDATE jobs SET status='queued', progress=CASE WHEN stage IS NULL THEN 0 ELSE 5 END, stage=CASE WHEN stage IS NULL THEN NULL ELSE 'queued' END WHERE status='processing'")

def max_passes():
    """Upper bound for separation passes per request (MAX_PASSES, default 3)."""
    try: return max(1,int(os.getenv('MAX_PASSES','3')))
    except ValueError: return 3

def digest(token): return hashlib.sha256(token.encode()).hexdigest()

def project_dir(pid):
    if len(pid) != 32 or any(c not in '0123456789abcdef' for c in pid): raise ValueError('Invalid project ID')
    return ROOT / 'projects' / pid

def run(*args):
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True, timeout=int(os.getenv('FFMPEG_TIMEOUT', '600'))).stdout
    except (subprocess.SubprocessError, OSError) as e:
        raise RuntimeError('Media processing failed: ' + (getattr(e, 'stderr', '') or str(e))[-1000:]) from e

def probe(path):
    data = json.loads(run('ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)))
    audio = next((s for s in data['streams'] if s['codec_type']=='audio'), None)
    if not audio: raise ValueError('Media must contain an audio stream')
    info = {'duration': float(data['format'].get('duration') or audio.get('duration') or 0), 'sample_rate': int(audio['sample_rate']), 'channels': int(audio['channels']), 'has_video': any(s['codec_type']=='video' for s in data['streams'])}
    video = next((s for s in data['streams'] if s['codec_type']=='video'), None)
    if video:
        from fractions import Fraction
        try: fps = float(Fraction(video.get('avg_frame_rate', '0/0')))
        except (ValueError, ZeroDivisionError): fps = None
        info.update(width=video['width'], height=video['height'], frame_rate=fps)
    return info

def discard(pid, paths):
    """Delete the audio files of replaced stems once nothing can bring them back. Call only after
    the replacing transaction committed, so a failed job leaves the old files intact. Kept: files
    any stem row uses, the rows a still-possible undo would restore, and the rows and source a
    still-applicable preview would use. An undo or preview that the current revisions rule out
    can never run, so its files go too (this also clears ones deferred by an earlier call)."""
    try:
        directory=(project_dir(pid)/'stems').resolve()
        with connect() as db:
            current={r['id']:r['revision'] for r in db.execute('SELECT id,revision FROM stems WHERE project_id=?',(pid,))}
            keep={r['path'] for r in db.execute('SELECT path FROM stems WHERE project_id=?',(pid,))}
            # Undo and redo across server changes (app/checkpoints.py) can bring these back.
            from .checkpoints import kept_paths
            keep|=kept_paths(db,pid)
            doomed=set(paths)
            for h in db.execute('SELECT h.previous_rows,h.applied_revisions FROM candidate_history h JOIN jobs j ON j.id=h.candidate_id WHERE j.project_id=? AND h.undone=0',(pid,)):
                rows=[r['path'] for r in json.loads(h['previous_rows'])]
                (keep if json.loads(h['applied_revisions'])==current else doomed).update(rows)
            for j in db.execute("SELECT output FROM jobs WHERE project_id=? AND kind='candidate' AND status='completed' AND id NOT IN (SELECT candidate_id FROM candidate_history)",(pid,)):
                data=json.loads(j['output'] or 'null') or {}; snapshot=data.get('snapshot') or {}
                if {r['id']:r['revision'] for r in snapshot.get('rows',[])}==current:
                    keep.update([r['path'] for r in snapshot.get('rows',[])+data.get('proposed',[])]+[snapshot.get('source')])
        for path in doomed-keep:
            path=Path(path).resolve()
            if path.parent==directory: path.unlink(missing_ok=True)
    except (OSError,ValueError,TypeError,KeyError,sqlite3.Error) as exc:
        # The job already succeeded; a leftover file is only wasted space.
        event('stem_cleanup_failed',project_id=pid,error=str(exc))

def stem(row):
    d = dict(row)
    d['muted'], d['solo'] = bool(d['muted']), bool(d['solo'])
    d['automation'], d['mute_ranges'] = json.loads(d['automation']), json.loads(d['mute_ranges'])
    d['spans'] = json.loads(d['spans']) if d.get('spans') else None
    d['refinement'] = json.loads(d['refinement']) if d.get('refinement') else None
    d['tags'] = json.loads(d['tags']) if d.get('tags') else None
    d.pop('path')
    d['audio_url'] = f"/api/projects/{d['project_id']}/stems/{d['id']}/audio"
    return d
