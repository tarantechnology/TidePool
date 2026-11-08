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

def init():
    with connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, status TEXT NOT NULL, filename TEXT, original TEXT, audio TEXT, metadata TEXT, visual TEXT);
        CREATE TABLE IF NOT EXISTS stems (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), name TEXT NOT NULL, path TEXT NOT NULL, duration REAL NOT NULL, start_offset REAL NOT NULL DEFAULT 0, volume REAL NOT NULL DEFAULT 1, muted INTEGER NOT NULL DEFAULT 0, solo INTEGER NOT NULL DEFAULT 0, automation TEXT NOT NULL DEFAULT '[]', mute_ranges TEXT NOT NULL DEFAULT '[]');
        CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), kind TEXT NOT NULL, status TEXT NOT NULL, progress INTEGER NOT NULL DEFAULT 0, error TEXT, payload TEXT NOT NULL DEFAULT '{}', output TEXT);
        CREATE UNIQUE INDEX IF NOT EXISTS one_active ON jobs(project_id,kind) WHERE status IN ('queued','processing');
        CREATE TABLE IF NOT EXISTS messages (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        ''')
        if "revision" not in {row[1] for row in db.execute("PRAGMA table_info(stems)")}:
            db.execute("ALTER TABLE stems ADD COLUMN revision INTEGER NOT NULL DEFAULT 0")
        db.execute("UPDATE jobs SET status='queued', progress=0 WHERE status='processing'")

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
    return {'duration': float(data['format'].get('duration') or audio.get('duration') or 0), 'sample_rate': int(audio['sample_rate']), 'channels': int(audio['channels']), 'has_video': any(s['codec_type']=='video' for s in data['streams'])}

def stem(row):
    d = dict(row)
    d['muted'], d['solo'] = bool(d['muted']), bool(d['solo'])
    d['automation'], d['mute_ranges'] = json.loads(d['automation']), json.loads(d['mute_ranges'])
    d.pop('path')
    d['audio_url'] = f"/api/projects/{d['project_id']}/stems/{d['id']}/audio"
    return d
