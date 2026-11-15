"""Checkpoints: whole-project snapshots of the layer rows, so undo and redo reach across
server-side changes (applying a preview, Explode, a Current command, Moon pull, refine).

The editor takes one just before such a change and one just after it. Undo restores the
"before" snapshot and redo the "after" one, whatever happened to the faders in between, so
neither depends on revisions matching the way candidate undo does.

Mounted by app.main. A snapshot holds complete stem rows, so its audio files must outlive the
stems that replaced them; ``kept_paths`` lists them for core.discard where that exists.
"""
import json
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from .core import connect, project_dir, uid

router = APIRouter()

#: Snapshots kept per project. Older ones go, and with them the files only they used.
KEEP = 60


def ensure_table(db):
    db.execute('CREATE TABLE IF NOT EXISTS checkpoints (id TEXT PRIMARY KEY, project_id TEXT NOT NULL, '
               'label TEXT NOT NULL, rows TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP)')
    db.execute('CREATE INDEX IF NOT EXISTS checkpoints_project ON checkpoints(project_id)')
    # Highest revision each layer id ever had, so one that was removed and comes back still moves forward.
    db.execute('CREATE TABLE IF NOT EXISTS checkpoint_revisions (project_id TEXT NOT NULL, stem_id TEXT NOT NULL, revision INTEGER NOT NULL, PRIMARY KEY(project_id, stem_id))')


def kept_paths(db, pid):
    """Audio files a remaining checkpoint can bring back."""
    ensure_table(db)
    return {r['path'] for c in db.execute('SELECT rows FROM checkpoints WHERE project_id=?', (pid,)) for r in json.loads(c['rows'])}


def _guard(pid, authorization):
    from .main import auth
    auth(pid, authorization)


class CheckpointRequest(BaseModel):
    label: str = Field(default='', max_length=200)


@router.post('/api/projects/{pid}/checkpoints', status_code=201)
def create_checkpoint(pid: str, body: CheckpointRequest = CheckpointRequest(), authorization: str | None = Header(None)):
    _guard(pid, authorization)
    cid = uid()
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        ensure_table(db)
        rows = [dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?', (pid,))]
        db.execute('INSERT INTO checkpoints(id,project_id,label,rows) VALUES(?,?,?,?)', (cid, pid, body.label.strip(), json.dumps(rows)))
        db.execute('DELETE FROM checkpoints WHERE project_id=? AND id NOT IN (SELECT id FROM checkpoints WHERE project_id=? ORDER BY rowid DESC LIMIT ?)', (pid, pid, KEEP))
    return {'id': cid, 'stems': len(rows)}


@router.post('/api/projects/{pid}/checkpoints/{cid}/restore')
def restore_checkpoint(pid: str, cid: str, authorization: str | None = Header(None)):
    """Put the project's layers back exactly as the checkpoint saw them, faders and curves included."""
    _guard(pid, authorization)
    from .candidates import completed_job
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        ensure_table(db)
        row = db.execute('SELECT rows FROM checkpoints WHERE id=? AND project_id=?', (cid, pid)).fetchone()
        if not row: raise HTTPException(404, 'That step is too old to restore.')
        if db.execute("SELECT 1 FROM jobs WHERE project_id=? AND kind IN ('process','normalize','refine') AND status IN ('queued','processing')", (pid,)).fetchone():
            raise HTTPException(409, 'Wait for the running separation to finish, then try again.')
        rows = json.loads(row['rows'])
        for r in rows:
            path = Path(r['path']).resolve()
            if not path.is_relative_to(project_dir(pid).resolve()) or not path.is_file():
                raise HTTPException(409, f"The audio of “{r.get('label') or r['name']}” from that step is gone, so the layers were left as they are.")
        current = {r['id']: r['revision'] for r in db.execute('SELECT id,revision FROM stems WHERE project_id=?', (pid,))}
        for sid, rev in current.items():
            db.execute('INSERT INTO checkpoint_revisions(project_id,stem_id,revision) VALUES(?,?,?) ON CONFLICT(project_id,stem_id) DO UPDATE SET revision=MAX(revision,excluded.revision)', (pid, sid, rev))
        seen = {r['stem_id']: r['revision'] for r in db.execute('SELECT stem_id,revision FROM checkpoint_revisions WHERE project_id=?', (pid,))}
        db.execute('DELETE FROM stems WHERE project_id=?', (pid,))
        present = {r[1] for r in db.execute('PRAGMA table_info(stems)')}
        for r in rows:
            r = {k: v for k, v in r.items() if k in present}
            # Revisions always move forward, so a tab holding the old layers cannot overwrite these.
            r['revision'] = max(seen.get(r['id'], 0), r.get('revision') or 0) + 1
            keys = list(r)
            db.execute('INSERT INTO stems(' + ','.join(keys) + ') VALUES(' + ','.join('?' for _ in keys) + ')', list(r.values()))
        jid = completed_job(db, pid)
    return {'status': 'restored', 'job_id': jid, 'stems': len(rows)}
