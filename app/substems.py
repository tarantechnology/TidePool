"""Separate the rest: replace one music stem with the instruments inside it, plus a 'Rest'."""
import json
from .candidates import completed_job, revisions, safe_path
from .core import connect, discard, probe, project_dir, stem, uid
from .media import separate
from .worker import ticking

# Fields a child takes from its parent, so the mix is unchanged until the user edits a child.
INHERITED = ('start_offset', 'volume', 'muted', 'solo', 'automation', 'mute_ranges')


def split(project, payload, job_id):
    pid = project['id']; sid = payload['stem_id']
    with connect() as db: parent = db.execute('SELECT * FROM stems WHERE project_id=? AND id=?', (pid, sid)).fetchone()
    if not parent or parent['revision'] != payload['revision']: raise ValueError('The track changed while the split was queued; nothing was changed. Retry after reloading.')
    body = {'prompts': [], 'request_id': job_id, 'options': {'mode': 'music_substems', 'parent_label': parent['label'] or parent['name'],
            'stems': payload.get('stems'), 'output_format': 'pcm_f32le'}}
    inference = {}
    duration = probe(safe_path(pid, parent['path']))['duration']
    with ticking(job_id, 'splitting', 0.15 * duration + 8):
        results = separate(parent['path'], project_dir(pid) / 'stems', body, report=inference)
    results = [r if isinstance(r, dict) else {'name': r[0], 'path': r[1]} for r in results]
    if sum(r.get('role') == 'residual' for r in results) != 1 or len(results) < 2:
        raise ValueError('The split must return at least one instrument and exactly one Rest; the track was preserved.')
    children = []
    for r in results:
        info = probe(r['path'])
        children.append({'id': uid(), 'project_id': pid, 'name': r['name'], 'path': str(r['path']), 'duration': info['duration'],
                         'role': r.get('role') or 'target', 'source_stem_id': sid, 'prompt': r.get('prompt'), 'label': r.get('label'),
                         'sample_rate': r.get('sample_rate'), 'channels': r.get('channels'), 'format': r.get('format'),
                         'tags': json.dumps(r['tags']) if r.get('tags') else None, 'revision': 0, **{k: parent[k] for k in INHERITED},
                         'parent_name': parent['name']})  # The editor groups the children under it.
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        current = [dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?', (pid,))]
        now = next((r for r in current if r['id'] == sid), None)
        if not now or now['revision'] != payload['revision']: raise ValueError('The track was edited during the split; it was preserved. Retry after reloading.')
        db.execute('DELETE FROM stems WHERE project_id=? AND id=?', (pid, sid))
        for child in children:
            db.execute('INSERT INTO stems(' + ','.join(child) + ') VALUES(' + ','.join('?' for _ in child) + ')', list(child.values()))
        after = [dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=?', (pid,))]
        # Undo restores the parent through the same history the preview apply uses.
        db.execute('INSERT INTO candidate_history(candidate_id,previous_rows,applied_revisions) VALUES(?,?,?)', (job_id, json.dumps(current), json.dumps(revisions(after))))
        stems_job = completed_job(db, pid)
        output = {'stem_id': sid, 'stems': [stem(r) for r in db.execute('SELECT * FROM stems WHERE project_id=? AND source_stem_id=?', (pid, sid))],
                  'stems_job_id': stems_job, 'inference': inference, 'routing': inference.get('routing')}
        db.execute("UPDATE jobs SET status='completed',progress=100,stage=NULL,output=? WHERE id=?", (json.dumps(output), job_id))
    discard(pid, [parent['path']])  # Kept while undo can restore it.
