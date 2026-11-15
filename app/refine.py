"""Per-stem refinement: rerun a target's prompt on the residual and fold the result back.

SAM Audio sometimes only attenuates a described sound, leaving a quieter copy in the
residual. Separating the residual again with the same prompt recovers more of it. Since
the service's residual is mixture minus targets, old target + residual equals
new target + new residual (target' = target + extracted, residual' = residual - extracted).
"""
import json
import shutil
import numpy as np
from .candidates import completed_job, safe_path
from .core import connect, discard, probe, project_dir, stem, uid
from .guidance import options
from .media import energy_db, read_wav, separate, write_wav
from .visual import prepare_payload


def is_residual(row):
    return row['role'] == 'residual' or row['name'].lower() == 'residual'


def origin(db, pid, sid):
    """Payload of the separation that created the stem (process or applied candidate), if any."""
    for job in db.execute("SELECT * FROM jobs WHERE project_id=? AND kind IN ('process','candidate') AND status='completed' ORDER BY rowid DESC", (pid,)):
        payload = json.loads(job['payload'] or '{}')
        if not payload.get('prompts') or not job['output']: continue  # Apply/undo bookkeeping jobs carry no request.
        output = json.loads(job['output'])
        if job['kind'] == 'process':
            ids = [s.get('id') for s in (output.get('stems', []) if isinstance(output, dict) else output or [])]
        else:
            applied = db.execute('SELECT undone FROM candidate_history WHERE candidate_id=?', (job['id'],)).fetchone()
            ids = [output.get('target', {}).get('id')] if applied and not applied['undone'] else []
        if sid in ids: return payload
    return None


def request(project, target, original, passes, job_id, warnings):
    text = target['prompt'] or target['name']
    anchors, visual, base = [], None, options('preview')
    if original is None:
        warnings.append('The separation that created this stem was not found; refining with its text prompt only.')
    else:
        anchors = list(original.get('anchors') or [])
        for prompt in original.get('prompts') or []:
            if isinstance(prompt, dict) and (prompt.get('name') == target['name'] or prompt.get('text') == text):
                anchors += prompt.get('anchors') or []
                break
        base = {k: v for k, v in (original.get('options') or base).items() if k != 'seed'}
        selection = (original.get('visual_prompt') or {}).get('selection')
        if selection:
            try:
                visual = prepare_payload(project, {'visual_prompt': {'selection': selection}})
            except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
                warnings.append(f'Visual guidance from the original separation could not be reused ({exc}); refined with text only.')
    payload = {'prompts': [text], 'anchors': anchors, 'visual_prompt': None, 'edit_scope': {'action': 'keep'},
               'profile': 'preview', 'options': {**base, 'passes': passes}, 'request_id': job_id}
    if visual:
        payload.update(visual_prompt=visual['visual_prompt'], source=visual['source'], asset_id=visual['asset_id'])
    return payload


def history(previous, block, requested, extracted, job_id):
    """Fold one refine job into the target's refinement record."""
    record = json.loads(previous) if previous else {'passes_requested': 1, 'passes_run': 1, 'per_pass_energy_db': [], 'stopped_early': False}
    block = block if isinstance(block, dict) else {}
    ran = block.get('passes_run') if isinstance(block.get('passes_run'), int) and block['passes_run'] >= 1 else requested
    energies = block.get('per_pass_energy_db')
    if not (isinstance(energies, list) and len(energies) == ran and all(isinstance(e, (int, float)) and not isinstance(e, bool) for e in energies)):
        energies = [extracted]
    record['passes_requested'] = int(record.get('passes_requested') or 0) + requested
    record['passes_run'] = int(record.get('passes_run') or 0) + ran
    record['per_pass_energy_db'] = list(record.get('per_pass_energy_db') or []) + list(energies)
    record['stopped_early'] = bool(block.get('stopped_early', False))
    record['refine_jobs'] = list(record.get('refine_jobs') or []) + [{'job_id': job_id, 'passes_run': ran, 'extracted_energy_db': extracted}]
    return record


def refine(project, payload, job_id):
    pid = project['id']; sid, rid = payload['stem_id'], payload['residual_stem_id']; passes = int(payload.get('passes') or 1)
    with connect() as db:
        rows = {r['id']: dict(r) for r in db.execute('SELECT * FROM stems WHERE project_id=? AND id IN (?,?)', (pid, sid, rid))}
        original = origin(db, pid, sid)
    if set(rows) != {sid, rid} or {k: rows[k]['revision'] for k in rows} != payload['revisions']:
        raise ValueError('The stem or residual changed while refinement was queued; nothing was changed. Retry after reloading.')
    target, residual = rows[sid], rows[rid]
    if target['start_offset'] != 0 or residual['start_offset'] != 0:
        raise ValueError('Refining requires the stem and residual to start at zero; nothing was changed.')
    warnings = []
    body = request(project, target, original, passes, job_id, warnings)
    scratch = project_dir(pid) / 'stems' / ('refine-' + job_id)
    created = []
    try:
        inference = {}
        results = separate(safe_path(pid, residual['path']), scratch, body, report=inference)
        results = [r if isinstance(r, dict) else {'name': r[0], 'path': r[1]} for r in results]
        found = [r for r in results if r.get('role') != 'residual' and r['name'].lower() != 'residual']
        rest = [r for r in results if r.get('role') == 'residual' or r['name'].lower() == 'residual']
        if len(found) != 1 or len(rest) != 1:
            raise ValueError('Refinement requires one target and one residual from the audio service; existing stems were preserved.')
        old, rate = read_wav(target['path'])
        extracted, extracted_rate = read_wav(found[0]['path'])
        remainder, remainder_rate = read_wav(rest[0]['path'])
        if (extracted_rate, extracted.shape[1]) != (rate, old.shape[1]) or (remainder_rate, remainder.shape[1]) != (rate, old.shape[1]):
            raise ValueError('Refined audio does not match the stem sample rate and channels; existing stems were preserved.')
        if len(extracted) != len(old):
            warnings.append(f'Refined target had {len(extracted)} samples, the stem {len(old)}; '
                            + ('padded with silence.' if len(extracted) < len(old) else 'trimmed to the stem length, so the stems no longer sum exactly to the previous mix.'))
            aligned = np.zeros_like(old); aligned[:min(len(old), len(extracted))] = extracted[:len(old)]
            extracted = aligned
        level = energy_db(extracted)
        new_target = project_dir(pid) / 'stems' / (uid() + '.wav'); created.append(new_target)
        write_wav(new_target, old + extracted, rate)
        new_residual = project_dir(pid) / 'stems' / (uid() + '.wav'); created.append(new_residual)
        write_wav(new_residual, remainder, rate)
        record = history(target['refinement'], found[0].get('refinement'), passes, level, job_id)
        target_duration, residual_duration = probe(new_target)['duration'], probe(new_residual)['duration']
        with connect() as db:
            db.execute('BEGIN IMMEDIATE')
            current = {r['id']: r['revision'] for r in db.execute('SELECT id,revision FROM stems WHERE project_id=? AND id IN (?,?)', (pid, sid, rid))}
            if current != payload['revisions']:
                raise ValueError('The stem or residual was edited during refinement; existing stems were preserved. Retry after reloading.')
            # Mix edits (volume, mute, solo, automation, ranges) stay; only the audio changes.
            db.execute('UPDATE stems SET path=?,duration=?,refinement=?,revision=revision+1 WHERE id=? AND project_id=?',
                       (str(new_target), target_duration, json.dumps(record), sid, pid))
            db.execute('UPDATE stems SET path=?,duration=?,revision=revision+1 WHERE id=? AND project_id=?',
                       (str(new_residual), residual_duration, rid, pid))
            stems_job = completed_job(db, pid)
            updated = [stem(r) for r in db.execute('SELECT * FROM stems WHERE project_id=? AND id IN (?,?)', (pid, sid, rid))]
            output = {'stem_id': sid, 'residual_stem_id': rid, 'extracted_energy_db': level, 'passes': passes,
                      'refinement': record, 'stems': updated, 'stems_job_id': stems_job, 'warnings': warnings, 'inference': inference}
    except Exception:
        for path in created: path.unlink(missing_ok=True)
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    shutil.rmtree(scratch, ignore_errors=True)
    # Committed: the replaced files go before the job reports completion, so a client that sees
    # "completed" never sees the old files still on disk.
    discard(pid, [target['path'], residual['path']])
    with connect() as db:
        db.execute("UPDATE jobs SET status='completed',progress=100,output=? WHERE id=?", (json.dumps(output), job_id))
