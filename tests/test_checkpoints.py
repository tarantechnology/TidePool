"""Undo and redo across server-side changes: a checkpoint before and after, restored at will."""
import subprocess
from tests.test_app import client, wait  # noqa: F401  (fixture)


def _project(client, tmp_path):
    p = client.post('/api/projects').json(); pid = p['project_id']; h = {'Authorization': 'Bearer ' + p['access_token']}
    video = tmp_path / 'in.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'color=c=blue:s=64x48:r=10:d=1', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=1', '-c:v', 'mpeg4', '-c:a', 'aac', '-shortest', str(video)], check=True)
    with video.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload', headers=h, files={'file': ('in.mp4', f, 'video/mp4')}).status_code == 200
    return pid, h


def test_undo_and_redo_a_separation_even_after_fader_moves(client, tmp_path):
    pid, h = _project(client, tmp_path)
    before = client.post(f'/api/projects/{pid}/checkpoints', headers=h, json={'label': 'before'}).json()
    assert before['stems'] == 0
    r = client.post(f'/api/projects/{pid}/process', headers=h, json={'prompts': ['dog']})
    job = wait(client, '/api/jobs/' + r.json()['job_id'], h); assert job['status'] == 'completed'
    after = client.post(f'/api/projects/{pid}/checkpoints', headers=h, json={'label': 'Separate dog'}).json()
    assert after['stems'] == len(job['stems']) > 0
    sid = job['stems'][0]['id']
    # A fader move after the change no longer blocks undo.
    assert client.patch(f'/api/projects/{pid}/stems/{sid}', headers=h, json={'volume': 0.3}).status_code == 200

    undone = client.post(f'/api/projects/{pid}/checkpoints/{before["id"]}/restore', headers=h)
    assert undone.status_code == 200, undone.text
    assert client.get(f'/api/projects/{pid}/stems', headers=h).json() == []

    redone = client.post(f'/api/projects/{pid}/checkpoints/{after["id"]}/restore', headers=h).json()
    stems = client.get(f'/api/projects/{pid}/stems', headers=h).json()
    assert {s['id'] for s in stems} == {s['id'] for s in job['stems']}
    restored = next(s for s in stems if s['id'] == sid)
    assert restored['volume'] == job['stems'][0]['volume']  # as it was right after the change
    assert restored['revision'] > 1  # moved forward past the fader edit
    assert client.get(f'/api/jobs/{redone["job_id"]}', headers=h).json()['status'] == 'completed'
    # A stale client is refused by the usual revision check.
    assert client.patch(f'/api/projects/{pid}/stems/{sid}', headers=h, json={'volume': 0.9, 'expected_revision': 1}).status_code == 409


def test_restore_refuses_other_projects_and_missing_audio(client, tmp_path):
    pid, h = _project(client, tmp_path)
    r = client.post(f'/api/projects/{pid}/process', headers=h, json={'prompts': ['dog']})
    job = wait(client, '/api/jobs/' + r.json()['job_id'], h)
    cp = client.post(f'/api/projects/{pid}/checkpoints', headers=h).json()
    other, oh = _project(client, tmp_path)
    assert client.post(f'/api/projects/{other}/checkpoints/{cp["id"]}/restore', headers=oh).status_code == 404
    assert client.post(f'/api/projects/{pid}/checkpoints/{cp["id"]}/restore').status_code == 401
    from pathlib import Path
    import app.core as core, sqlite3
    with sqlite3.connect(core.DB) as db: path = db.execute('SELECT path FROM stems WHERE project_id=?', (pid,)).fetchone()[0]
    Path(path).unlink()
    res = client.post(f'/api/projects/{pid}/checkpoints/{cp["id"]}/restore', headers=h)
    assert res.status_code == 409 and 'gone' in res.json()['detail']


def test_a_resplit_keeps_the_audio_an_undo_needs(client, tmp_path):
    pid, h = _project(client, tmp_path)
    first = wait(client, '/api/jobs/' + client.post(f'/api/projects/{pid}/process', headers=h, json={'prompts': ['dog']}).json()['job_id'], h)
    assert first['status'] == 'completed'
    before = client.post(f'/api/projects/{pid}/checkpoints', headers=h, json={'label': 'Before: split again'}).json()
    second = wait(client, '/api/jobs/' + client.post(f'/api/projects/{pid}/process', headers=h, json={'prompts': ['music']}).json()['job_id'], h)
    assert second['status'] == 'completed'
    r = client.post(f'/api/projects/{pid}/checkpoints/{before["id"]}/restore', headers=h)
    assert r.status_code == 200, r.text
    names = {s['name'] for s in client.get(f'/api/projects/{pid}/stems', headers=h).json()}
    assert names == {s['name'] for s in first['stems']}
