"""Replaced stem audio is deleted after the replacing write commits, unless something can restore it."""
from pathlib import Path
from test_app import client, wait  # noqa: F401  (fixture)
from test_guided_preview import preview, uploaded


def paths(pid):
    import app.core as core
    with core.connect() as db: return {r['path'] for r in db.execute('SELECT path FROM stems WHERE project_id=?',(pid,))}


def wavs(pid):
    import app.core as core
    return {str(p) for p in (core.ROOT/'projects'/pid/'stems').glob('*.wav')}


def split(client, pid, h):
    import time
    jid=client.post(f'/api/projects/{pid}/process',headers=h,json={'mode':'music_stems'}).json()['job_id']
    job=wait(client,'/api/jobs/'+jid,h)
    # The worker deletes replaced files just after it reports completion.
    for _ in range(40):
        if wavs(pid)==paths(pid) or job['status']!='completed': break
        time.sleep(.05)
    return job


def test_resplit_leaves_only_the_current_stems_files(client, tmp_path):
    pid,h=uploaded(client,tmp_path)
    assert split(client,pid,h)['status']=='completed'
    first=paths(pid)
    assert split(client,pid,h)['status']=='completed'
    assert wavs(pid)==paths(pid) and len(paths(pid))==4 and not first & paths(pid)


def test_failed_job_keeps_the_old_files(client, tmp_path, monkeypatch):
    import app.core as core
    import app.worker as worker
    from app.media import separate
    pid,h=uploaded(client,tmp_path)
    split(client,pid,h); old=paths(pid)
    def racing(*args, **kwargs):
        # Someone edits a layer mid-job, so the replacing transaction refuses to commit.
        with core.connect() as db: db.execute('UPDATE stems SET revision=revision+1 WHERE project_id=?',(pid,))
        return separate(*args, **kwargs)
    monkeypatch.setattr(worker,'separate',racing)
    assert split(client,pid,h)['status']=='failed'
    assert paths(pid)==old and all(Path(p).is_file() for p in old)


def test_undoable_candidate_keeps_the_files_undo_restores(client, tmp_path):
    pid,h=uploaded(client,tmp_path)
    first,_=preview(client,pid,h)
    assert client.post(f'/api/candidates/{first}/apply',headers=h).status_code==200
    before=paths(pid)
    second,_=preview(client,pid,h,prompt='music')
    applied=client.post(f'/api/candidates/{second}/apply',headers=h).json()
    replaced=before-paths(pid)
    assert replaced and all(Path(p).is_file() for p in replaced)  # undo can still restore them
    assert client.post(f'/api/candidates/{second}/undo',headers=h).status_code==200
    assert paths(pid)==before and all(Path(p).is_file() for p in before)
    assert applied['job_id']
    # Once a re-split rules undo out, the old layers' files are gone too.
    split(client,pid,h)
    assert not before & wavs(pid) and paths(pid) <= wavs(pid)
