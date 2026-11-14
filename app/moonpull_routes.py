"""HTTP routes for Moon pull (see app/moonpull.py). Mounted by app.main."""
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from . import moonpull as mp
from .core import connect, stem

router = APIRouter()

Range = tuple[float, float]


def _guard(pid, authorization):
    from .main import auth, refuse_during_refine
    auth(pid, authorization); refuse_during_refine(pid)


def _stem_row(pid, sid):
    from .main import find_stem
    return find_stem(pid, sid)


def _check_revision(row, expected):
    if expected is not None and row['revision'] != expected:
        raise HTTPException(409, 'This layer changed since it was loaded; reload and try again')


def _swap(pid, sid, row, dest, rule):
    """Point the stem at ``dest`` if nothing changed while it rendered, and note what it replaced."""
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        cur = db.execute('SELECT revision FROM stems WHERE project_id=? AND id=?', (pid, sid)).fetchone()
        if not cur or cur['revision'] != row['revision']:
            Path(dest).unlink(missing_ok=True)
            raise HTTPException(409, 'This layer changed while the rule was rendering; try again')
        mp.write_note(dest, row['path'], rule)
        db.execute('UPDATE stems SET path=?,revision=revision+1 WHERE project_id=? AND id=?', (str(dest), pid, sid))
        out = stem(db.execute('SELECT * FROM stems WHERE project_id=? AND id=?', (pid, sid)).fetchone())
    # Names this change, so a later revert can tell it is still the latest one on the layer.
    out['moonpull_file'] = Path(dest).name
    return out


def _render(fn):
    try: return fn()
    except ValueError as exc: raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc: raise HTTPException(500, str(exc)) from exc


class EffectsRequest(BaseModel):
    effect: Literal['deepen', 'impact', 'disorient', 'muffle']
    ranges: list[Range] = Field(min_length=1, max_length=mp.MAX_RANGES)
    amount: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)
    expected_revision: int | None = Field(default=None, ge=0)
    rule: str | None = Field(default=None, max_length=120)


@router.post('/api/projects/{pid}/stems/{sid}/moonpull/effect')
def moonpull_effect(pid: str, sid: str, body: EffectsRequest, authorization: str | None = Header(None)):
    """One effect over every moment of a rule, in timeline seconds. Revertible."""
    _guard(pid, authorization)
    row = _stem_row(pid, sid); _check_revision(row, body.expected_revision)
    ranges = _render(lambda: mp.to_stem_ranges(body.ranges, row['start_offset'], row['duration']))
    dest = _render(lambda: mp.apply_effect_ranges(row['path'], body.effect, ranges, body.amount))
    return _swap(pid, sid, row, dest, body.rule)


class EchoRequest(BaseModel):
    source_stem_id: str
    #: The motif sound in timeline seconds. Absent: the loudest stretch of the source layer.
    source_range: Range | None = None
    times: list[float] = Field(min_length=1, max_length=mp.MAX_RANGES)
    gain: float = Field(default=1.0, ge=0, le=2, allow_inf_nan=False)
    expected_revision: int | None = Field(default=None, ge=0)
    rule: str | None = Field(default=None, max_length=120)


@router.post('/api/projects/{pid}/stems/{sid}/moonpull/echo')
def moonpull_echo(pid: str, sid: str, body: EchoRequest, authorization: str | None = Header(None)):
    """Place the motif sound on this layer at every time. Revertible."""
    _guard(pid, authorization)
    row = _stem_row(pid, sid); _check_revision(row, body.expected_revision)
    source = _stem_row(pid, body.source_stem_id)
    if body.source_range is None: a, b = mp.loudest_window(source['path'])
    else:
        a, b = body.source_range[0] - source['start_offset'], body.source_range[1] - source['start_offset']
    times = [t - row['start_offset'] for t in body.times]
    dest = _render(lambda: mp.apply_echo(row['path'], source['path'], (a, b), times, body.gain))
    return _swap(pid, sid, row, dest, body.rule)


@router.get('/api/projects/{pid}/stems/{sid}/moonpull/motif')
def moonpull_motif(pid: str, sid: str, seconds: float = 2.0, authorization: str | None = Header(None)):
    """Where the loudest ``seconds`` of this layer are, in timeline seconds: the default motif sound."""
    from .main import auth
    auth(pid, authorization)
    row = _stem_row(pid, sid)
    a, b = mp.loudest_window(row['path'], min(max(seconds, 0.2), 10.0))
    return {'start': round(a + row['start_offset'], 3), 'end': round(b + row['start_offset'], 3)}


class RevertRequest(BaseModel):
    #: The moonpull_file the change returned. Fader and curve edits since then do not matter.
    file: str = Field(min_length=1, max_length=200)


@router.post('/api/projects/{pid}/stems/{sid}/moonpull/revert')
def moonpull_revert(pid: str, sid: str, body: RevertRequest, authorization: str | None = Header(None)):
    """Undo a Moon pull change, if it is still the latest change to this layer's audio."""
    _guard(pid, authorization)
    row = _stem_row(pid, sid)
    if Path(row['path']).name != body.file:
        raise HTTPException(409, 'The audio of this layer changed after the rule was applied, so it was left as it is')
    previous = mp.previous_of(row['path'])
    if not previous: raise HTTPException(409, 'The file this rule replaced is gone, so the layer was left as it is')
    with connect() as db:
        db.execute('BEGIN IMMEDIATE')
        cur = db.execute('SELECT path FROM stems WHERE project_id=? AND id=?', (pid, sid)).fetchone()
        if cur['path'] != row['path']: raise HTTPException(409, 'This layer changed while reverting; try again')
        db.execute('UPDATE stems SET path=?,revision=revision+1 WHERE project_id=? AND id=?', (previous, pid, sid))
        return stem(db.execute('SELECT * FROM stems WHERE project_id=? AND id=?', (pid, sid)).fetchone())


@router.post('/api/projects/{pid}/stems/{sid}/moonpull/transcript')
def moonpull_transcript(pid: str, sid: str, authorization: str | None = Header(None)):
    """Timed lines and words of a layer, so a rule can follow what is said."""
    from .main import auth
    auth(pid, authorization)
    row = _stem_row(pid, sid)
    try: return mp.transcribe(row['path'], row['start_offset'])
    except mp.TranscriptError as exc: raise HTTPException(503, str(exc)) from exc
    except RuntimeError as exc: raise HTTPException(500, str(exc)) from exc
