"""Several edits from one chat prompt ("reverb on the vocals, drums like a phone, add a piano part and a vine boom at 0:12").

Planner: ``kind: 'multi'`` with ``steps``; each step is one of the existing single-edit kinds (fx, duck, loudness,
insert, generate, edit) carrying that kind's own object, so every step goes through that kind's existing validator.

Execution (``multi-preview`` -> ONE candidate, so compare / apply / undo stay atomic and revision-checked):

1. In parallel (bounded pool, ``MAX_WORKERS``): one render per affected layer (all of that layer's effect steps,
   in prompt order, appended to its chip list and rendered once from the dry audio, so chips still work), one
   audio-service call per generate step, and one anchor resolution for all inserts (word timings fetched once).
   numpy/ffmpeg renders release the GIL; the audio service serialises per model on its side.
2. Then, in prompt order: new generated layers, inserted sound layers (and their optional ducking), layer edits
   (gain, mute, solo, automation, timed mutes) and auto-ducking. Loudness normalization runs last on the result.

Generated parts follow the song as it was before this prompt's other edits (the service analyses the current mix
and stems). Tempo/speed changes, "more/less" adjustments, separations and bleeping are not allowed inside a multi
plan: they change the timeline or depend on an earlier preview, so they are asked for on their own.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait

from . import fx as F
from .candidates import revisions, safe_path
from .core import connect, probe, project_dir, uid
from .media import export

MAX_STEPS = 8
MAX_GENERATE = 3
MAX_WORKERS = 4
STEP_KINDS = ['fx', 'duck', 'loudness', 'insert', 'generate', 'edit']
EDIT_ACTIONS = ('set_gain', 'set_mute', 'set_solo', 'set_automation', 'clear_automation', 'mute_range')

TOO_MANY = (f'That is a lot of edits at once. I can do up to {MAX_STEPS} edits (and {MAX_GENERATE} new parts) per message; '
            'please split the request into smaller ones.')


class MultiError(ValueError):
    """A user-facing reason a multi plan cannot run; the message names the step."""


def schema(fx_schema, inserts_property, action_schema):
    from .generate_part import GENERATE
    props = {'kind': {'type': 'string', 'enum': STEP_KINDS},
             'label': {'type': 'string'},
             'fx': {'anyOf': [fx_schema, {'type': 'null'}]},
             'inserts': inserts_property,
             'generate': {'anyOf': [GENERATE, {'type': 'null'}]},
             'actions': {'type': 'array', 'items': action_schema}}
    step = {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}
    return {'anyOf': [{'type': 'array', 'items': step}, {'type': 'null'}]}


INSTRUCTIONS = (
    'Several edits in one message: when the user asks for two or more different edits at once (joined by "and", commas, '
    '"then", "also", "plus"), return kind=multi with one entry in steps per distinct edit, in the order the user said them. '
    'Each step has kind fx, duck, loudness, insert, generate or edit and exactly the object that single kind uses '
    '(fx for fx/duck/loudness, inserts for insert, generate for generate, actions for edit), with the other objects null and '
    'actions empty. Do not merge edits on different targets into one step ("reverb on the vocals and phone on the drums" is two '
    'fx steps); the same effect on several layers is one fx step with several targets; several sounds in one insert ask may share '
    'one insert step. label is a 2-6 word summary of the step, e.g. "Reverb on vocals", "Drums like a phone", "Bass down 2 semitones", '
    '"Add a piano part", "Vine boom at 0:12". A generate step adds a new musical part: generate.instrument is the instrument words, '
    'generate.prompt the wording of that clause, role, sections and interval null unless stated, density null, engine "auto". '
    'Edit steps use the kind=edit actions set_gain, set_mute, set_solo, set_automation, clear_automation and mute_range (never '
    'follow_intensity). Inside multi never use fx.adjust or speed/tempo presets (slowed, nightcore, sped up). '
    f'At most {MAX_STEPS} steps and {MAX_GENERATE} generate steps; for more, return clarify asking the user to split the request. '
    'Separation, isolating or removing a sound by splitting, cinematic/director, sound replacement, bleeping and speed changes cannot be '
    'combined with other edits: return clarify asking to do that one on its own first. For kind=multi the top-level fx and inserts '
    'are null and actions empty; for every other kind steps is null. A request to add one new instrument part may be multi with one generate step; any other single edit is never multi.'
)


# ---------------------------------------------------------------- planner validation

def _name(step, index):
    label = step.get('label') if isinstance(step, dict) else None
    label = label.strip()[:80] if isinstance(label, str) and label.strip() else None
    return f'Edit {index + 1}' + (f' ("{label}")' if label else '')


def validate(data, body):
    """Validate a planner multi plan with each kind's own validator. Raises MultiError naming the step."""
    from .agent import validate_operations
    from .fx_schema import FxError, validate_plan as validate_fx
    from .generate_part import validate_generate
    from .sfx_plan import normalize_insert
    steps = data.get('steps')
    if not isinstance(steps, list) or not steps: raise MultiError('I could not split that into separate edits. Please rephrase.')
    if len(steps) > MAX_STEPS or sum(1 for s in steps if isinstance(s, dict) and s.get('kind') == 'generate') > MAX_GENERATE:
        raise MultiError(TOO_MANY)
    supplied = {float(n) for n in re.findall(r'(?<![a-zA-Z])\d+(?:\.\d+)?', body.prompt)}
    out = []
    for i, step in enumerate(steps):
        name = _name(step, i)
        try:
            if not isinstance(step, dict) or step.get('kind') not in STEP_KINDS: raise MultiError('That edit is not supported here.')
            kind = step['kind']
            clean = {'kind': kind, 'label': None, 'fx': None, 'inserts': None, 'generate': None, 'actions': []}
            if kind in ('fx', 'duck', 'loudness'):
                plan = step.get('fx')
                if isinstance(plan, dict) and plan.get('adjust'):
                    raise MultiError('Ask for "more", "less" or removing an effect on its own.')
                if isinstance(plan, dict) and (plan.get('speed') is not None or plan.get('preset') in _tempo_presets()):
                    raise MultiError('Speed changes reshape the whole timeline; ask for that on its own.')
                clean['fx'] = validate_fx(kind, plan, body.stems, body.duration, body.prompt)
            elif kind == 'insert':
                inserts = step.get('inserts')
                if not isinstance(inserts, list) or not 1 <= len(inserts) <= 10: raise MultiError('Which sound should I add, and where?')
                clean['inserts'] = [normalize_insert(item, body.prompt, body.duration) for item in inserts]
            elif kind == 'generate':
                plan = step.get('generate')
                if not isinstance(plan, dict): raise MultiError('Which instrument should I add?')
                clean['generate'] = validate_generate(plan, body.duration)
            else:
                actions = [{k: v for k, v in a.items() if v is not None} for a in step.get('actions') or [] if isinstance(a, dict)]
                if not actions or len(actions) > 20: raise MultiError('What should change on which layer?')
                if any(a.get('type') not in EDIT_ACTIONS for a in actions): raise MultiError('Ask for that layer edit on its own.')
                validate_operations(actions, body)
                times = [t for a in actions if a['type'] == 'mute_range' for t in (a['start'], a['end'])]
                if any(t not in supplied for t in times): raise MultiError('Give the exact times to mute.')
                clean['actions'] = actions
            if kind != 'insert' and step.get('inserts'): raise MultiError('That edit mixes two kinds of change.')
            clean['label'] = label(clean, body.stems, step.get('label'))
            out.append(clean)
        except FxError as exc:
            raise MultiError(f'{name}: {exc}') from exc
        except MultiError as exc:
            if exc.args and exc.args[0] == TOO_MANY: raise
            raise MultiError(f'{name}: {exc}') from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise MultiError(f'{name} could not be understood ({exc}). Please rephrase that part.') from exc
    message = 'Previewing ' + str(len(out)) + ' edits: ' + '; '.join(s['label'] for s in out) + '.'
    return {'kind': 'multi', 'steps': out, 'message': message, 'description': None, 'remove': False, 'actions': [], 'fx': None,
            'inserts': None, 'processing_time': None, 'separation_mode': None, 'stems': None, 'use_visual': False,
            'guidance': {'anchors': []}, 'edit_scope': {'action': 'keep', 'interval': None}}


def _tempo_presets():
    from .fx_schema import TEMPO_PRESETS
    return TEMPO_PRESETS


def finalize(plan, duration, features):
    """Resolve each insert step's sounds against the library (as a single insert ask would); a miss becomes a clarify."""
    from .sfx_plan import clarify, finalize as finalize_insert
    steps = []
    for i, step in enumerate(plan['steps']):
        if step['kind'] == 'insert':
            done = finalize_insert({'kind': 'insert', 'inserts': step['inserts'], 'message': ''}, duration, features)
            if done['kind'] == 'clarify':
                return clarify(f"{_name(step, i)}: {done['message']}")
            step = {**step, 'inserts': done['inserts']}
        steps.append(step)
    return {**plan, 'steps': steps}


def label(step, stems, given=None):
    if isinstance(given, str) and given.strip(): return given.strip()[:60]
    names = {s['id']: s.get('label') or s['id'] for s in stems}
    kind = step['kind']
    if kind == 'fx':
        fx = step['fx']; where = 'the mix' if fx['targets'] == ['mix'] else ', '.join(names.get(t, t) for t in fx['targets'])
        what = (fx['preset'] or '').replace('_', ' ').title() or ', '.join(e['type'] for e in fx['chain'] or []) or 'Effect'
        return f'{what} on {where}'[:60]
    if kind == 'duck': return f"Duck under {names.get(step['fx']['duck_source'], 'the voice')}"[:60]
    if kind == 'loudness': return f"Loudness {step['fx']['loudness_lufs']:g} LUFS"
    if kind == 'insert': return ('Add ' + ', '.join(i['sfx_query'] for i in step['inserts']))[:60]
    if kind == 'generate': return f"Add a {step['generate'].get('instrument') or 'new'} part"[:60]
    return ('Edit ' + ', '.join(dict.fromkeys(names.get(a['stemId'], 'layer') for a in step['actions'])))[:60]


def fx_request(kind, fx):
    """The fx-preview request body for a planner fx object (mirrors frontend fxRequest)."""
    targets = fx.get('targets') or ['mix']
    if kind == 'duck': return {'op': 'duck', 'targets': targets, 'duck': {'source': fx.get('duck_source') or '', 'depth_db': fx.get('duck_depth_db')}}
    if kind == 'loudness': return {'op': 'loudness', 'targets': targets, 'loudness_lufs': fx.get('loudness_lufs')}
    return {'op': 'chain', 'targets': targets, 'preset': fx.get('preset'), 'chain': fx.get('chain'), 'interval': fx.get('interval'), 'speed': fx.get('speed')}


# ---------------------------------------------------------------- rule-based clause splitting (detection)

_VERBS = (r'add|put|insert|drop|throw|stick|slap|play|place|make|turn|pitch|shift|transpose|give|apply|boost|cut|lower|raise|'
          r'mute|unmute|solo|duck|normali[sz]e|bring|set|fade|remove|take|reduce|increase|pan|widen|mix|layer|change|run|'
          r'compress|distort|send|drown|bump|crank|quiet|silence|speed|slow|bleep|censor|isolate|separate|split|replace|swap')
_LEAD = r'(?:(?:please|can you|could you|would you|also|then|just|now|and)\s+)*'
_SPLIT = re.compile(r'\s*(?:[;]|,|\.(?=\s))\s*(?:and\s+then\s+|and\s+also\s+|and\s+|then\s+|also\s+|plus\s+)?(?=' + _LEAD + r'(?:' + _VERBS + r')\b)'
                    r'|\s+(?:and\s+then|and\s+also|and|then|also|plus|after\s+that)\s+(?=' + _LEAD + r'(?:' + _VERBS + r')\b)', re.I)
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”|‘[^’]*’|(?<![\w])\'[^\']*\'(?![\w])')


def split_clauses(prompt):
    """Split a compound ask on clause boundaries that start a new edit verb ("..., make the drums ...", "... and add ...").
    Quoted words ("when he says 'what and add'") never split. Mirrors frontend/src/agent/multiCommand.ts."""
    text = str(prompt or '').strip()
    quotes = []
    def hide(m):
        quotes.append(m.group(0)); return f'\x00{len(quotes) - 1}\x00'
    masked = _QUOTED.sub(hide, text)
    parts = [p.strip(' ,;.') for p in _SPLIT.split(masked)]
    restore = lambda s: re.sub(r'\x00(\d+)\x00', lambda m: quotes[int(m.group(1))], s)
    return [restore(p) for p in parts if p and p.strip(' ,;.')]


def is_compound(prompt):
    return len(split_clauses(prompt)) >= 2


# ---------------------------------------------------------------- execution spec (multi-preview route)

def edit_rows(rows, actions):
    """Apply validated layer edits to proposed rows in place."""
    by_id = {r['id']: r for r in rows}
    for a in actions:
        row = by_id.get(a['stemId'])
        if row is None: raise ValueError('That layer no longer exists.')
        t = a['type']
        if t == 'set_gain': row['volume'] = float(a['value'])
        elif t == 'set_mute': row['muted'] = int(bool(a['muted']))
        elif t == 'set_solo': row['solo'] = int(bool(a['solo']))
        elif t == 'set_automation': row['automation'] = json.dumps([{'time': float(k['t']), 'value': float(k['value'])} for k in a['keyframes']])
        elif t == 'clear_automation': row['automation'] = '[]'
        elif t == 'mute_range':
            ranges = json.loads(row.get('mute_ranges') or '[]') + [[float(a['start']), float(a['end'])]]
            row['mute_ranges'] = json.dumps(sorted(ranges))
        else: raise ValueError('Unsupported layer edit.')


# ---------------------------------------------------------------- the candidate

def _progress(job, done, total):
    if not job: return
    with connect() as db:
        db.execute("UPDATE jobs SET stage=?,progress=? WHERE id=? AND status='processing'",
                   (f'Applying {min(done + 1, total)} of {total} edits…' if done < total else f'Finishing {total} edits…',
                    int(5 + 90 * done / max(1, total)), job))


def build(project, payload, cid):
    from .sfx_insert import Anchors, place, render_layers
    from .generate_part import generate_layer
    pid = project['id']; snapshot = payload['_candidate']; steps = payload['steps']
    source = F._check_snapshot(pid, snapshot)
    info = probe(source)
    rate, channels, duration = info['sample_rate'], info['channels'], info['duration']
    rows = [dict(r) for r in snapshot['rows']]
    if not rows:
        rows = [{'id': uid(), 'project_id': pid, 'name': 'Original audio', 'path': str(source), 'duration': duration,
                 'start_offset': 0, 'volume': 1, 'muted': 0, 'solo': 0, 'automation': '[]', 'mute_ranges': '[]', 'revision': 0}]
    directory = project_dir(pid) / 'stems'; directory.mkdir(parents=True, exist_ok=True)
    created = []; lock = threading.Lock()
    def new_file(suffix='.wav'):
        path = directory / (uid() + suffix)
        with lock: created.append(path)
        return path
    total = len(steps); done = 0; notes: list[str] = []
    timings: list[dict] = []
    _progress(cid, 0, total)
    def fail(indices, exc):
        names = ', '.join(_name(steps[i], i) for i in indices)
        return ValueError(f'{names} failed: {exc}' if names else str(exc))
    try:
        original = new_file()
        if snapshot['rows']: export(snapshot['rows'], source, original, 'wav', False)
        else: shutil.copyfile(source, original)
        by_id = {r['id']: r for r in rows}
        records = {r['id']: F.record(r) or {'version': 1, 'dry_path': r['path'], 'items': [], 'path': None} for r in rows}
        plans: dict[str, list] = {}; owners: dict[str, list[int]] = {}
        for i, step in enumerate(steps):
            if step['kind'] != 'fx': continue
            spec = step['spec']; item = spec['item']; mix = spec['targets'] == ['mix']
            for row in (rows if mix else [by_id[t] for t in spec['targets'] if t in by_id]):
                plans.setdefault(row['id'], list(records[row['id']]['items'])).append({**item, 'scope': 'mix' if mix else 'layer'})
                owners.setdefault(row['id'], []).append(i)
        inserts, insert_steps = [], []
        for i, step in enumerate(steps):
            if step['kind'] == 'insert':
                inserts.extend(step['spec']); insert_steps.append(i)

        # Phase 1: independent work in parallel.
        def timed(key, fn, *args):
            began = time.monotonic()
            try: return fn(*args)
            finally:
                with lock: timings.append({'task': key, 'start': began, 'end': time.monotonic()})
        tasks = {}
        for row_id, items in plans.items():
            tasks[('layer', row_id)] = (owners[row_id], lambda r=by_id[row_id], it=items: timed('layer:' + r['id'], F.render_layer, r, records[r['id']], it, new_file))
        for i, step in enumerate(steps):
            if step['kind'] == 'generate':
                tasks[('generate', i)] = ([i], lambda s=step, i=i: timed(f'generate:{i}', generate_layer, pid, original, snapshot['rows'], s['spec'], 0, new_file))
        if inserts:
            anchors = Anchors(pid, source, snapshot['rows'], duration, payload.get('_features'))
            tasks[('insert', 0)] = (insert_steps, lambda: timed('insert', place, inserts, anchors))
        remaining = {i: sum(1 for owners_, _ in tasks.values() if i in owners_) for i in range(total)}
        results = {}
        if tasks:
            pool = ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks)), thread_name_prefix='multi')
            try:
                futures = {pool.submit(fn): key for key, (_, fn) in tasks.items()}
                pending = set(futures)
                while pending:
                    finished, pending = wait(pending, return_when=FIRST_EXCEPTION)
                    for future in finished:
                        key = futures[future]; indices = tasks[key][0]
                        exc = future.exception()
                        if exc is not None:
                            for other in pending: other.cancel()
                            raise fail(indices, exc) from exc
                        results[key] = future.result()
                        for i in indices:
                            remaining[i] -= 1
                            if remaining[i] == 0 and steps[i]['kind'] in ('fx', 'generate'): done += 1
                        _progress(cid, done, total)
            finally:
                pool.shutdown(wait=True, cancel_futures=True)

        # Phase 2: fold the results in, then the cheap and order-dependent steps in prompt order.
        changed = []
        for row_id in plans:
            row, path, record_, length, _, _, local, rise = results[('layer', row_id)]
            row['path'] = path; row['fx'] = json.dumps(record_) if record_ else None; row['duration'] = length
            notes.extend(local)
            changed.append({'id': row['id'], 'name': row['name'], 'loudness_change_db': round(rise, 1) if rise is not None else None})
        generated = []
        for i, step in enumerate(steps):
            if step['kind'] == 'generate':
                layer, result, _, _ = results[('generate', i)]
                rows.append(layer); generated.append({'step': i, 'layer_id': layer['id'], 'name': layer['name']})
                notes.extend(result.get('warnings') or [])
        placements = []
        if inserts:
            placements, warnings = results[('insert', 0)]
            notes.extend(warnings)
            try:
                _, rows = render_layers(pid, placements, inserts, rows, rate, channels, round(duration * rate), created, notes, 6.0)
            except Exception as exc:
                raise fail(insert_steps, exc) from exc
            done += len(insert_steps); _progress(cid, done, total)
        for i, step in enumerate(steps):
            if step['kind'] not in ('edit', 'duck'): continue
            try:
                if step['kind'] == 'edit': edit_rows(rows, step['spec'])
                else: F._duck(rows, {r['id']: r for r in rows}, step['spec'], {})
            except Exception as exc:
                raise fail([i], exc) from exc
            done += 1; _progress(cid, done, total)
        for i, step in enumerate(steps):
            if step['kind'] != 'loudness': continue
            try:
                interim = new_file(); export(rows, source, interim, 'wav', False)
                F._render_rows(rows, {r['id']: r for r in rows}, step['spec'], {}, interim, new_file, notes)
            except Exception as exc:
                raise fail([i], exc) from exc
            done += 1; _progress(cid, done, total)

        mix = new_file(); export(rows, source, mix, 'wav', False)
        mix_x, _ = F.decode(mix)
        peak = F.true_peak_db(mix_x)
        if peak > F.TRUE_PEAK_CEILING_DB + .05:
            notes.append(f'The combined mix peaks at {peak:.1f} dBTP; lower a layer or normalize loudness before export.')
        if any(r.get('fx') and not F.record(r) for r in rows if r['id'] in plans):
            notes.append('Ducking under the inserted sounds bakes those layers\' new effects into their audio.')
        for path in [p for p in created if p.name.endswith('-sfx-src.wav')]: path.unlink(missing_ok=True)
        labels = [{'index': i, 'kind': s['kind'], 'label': s['label']} for i, s in enumerate(steps)]
        return {'target': {'path': str(mix)}, 'mix': {'path': str(mix)}, 'original': {'path': str(original)}, 'snapshot': snapshot,
                'proposed': rows,
                'inference': {'model': {'backend': 'multi', 'is_mock': False},
                              'warnings': list(dict.fromkeys(notes + ['All edits apply together; Undo restores everything at once.']))},
                'provenance': {'kind': 'multi', 'request': payload.get('request'), 'edits': labels, 'raw_steps': payload.get('raw_steps'),
                               'targets': changed, 'generated': generated,
                               'placements': [{k: p[k] for k in ('id', 'sfx_id', 'title', 'anchor_time', 'start', 'end', 'gain_db', 'duck')} for p in placements],
                               'concurrency': {'workers': min(MAX_WORKERS, len(tasks)) if tasks else 0, 'tasks': len(tasks),
                                               'timings': [{**t, 'start': round(t['start'], 4), 'end': round(t['end'], 4)} for t in timings]},
                               'stem_revisions': revisions(snapshot['rows']), 'source_sha256': snapshot['source_sha256']}}
    except Exception:
        for path in created: path.unlink(missing_ok=True)
        raise
