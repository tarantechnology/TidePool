"""Effect DSL: bounded effect types, named presets and plan validation.

The planner (and the rule-based parser in the editor) only ever names an effect type, a preset
or numbers for a fixed set of parameter keys. Everything is clamped here before rendering;
the renderer never sees a filter string from outside this module.
"""
from __future__ import annotations

import math
import re
import uuid

# type -> {param: (min, max, default)}. Units: ms for time/predelay/attack/release, s for
# decay/fades, Hz for frequencies and rates, dB for gains/thresholds, 0..1 for amounts.
EFFECTS: dict[str, dict[str, tuple[float, float, float]]] = {
    'reverb': {'size': (0, 1, .5), 'decay': (.1, 8, 1.5), 'predelay': (0, 200, 10), 'mix': (0, 1, .3), 'damping': (0, 1, .4)},
    'delay': {'time': (20, 2000, 350), 'feedback': (0, .9, .35), 'mix': (0, 1, .3), 'pingpong': (0, 1, 0)},
    'eq': {'frequency': (20, 20000, 1000), 'gain': (-18, 18, 0), 'q': (.1, 10, 1)},
    'low_shelf': {'frequency': (20, 1000, 120), 'gain': (-18, 18, 0)},
    'high_shelf': {'frequency': (1000, 16000, 6000), 'gain': (-18, 18, 0)},
    'highpass': {'frequency': (20, 5000, 80)},
    'lowpass': {'frequency': (100, 20000, 8000)},
    'bandpass': {'low': (20, 5000, 300), 'high': (200, 20000, 3400)},
    'compressor': {'threshold': (-60, 0, -18), 'ratio': (1, 20, 3), 'attack': (.1, 200, 15), 'release': (5, 2000, 200), 'makeup': (0, 24, 0)},
    'limiter': {'ceiling': (-12, -1, -1)},
    'saturation': {'drive': (0, 1, .3), 'mix': (0, 1, 1)},
    'distortion': {'drive': (0, 1, .5), 'tone': (0, 1, .5), 'mix': (0, 1, 1)},
    'bitcrush': {'bits': (2, 16, 8), 'downsample': (1, 32, 1), 'mix': (0, 1, 1)},
    'chorus': {'rate': (.05, 5, .8), 'depth': (0, 1, .5), 'mix': (0, 1, .5)},
    'flanger': {'rate': (.05, 5, .3), 'depth': (0, 1, .5), 'feedback': (0, .9, .4)},
    'phaser': {'rate': (.1, 2, .5), 'depth': (0, 1, .5), 'feedback': (0, .9, .4)},
    'tremolo': {'rate': (.1, 20, 5), 'depth': (0, 1, .5)},
    'autopan': {'rate': (.05, 10, .5), 'depth': (0, 1, .7)},
    'width': {'amount': (0, 2, 1.4)},
    'pitch': {'semitones': (-12, 12, 0), 'formant': (0, 1, 0)},
    'gain': {'gain': (-24, 12, 0)},
    'fade': {'fade_in': (0, 30, 0), 'fade_out': (0, 30, 0)},
    'noise': {'amount': (0, 1, .2)},
    'wobble': {'rate': (.1, 10, .5), 'depth': (0, 1, .2)},
    'ring_mod': {'frequency': (10, 2000, 50), 'mix': (0, 1, .5)},
}
EFFECT_TYPES = list(EFFECTS)
PARAM_KEYS = sorted({k for spec in EFFECTS.values() for k in spec})
EFFECT_LABELS = {'reverb': 'Reverb', 'delay': 'Echo', 'eq': 'EQ', 'low_shelf': 'Bass', 'high_shelf': 'Treble',
                 'highpass': 'Low cut', 'lowpass': 'High cut', 'bandpass': 'Band-pass', 'compressor': 'Compressor',
                 'limiter': 'Limiter', 'saturation': 'Saturation', 'distortion': 'Distortion', 'bitcrush': 'Bitcrush',
                 'chorus': 'Chorus', 'flanger': 'Flanger', 'phaser': 'Phaser', 'tremolo': 'Tremolo', 'autopan': 'Auto-pan',
                 'width': 'Stereo width', 'pitch': 'Pitch', 'gain': 'Gain', 'fade': 'Fade', 'noise': 'Vinyl noise',
                 'wobble': 'Wow & flutter', 'ring_mod': 'Ring mod'}
SPEED_RANGE = (.5, 2.0)
MAX_CHAIN = 12
LOUDNESS_RANGE = (-30.0, -6.0)
DUCK = {'depth_db': (3, 24, 9), 'threshold_db': (-70, -10, None), 'attack_ms': (10, 1000, 120), 'release_ms': (50, 3000, 500)}


def fx(type_, **params):
    return {'type': type_, 'params': params}


_REVERB = lambda size, decay, predelay, mix, damping: fx('reverb', size=size, decay=decay, predelay=predelay, mix=mix, damping=damping)
# name -> label, chain, optional whole-song speed (tempo and pitch together, like a turntable),
# optional loudness cap (dB over the input) for presets that are meant to sound "too loud".
PRESETS: dict[str, dict] = {
    'reverb': {'label': 'Reverb', 'chain': [_REVERB(.5, 1.6, 15, .28, .45)]},
    'church': {'label': 'Church', 'chain': [_REVERB(.9, 4.5, 40, .38, .35)]},
    'stadium': {'label': 'Stadium', 'chain': [_REVERB(1, 3.2, 90, .32, .5), fx('delay', time=280, feedback=.2, mix=.12)]},
    'hall': {'label': 'Concert hall', 'chain': [_REVERB(.75, 2.4, 25, .3, .45)]},
    'bathroom': {'label': 'Bathroom', 'chain': [_REVERB(.2, .9, 3, .35, .1)]},
    'small_room': {'label': 'Small room', 'chain': [_REVERB(.3, .5, 5, .22, .5)]},
    'plate': {'label': 'Plate reverb', 'chain': [_REVERB(.55, 1.8, 10, .3, .2)]},
    'echo': {'label': 'Echo', 'chain': [fx('delay', time=380, feedback=.4, mix=.3, pingpong=1)]},
    'telephone': {'match_level': True, 'label': 'Telephone', 'chain': [fx('bandpass', low=300, high=3400), fx('saturation', drive=.35), fx('compressor', threshold=-24, ratio=4, attack=5, release=120, makeup=4), fx('width', amount=0)]},
    'radio': {'match_level': True, 'label': 'Radio', 'chain': [fx('bandpass', low=200, high=5000), fx('eq', frequency=1500, gain=3, q=.8), fx('saturation', drive=.25), fx('compressor', threshold=-20, ratio=4, attack=5, release=150, makeup=3)]},
    'walkie_talkie': {'match_level': True, 'label': 'Walkie-talkie', 'chain': [fx('bandpass', low=400, high=2800), fx('distortion', drive=.45, tone=.6), fx('bitcrush', bits=10, downsample=2), fx('noise', amount=.12), fx('width', amount=0)]},
    'through_the_wall': {'label': 'Through the wall', 'chain': [fx('lowpass', frequency=450), fx('low_shelf', frequency=120, gain=3), _REVERB(.2, .4, 2, .15, .8), fx('gain', gain=-6)]},
    'underwater': {'label': 'Underwater', 'chain': [fx('lowpass', frequency=600), fx('wobble', rate=.6, depth=.3), _REVERB(.4, 1.2, 10, .3, .9)]},
    'megaphone': {'match_level': True, 'label': 'Megaphone', 'chain': [fx('bandpass', low=500, high=4000), fx('eq', frequency=2000, gain=6, q=1), fx('distortion', drive=.55, tone=.7), fx('compressor', threshold=-20, ratio=6, attack=3, release=100, makeup=2), fx('width', amount=0)]},
    'lofi': {'label': 'Lo-fi', 'chain': [fx('lowpass', frequency=4500), fx('bitcrush', bits=12, downsample=2), fx('wobble', rate=.5, depth=.1), fx('noise', amount=.2), fx('saturation', drive=.2)]},
    'vinyl': {'label': 'Vinyl', 'chain': [fx('highpass', frequency=40), fx('lowpass', frequency=9000), fx('wobble', rate=.55, depth=.05), fx('noise', amount=.35), fx('saturation', drive=.1)]},
    'eight_bit': {'label': '8-bit', 'chain': [fx('bitcrush', bits=6, downsample=6), fx('highpass', frequency=120)]},
    'old_movie': {'match_level': True, 'label': 'Old movie', 'chain': [fx('bandpass', low=300, high=3500), fx('wobble', rate=.7, depth=.08), fx('noise', amount=.45), fx('saturation', drive=.2), fx('width', amount=0)]},
    'chipmunk': {'label': 'Chipmunk', 'chain': [fx('pitch', semitones=8, formant=0)]},
    'deep_voice': {'label': 'Deep voice', 'chain': [fx('pitch', semitones=-5, formant=0), fx('low_shelf', frequency=150, gain=2)]},
    'robot': {'match_level': True, 'label': 'Robot', 'chain': [fx('ring_mod', frequency=60, mix=.6), fx('flanger', rate=.1, depth=.3, feedback=.6), fx('compressor', threshold=-20, ratio=4, attack=5, release=100, makeup=2)]},
    'announcer': {'label': 'Announcer', 'chain': [fx('compressor', threshold=-24, ratio=4, attack=10, release=200, makeup=4), fx('low_shelf', frequency=120, gain=3), fx('eq', frequency=3000, gain=2, q=.8), _REVERB(.6, 1.4, 20, .12, .5)]},
    'giant': {'label': 'Giant', 'chain': [fx('pitch', semitones=-9, formant=0), fx('low_shelf', frequency=120, gain=4), _REVERB(.9, 3, 30, .3, .5)]},
    'dreamy': {'label': 'Dreamy', 'chain': [fx('chorus', rate=.4, depth=.5, mix=.4), _REVERB(.85, 5, 30, .5, .6), fx('lowpass', frequency=8000)]},
    'wide': {'label': 'Wider', 'chain': [fx('width', amount=1.6)]},
    'mono': {'label': 'Mono', 'chain': [fx('width', amount=0)]},
    'warm': {'label': 'Warmer', 'chain': [fx('low_shelf', frequency=200, gain=3), fx('high_shelf', frequency=6000, gain=-2.5), fx('saturation', drive=.15)]},
    'bright': {'label': 'Brighter', 'chain': [fx('highpass', frequency=30), fx('high_shelf', frequency=5000, gain=4)]},
    'less_harsh': {'label': 'Less harsh', 'chain': [fx('eq', frequency=3500, gain=-4, q=1.2), fx('eq', frequency=7000, gain=-2, q=1)]},
    'less_muddy': {'label': 'Less muddy', 'chain': [fx('highpass', frequency=60), fx('eq', frequency=300, gain=-4, q=1)]},
    'bass_boost': {'label': 'Bass boost', 'chain': [fx('low_shelf', frequency=100, gain=6)]},
    'overdrive': {'label': 'Overdrive', 'chain': [fx('distortion', drive=.6, tone=.55), fx('eq', frequency=800, gain=2, q=.7)]},
    'chorus': {'label': 'Chorus', 'chain': [fx('chorus', rate=.8, depth=.5, mix=.5)]},
    'flanger': {'label': 'Flanger', 'chain': [fx('flanger', rate=.3, depth=.5, feedback=.4)]},
    'phaser': {'label': 'Phaser', 'chain': [fx('phaser', rate=.5, depth=.5, feedback=.4)]},
    'tremolo': {'label': 'Tremolo', 'chain': [fx('tremolo', rate=5, depth=.5)]},
    'autopan': {'label': 'Auto-pan', 'chain': [fx('autopan', rate=.5, depth=.7)]},
    'deep_fried': {'label': 'Deep fried', 'loudness_cap_db': 3, 'chain': [fx('saturation', drive=1), fx('distortion', drive=.8, tone=.8), fx('bitcrush', bits=8, downsample=2), fx('eq', frequency=1000, gain=8, q=.7), fx('high_shelf', frequency=4000, gain=6), fx('gain', gain=6)]},
    'slowed_reverb': {'label': 'Slowed + reverb', 'speed': .85, 'chain': [_REVERB(.8, 3.5, 30, .35, .55)]},
    'slowed': {'label': 'Slowed', 'speed': .85, 'chain': []},
    'nightcore': {'label': 'Nightcore', 'speed': 1.25, 'chain': []},
    'sped_up': {'label': 'Sped up', 'speed': 1.15, 'chain': []},
}
PRESET_NAMES = list(PRESETS)
TEMPO_PRESETS = [name for name, p in PRESETS.items() if p.get('speed')]


class FxError(ValueError):
    """A plan or request that cannot be rendered safely; the message is user-facing."""


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise FxError(f'{name} must be a finite number.')
    return float(value)


def clamp_effect(effect, notes=None):
    """One effect with every parameter of its type filled and clamped; unknown types are rejected.
    Parameters that do not belong to the type are ignored (the strict schema always sends them as null)."""
    if not isinstance(effect, dict) or effect.get('type') not in EFFECTS:
        raise FxError(f"Unknown effect type {effect.get('type') if isinstance(effect, dict) else effect!r}.")
    kind = effect['type']; raw = effect.get('params') or {}
    if not isinstance(raw, dict): raise FxError('Effect parameters must be an object.')
    params = {}
    for key, (low, high, default) in EFFECTS[kind].items():
        value = raw.get(key)
        if value is None: params[key] = float(default); continue
        value = _number(value, f'{kind} {key}')
        clamped = min(high, max(low, value))
        if clamped != value and notes is not None: notes.append(f'{EFFECT_LABELS[kind]} {key} limited to {clamped:g}.')
        params[key] = clamped
    if kind == 'bandpass' and params['low'] >= params['high']:
        if raw.get('low') is not None and raw.get('high') is not None: raise FxError('Band-pass low edge must be below its high edge.')
        params['low'], params['high'] = EFFECTS['bandpass']['low'][2], EFFECTS['bandpass']['high'][2]
    if kind in ('bitcrush',): params['bits'] = round(params['bits']); params['downsample'] = round(params['downsample'])
    if kind == 'pitch': params['formant'] = 1.0 if params['formant'] >= .5 else 0.0
    if kind == 'delay': params['pingpong'] = 1.0 if params['pingpong'] >= .5 else 0.0
    return {'type': kind, 'params': params}


def clamp_chain(chain, notes=None):
    if not isinstance(chain, list): raise FxError('An effect chain must be a list.')
    if len(chain) > MAX_CHAIN: raise FxError(f'At most {MAX_CHAIN} effects per edit.')
    return [clamp_effect(e, notes) for e in chain]


def clamp_speed(value):
    if value is None: return None
    value = _number(value, 'Speed')
    if not SPEED_RANGE[0] <= value <= SPEED_RANGE[1]: raise FxError(f'Speed must be between {SPEED_RANGE[0]}x and {SPEED_RANGE[1]}x.')
    return None if abs(value - 1) < 1e-6 else round(value, 4)


def clamp_interval(interval, duration):
    if interval is None: return None
    start = _number(interval.get('start'), 'Interval start'); end = _number(interval.get('end'), 'Interval end')
    if start < 0 or end <= start + .05 or (duration and end > duration + 1e-6):
        raise FxError('The effect interval must be inside the timeline and at least 50 ms long.')
    return {'start': start, 'end': end}


def describe(chain):
    return ' · '.join(EFFECT_LABELS[e['type']] for e in chain)


def item(preset=None, chain=None, interval=None, speed=None, duration=0, notes=None, label=None):
    """One chip: a preset and/or chain, an optional interval, an optional whole-song speed."""
    if preset is not None and preset not in PRESETS: raise FxError(f'Unknown preset {preset!r}.')
    base = PRESETS[preset] if preset else {}
    effects = clamp_chain([*base.get('chain', []), *(chain or [])], notes)
    if len(effects) > MAX_CHAIN: raise FxError(f'At most {MAX_CHAIN} effects per edit.')
    rate = clamp_speed(speed if speed is not None else base.get('speed'))
    interval = clamp_interval(interval, duration)
    if rate and interval: raise FxError('A speed change applies to the whole song, not a time range.')
    if not effects and not rate: raise FxError('Choose an effect or a preset.')
    name = label or base.get('label') or describe(effects)
    if rate and not preset: name = f'Speed {rate:g}x' + (f' + {describe(effects)}' if effects else '')
    out = {'id': uuid.uuid4().hex[:12], 'label': name[:60], 'preset': preset, 'chain': effects, 'interval': interval, 'speed': rate}
    if base.get('loudness_cap_db') is not None: out['loudness_cap_db'] = base['loudness_cap_db']
    if base.get('match_level'): out['match_level'] = True
    return out


# "More" / "less": intensity-like parameters move by a step and stay clamped.
def _scale(kind, key, value, f):
    low, high, _ = EFFECTS[kind][key]
    if key in ('mix', 'depth', 'drive', 'feedback', 'amount') and not (kind == 'width'):
        v = value * f
    elif kind == 'width' and key == 'amount':
        v = 1 + (value - 1) * f
    elif key in ('gain', 'makeup', 'semitones'):
        v = value * f
    elif key == 'decay' or key == 'size':
        v = value * (1 + (f - 1) * .6)
    elif key == 'ratio':
        v = 1 + (value - 1) * f
    elif key == 'bits':
        v = value - 2 if f > 1 else value + 2
    elif key == 'downsample':
        v = value * f
    else:
        return value
    v = min(high, max(low, v))
    return round(v) if key in ('bits', 'downsample') else round(v, 4)


def adjust_item(it, direction):
    if direction not in ('more', 'less'): raise FxError('Adjust must be "more" or "less".')
    f = 1.5 if direction == 'more' else 1 / 1.5
    chain = [{'type': e['type'], 'params': {k: _scale(e['type'], k, v, f) for k, v in e['params'].items()}} for e in it['chain']]
    speed = it.get('speed')
    if speed: speed = round(min(SPEED_RANGE[1], max(SPEED_RANGE[0], 1 + (speed - 1) * f)), 4)
    if chain == it['chain'] and speed == it.get('speed'): raise FxError(f'"{it["label"]}" is already at its {"strongest" if direction == "more" else "subtlest"} setting.')
    level = it.get('level', 0) + (1 if direction == 'more' else -1)
    label = re.sub(r' \((?:more|less)(?: x\d+)?\)$', '', it['label'])
    if level: label += f" ({'more' if level > 0 else 'less'}{'' if abs(level) == 1 else f' x{abs(level)}'})"
    return {**it, 'chain': chain, 'speed': speed, 'level': level, 'label': label[:60]}


def duck_params(raw):
    raw = raw or {}
    out = {}
    for key, (low, high, default) in DUCK.items():
        value = raw.get(key)
        if value is None: out[key] = default; continue
        out[key] = min(high, max(low, _number(value, key)))
    return out


def loudness_target(value):
    if value is None: return -14.0
    value = _number(value, 'Loudness target')
    return min(LOUDNESS_RANGE[1], max(LOUDNESS_RANGE[0], value))


# ---- Planner schema (OpenAI strict Structured Outputs: every key required, nulls when irrelevant) ----

def _obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


_NUM = {'type': ['number', 'null']}
EFFECT_SCHEMA = _obj({'type': {'type': 'string', 'enum': EFFECT_TYPES},
                      'params': _obj({k: _NUM for k in PARAM_KEYS})})
FX_SCHEMA = _obj({
    'targets': {'type': 'array', 'items': {'type': 'string'}},
    'preset': {'type': ['string', 'null'], 'enum': [*PRESET_NAMES, None]},
    'chain': {'anyOf': [{'type': 'array', 'items': EFFECT_SCHEMA}, {'type': 'null'}]},
    'interval': {'anyOf': [_obj({'start': {'type': 'number'}, 'end': {'type': 'number'}}), {'type': 'null'}]},
    'adjust': {'type': ['string', 'null'], 'enum': ['more', 'less', 'remove', None]},
    'speed': _NUM,
    'duck_source': {'type': ['string', 'null']},
    'duck_depth_db': _NUM,
    'loudness_lufs': _NUM,
})

FX_INSTRUCTIONS = (
    'Layer effects: use kind=fx to put an audio effect on existing layers or the whole mix. fx.targets lists exact stem ids '
    '(match the user words against label, canonical_label and tags, e.g. "the singer" or "vocals" is the stem labelled or tagged vocals; '
    '"the guitar" may be the stem tagged Guitar) or ["mix"] for the whole song, everything, or when the project has no stems. '
    'Prefer a named preset: ' + ', '.join(PRESET_NAMES) + '. '
    'Presets slowed_reverb, slowed, nightcore and sped_up change the speed and length of the whole song and need targets ["mix"]; never use them on one layer. '
    'Use fx.chain (effect types with params; unused params null, units ms/s/Hz/dB, amounts 0..1) only for requests no preset covers, '
    'for example "echo with 500 ms delay" or "pitch the vocals up 3 semitones"; a chain may extend a preset. Never write filter strings. '
    'fx.interval only for an explicit user-supplied time range; never infer one. '
    'fx.adjust="more" or "less" tweaks the most recent effect ("more", "a bit wetter", "too much", "subtler"), with preset and chain null '
    'unless the user names the effect; adjust="remove" removes the named or most recent effect from the targets ("remove the reverb from the vocals", "take the effects off"). '
    'Use kind=duck for auto-ducking ("duck the music under the voice", "lower the music when he talks"): fx.duck_source is the exact '
    'id of the speech or vocal layer, fx.targets the music layers to lower, fx.duck_depth_db null unless stated. Ducking needs separate '
    'voice and music layers; otherwise clarify. Use kind=loudness to normalize loudness ("normalize for YouTube", "make it loud enough '
    'for TikTok"): targets ["mix"] unless a layer is named, fx.loudness_lufs -14 for YouTube/TikTok/Spotify/Instagram, -16 for podcasts '
    'and Apple, -23 for broadcast, or the stated number. For kind fx, duck and loudness actions must be empty. For every other kind fx is null. '
    'Effects cannot add new sounds; requests for new sound effects (a vine boom, a laugh track) are not effects: clarify.'
)


def validate_plan(kind, plan, stems, duration, prompt):
    """Validate a planner fx object against the project's stems. Returns a normalized copy.
    Raises FxError (user-facing) for anything that cannot be rendered."""
    if not isinstance(plan, dict): raise FxError('The effect plan is missing.')
    ids = {s['id'] for s in stems}
    targets = plan.get('targets') or ['mix']
    if not isinstance(targets, list) or len(targets) > 20 or any(not isinstance(t, str) for t in targets): raise FxError('Invalid effect targets.')
    targets = list(dict.fromkeys(targets))
    if 'mix' in targets: targets = ['mix']
    unknown = [t for t in targets if t != 'mix' and t not in ids]
    if unknown: raise FxError('Choose an existing layer for the effect.')
    supplied = {float(n) for n in re.findall(r'(?<![a-zA-Z])\d+(?:\.\d+)?', prompt or '')}
    out = {'targets': targets, 'preset': None, 'chain': None, 'interval': None, 'adjust': None, 'speed': None,
           'duck_source': None, 'duck_depth_db': None, 'loudness_lufs': None}
    if kind == 'duck':
        source = plan.get('duck_source')
        if source not in ids: raise FxError('Ducking needs a voice or dialogue layer. Split the voice from the music first.')
        targets = [t for t in targets if t not in ('mix', source)] or [s['id'] for s in stems if s['id'] != source]
        if not targets: raise FxError('Ducking needs at least one music layer besides the voice.')
        depth = plan.get('duck_depth_db')
        out.update(targets=targets, duck_source=source, duck_depth_db=duck_params({'depth_db': depth})['depth_db'] if depth is not None else None)
        return out
    if kind == 'loudness':
        value = plan.get('loudness_lufs')
        if value is not None and abs(value) not in supplied and value not in (-14, -16, -23): raise FxError('Unsupplied loudness target.')
        out['loudness_lufs'] = loudness_target(value)
        return out
    adjust = plan.get('adjust')
    if adjust not in (None, 'more', 'less', 'remove'): raise FxError('Invalid adjustment.')
    out['adjust'] = adjust
    preset = plan.get('preset')
    if preset is not None and preset not in PRESETS: raise FxError('Unknown preset.')
    out['preset'] = preset
    if adjust: return out
    chain = plan.get('chain')
    if chain is not None: out['chain'] = clamp_chain(chain)
    out['speed'] = clamp_speed(plan.get('speed'))
    interval = plan.get('interval')
    if interval is not None:
        if any(interval.get(k) not in supplied for k in ('start', 'end')): raise FxError('Unsupplied timestamp.')
        out['interval'] = clamp_interval(interval, duration)
    probe = item(preset, out['chain'], out['interval'], out['speed'], duration)  # Rejects empty or conflicting plans.
    if probe['speed'] and targets != ['mix']: raise FxError('Speed changes apply to the whole song. Ask for it on the whole mix.')
    return out


# Effect words, minus ones that mostly name sounds or song sections ("room", "wall", "hall", "chorus", "monster").
PRESET_WORDS = re.compile(r'\b(?:reverb\w*|echo(?:es|ey|y)?|church|cathedral|stadium|arena|concert hall|bathroom|shower|small room|next room|'
                          r'phone|telephone|radio|walkie[- ]?talkie|through (?:a |the )?walls?|underwater|megaphone|bullhorn|lo-?fi|vinyl|8-?bit|'
                          r'chiptune|bitcrush\w*|old (?:movie|film|radio)|chipmunks?|robot\w*|announcer|giant|dreamy|wider|mono|warm(?:er)?|'
                          r'bright(?:er)?|less harsh|less muddy|distort\w*|overdrive|flanger|phaser|tremolo|auto-?pan|deep[- ]fried|earrape|'
                          r'slowed|nightcore|sped up|deep(?:er)? voice)\b', re.I)
SOUND_NOUN = re.compile(r'\s*(?:roar\w*|ring(?:s|ing)?|ringtones?|barks?|barking|growl\w*|scream\w*|chatter\w*|static|buzz\w*|'
                        r'notifications?|alarms?|noises?|beeps?|beeping|sirens?|footsteps?|crash\w*|explosions?|sfx|sound effects?)\b', re.I)


def mentions_effect(text):
    """True when a "make X sound like Y" / "turn X into Y" ask names an effect preset as Y ("sound like a phone",
    "into a robot"), so it is a layer effect, not a replacement. The X part never counts: "turn the chorus into
    a verse" and "make the phone ringing sound like a doorbell" stay replacements."""
    m = re.search(r'\b(?:sound(?:s|ing)?\s+like|into)\b', text or '', re.I)
    if not m: return False
    tail = text[m.end():]
    return any(not SOUND_NOUN.match(tail, w.end()) for w in PRESET_WORDS.finditer(tail))


def find_chip(rows, targets, item_id=None, preset=None, removing=False):
    """The applied chip an adjust/remove refers to: by id; else the latest chip on the targets with that
    preset (or the same main effect, so "more reverb" finds a Church chip); else the latest chip."""
    from .fx import record
    pool = [it for r in rows if targets == ['mix'] or r['id'] in targets for it in (record(r) or {}).get('items', [])]
    if item_id: return next((it for r in rows for it in (record(r) or {}).get('items', []) if it['id'] == item_id), None)
    if preset:
        main = (PRESETS[preset]['chain'] or [{}])[0].get('type')
        return (next((it for it in reversed(pool) if it.get('preset') == preset), None)
                or next((it for it in reversed(pool) if main and it['chain'] and it['chain'][0]['type'] == main), None))
    return next((it for it in reversed(pool) if removing or it.get('kind') != 'loudness'), None)
