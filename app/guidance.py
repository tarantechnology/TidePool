"""Public guidance and edit contracts. Guidance never changes edit scope."""
import os
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


def description(text):
    text = ' '.join(text.lower().strip().rstrip('.!?,;:').split())
    for prefix in ('remove the ', 'remove ', 'mute the ', 'mute ', 'isolate the ', 'isolate ', 'extract the ', 'extract ', 'separate the ', 'separate ', 'the '):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text


class Interval(BaseModel):
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode='after')
    def ordered(self):
        if self.end <= self.start: raise ValueError('End must be after start')
        return self


class Anchor(Interval):
    token: Literal['+', '-'] = '+'


class Guidance(BaseModel):
    anchors: list[Anchor] = Field(default_factory=list, max_length=32)

    @model_validator(mode='after')
    def consistent(self):
        if any(a.token!=b.token and max(a.start,b.start)<min(a.end,b.end) for a in self.anchors for b in self.anchors):
            raise ValueError('Sound-present and sound-absent anchors cannot overlap')
        return self


class SoundPrompt(BaseModel):
    text: str = Field(min_length=1, max_length=200)
    name: str | None = Field(default=None, max_length=200)
    anchors: list[Anchor] = Field(default_factory=list, max_length=32)

    @field_validator('text')
    @classmethod
    def normalize(cls, value):
        value = description(value)
        if not value: raise ValueError('Describe a sound')
        return value


class EditScope(BaseModel):
    action: Literal['keep', 'mute', 'lower'] = 'keep'
    interval: Interval | None = None

    @model_validator(mode='after')
    def supported(self):
        if self.interval and self.action != 'mute': raise ValueError('Only mute supports a timed edit')
        return self


def validate_timing(prompts, guidance, edit_scope, duration):
    intervals = list(guidance.anchors)
    for prompt in prompts:
        if isinstance(prompt, SoundPrompt): intervals.extend(prompt.anchors)
    if edit_scope.interval: intervals.append(edit_scope.interval)
    if any(i.end > duration + 1e-6 for i in intervals):
        raise ValueError('Guidance and edit intervals must be inside the audio timeline')


def _env_number(name, default, cast=float):
    value = os.getenv(name, '').strip()
    try: return cast(value) if value else default
    except ValueError: return default


def profile_extras(profile):
    """Per-profile separation options tuned on the audio service's benchmark sweep (see
    audio-service/docs/quality_tuning.md). ``quality`` asks the service for K noise draws per
    chunk window chosen for agreement across windows (no re-ranker needed) and rebuilds the
    stem through a time-frequency mask; ``preview`` keeps the model output unless
    PREVIEW_REFINE is set. All values can be overridden with environment variables."""
    if profile == 'quality':
        extras = {'candidates': _env_number('QUALITY_CANDIDATES', 2, int),
                  'candidate_select': os.getenv('QUALITY_CANDIDATE_SELECT', 'overlap'),
                  'refine': os.getenv('QUALITY_REFINE', 'mask')}
        p, threshold, gate = (_env_number('QUALITY_REFINE_P', 2.0), _env_number('QUALITY_REFINE_THRESHOLD', 0.0),
                              _env_number('QUALITY_REFINE_GATE_DB', 0.0))
    else:
        extras = {'refine': os.getenv('PREVIEW_REFINE', 'off')}
        p, threshold, gate = (_env_number('PREVIEW_REFINE_P', 2.0), _env_number('PREVIEW_REFINE_THRESHOLD', 0.0),
                              _env_number('PREVIEW_REFINE_GATE_DB', 0.0))
    if extras['refine'] == 'mask':
        extras.update({'refine_p': p, 'refine_threshold': threshold, 'refine_gate_db': gate})
    else:
        extras.pop('refine', None)
    return extras


def options(profile, candidates=1):
    """Request options for the audio service. ``quality`` re-ranks inside the model only when
    QUALITY_RERANKING=true (needs the service's text ranker); otherwise it relies on
    :func:`profile_extras`, which works without any ranker."""
    rerank = profile == 'quality' and os.getenv('QUALITY_RERANKING', '').lower() in ('1', 'true', 'yes')
    return {'reranking_candidates': 2 if rerank else candidates,
            'predict_spans': False, 'chunk_seconds': 10, 'chunk_batch': 2,
            'residual': 'mixture_minus_targets', 'output_format': 'pcm_f32le', **profile_extras(profile)}


def edit_target(row, scope):
    """Preserve automation and existing ranges; edit only the separated target."""
    row = dict(row)
    if scope['action'] == 'mute':
        if scope.get('interval'):
            import json
            r = scope['interval']
            row['mute_ranges'] = json.dumps(json.loads(row['mute_ranges']) + [[r['start'], r['end']]])
        else: row['muted'] = 1
    elif scope['action'] == 'lower': row['volume'] *= .5
    return row
