"""Public guidance and edit contracts. Guidance never changes edit scope."""
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


def options(profile, candidates=1):
    return {'reranking_candidates': 2 if profile == 'quality' else candidates,
            'predict_spans': False, 'chunk_seconds': 10, 'chunk_batch': 2,
            'residual': 'mixture_minus_targets', 'output_format': 'pcm_f32le'}


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
