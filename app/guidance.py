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
    audio-service/docs/quality_tuning.md). ``quality`` asks the service for 3 noise draws per
    chunk window (consensus/majority pick, average when the draws disagree, no re-ranker
    needed; 2 draws cannot out-vote a wrong first draw) and rebuilds the
    stem through a time-frequency mask with thresholds on both sides (background bins the
    model only half-attenuated are zeroed; bins where the target dominates are moved fully out
    of the residual); ``preview`` applies the plain mask. Measured 2026-11-13 (sam-audio-large,
    A40, real speech, mean of 4 fixtures): mask alone +10 dB SI-SDRi vs raw; 3 draws + both
    thresholds +17 dB SI-SDRi, 5.9 dB less background in the stem and 6.4 dB less of the removed
    sound in the residual, at about 2.6x the GPU time. Override with env vars."""
    if profile == 'quality':
        extras = {'candidates': _env_number('QUALITY_CANDIDATES', 3, int),
                  'candidate_select': os.getenv('QUALITY_CANDIDATE_SELECT', 'auto'),
                  'refine': os.getenv('QUALITY_REFINE', 'mask')}
        p, threshold, residual = (_env_number('QUALITY_REFINE_P', 2.0), _env_number('QUALITY_REFINE_THRESHOLD', 0.5),
                                  _env_number('QUALITY_REFINE_RESIDUAL_THRESHOLD', 0.5))
    else:
        extras = {'refine': os.getenv('PREVIEW_REFINE', 'mask')}
        p, threshold, residual = (_env_number('PREVIEW_REFINE_P', 2.0), _env_number('PREVIEW_REFINE_THRESHOLD', 0.0),
                                  _env_number('PREVIEW_REFINE_RESIDUAL_THRESHOLD', 0.0))
    if extras['refine'] == 'mask':
        extras.update({'refine_p': p, 'refine_threshold': threshold, 'refine_residual_threshold': residual})
    else:
        extras.pop('refine', None)
    return extras


def options(profile, candidates=1, backend=None):
    """Request options for the audio service. ``quality`` re-ranks inside the model only when
    QUALITY_RERANKING=true (needs the service's text ranker); otherwise it relies on
    :func:`profile_extras`, which works without any ranker."""
    backend = backend or os.getenv('AUDIO_BACKEND', 'audiosep')
    # Always float32, music_stems splits too: 16-bit stems are hard-clipped at full scale by the service and would
    # stop summing to the mix (see media.STEM_OUTPUT_FORMAT).
    if backend == 'audiosep':
        return {'reranking_candidates':1, 'candidates':1, 'predict_spans':False,
                'chunk_seconds':10, 'chunk_batch':1, 'residual':'mixture_minus_targets',
                'output_format':'pcm_f32le', 'refine':'off'}
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
