"""Request/response models for /v1/words, /v1/enhance and /v1/beats (Contract F, proposed).

All times are seconds on the *uploaded media's* timeline (0 = its first sample), rounded to
milliseconds. Requests are multipart ``audio`` + ``metadata`` (a JSON string, may be omitted),
exactly like ``/separate``. Unknown metadata fields are ignored (forward compatible).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Meta(BaseModel):
    model_config = ConfigDict(extra="ignore")
    request_id: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    asset_id: str | None = None


class WordsMetadata(_Meta):
    # ISO 639-1 code ("en", "es"); None = detect.
    language: str | None = Field(default=None, min_length=2, max_length=3, pattern=r"^[a-z]{2,3}$")
    # Bias Whisper to write hesitations ("um", "uh") so they can be cut. English only.
    transcribe_fillers: bool = True
    # Re-time words with CTC forced alignment (English). None = service default (AUDIO_WORDS_ALIGN).
    align: bool | None = None
    # Words added to the profanity list for this request (same syntax as the list file).
    extra_profanity: list[str] = Field(default_factory=list, max_length=200)
    min_pause_seconds: float | None = Field(default=None, ge=0.1, le=10)

    @field_validator("language", mode="before")
    @classmethod
    def _lower(cls, v: Any) -> Any:
        return v.strip().lower() or None if isinstance(v, str) else v


class Word(BaseModel):
    word: str  # as transcribed, with punctuation ("fuck,")
    norm: str  # lower-case, punctuation stripped ("fuck"): match "when he says 'what'" on this
    start: float
    end: float
    confidence: float  # Whisper word probability, 0-1
    segment: int


class Segment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    word_start: int  # words[word_start:word_end] belong to this segment
    word_end: int


class Pause(BaseModel):
    start: float
    end: float
    after_word: int


class SpeechModelInfo(BaseModel):
    model_config = ConfigDict(extra="allow")
    backend: str
    model_id: str
    is_mock: bool


class SourceSummary(BaseModel):
    asset_id: str | None = None
    duration: float
    sample_rate: int
    channels: int


class WordsResponse(BaseModel):
    request_id: str
    language: str | None
    language_probability: float | None
    words: list[Word]
    segments: list[Segment]
    fillers: list[int]  # indices into words
    profanity: list[int]  # indices into words
    pauses: list[Pause]
    aligned: bool  # True when word times come from forced alignment
    model: SpeechModelInfo
    source: SourceSummary
    profanity_list: dict[str, Any]
    timing: dict[str, float]
    warnings: list[str] = Field(default_factory=list)


class EnhanceMetadata(_Meta):
    # Max noise attenuation in dB (strength). None = the service default (unlimited unless
    # AUDIO_ENHANCE_ATTEN_LIMIT_DB is set). ~6 gentle, 12-20 moderate, 100 = unlimited.
    atten_lim_db: float | None = Field(default=None, gt=0, le=100)
    output_format: Literal["pcm_f32le", "pcm_s16le"] = "pcm_f32le"
    return_mode: Literal["base64", "file"] = "base64"
    # True: the output has the upload's own sample rate. False: the service's 48 kHz.
    preserve_sample_rate: bool = True


class EnhanceResponse(BaseModel):
    request_id: str
    format: str
    sample_rate: int
    channels: int
    sample_count: int
    duration: float
    audio_base64: str | None = None
    audio_ref: str | None = None
    atten_lim_db: float | None
    levels: dict[str, float | None]  # rms/peak dBFS in and out
    model: SpeechModelInfo
    source: SourceSummary
    timing: dict[str, float]
    warnings: list[str] = Field(default_factory=list)


class BeatsMetadata(_Meta):
    pass


class BeatsResponse(BaseModel):
    request_id: str
    beats: list[float]
    downbeats: list[float]  # empty when the tracker has no downbeat model (librosa)
    tempo_bpm: float | None
    beats_per_bar: int | None
    downbeats_available: bool
    model: SpeechModelInfo
    source: SourceSummary
    timing: dict[str, float]
    warnings: list[str] = Field(default_factory=list)
