"""Pydantic models for the audio inference API.

These are Person 1's *proposed* audio-specific schemas (Contracts B, C, D). They are
designed to be a strict superset of what Person 4's backend already sends and reads
(``metadata = {"prompts": [...], "visual_prompt": {...}}`` in, ``stems[].name`` and
``stems[].audio_base64`` out), so adopting the extra fields is optional for every team.
Status: awaiting team agreement; see docs/contracts/.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import CONTRACT_VERSION

AnchorToken = Literal["+", "-"]


class TimeRange(BaseModel):
    """Seconds on the *source media timeline* (0 = first sample of the uploaded media)."""

    model_config = ConfigDict(extra="ignore")
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self) -> "TimeRange":
        if self.end <= self.start:
            raise ValueError("time range end must be greater than start")
        return self


class Anchor(BaseModel):
    """Span prompt. ``+`` marks a span where the target sound is present, ``-`` where it is absent."""

    model_config = ConfigDict(extra="ignore")
    token: AnchorToken = "+"
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self) -> "Anchor":
        if self.end <= self.start:
            raise ValueError("anchor end must be greater than start")
        return self


class VisualPrompt(BaseModel):
    """Contract D (proposed): visual guidance from Person 2.

    SAM Audio's visual prompting needs, for every video frame, the frame itself and a
    binary mask of the selected object. Bounding boxes, labels and track ids alone are
    not sufficient model input, so the model-facing requirement is a *mask video*: a
    losslessly encoded video with the same frame count, frame rate and resolution as the
    source video where pixels > 127 belong to the selected object. Frames where the object
    is absent are all zero. Visible intervals do not imply audible sound; supply explicit anchors separately.
    Unknown extra fields (bboxes, scores, ...) are accepted and ignored.
    """

    model_config = ConfigDict(extra="allow")
    object_id: str | None = None
    label: str | None = None
    asset_id: str | None = None
    video_ref: str | None = Field(default=None, description="file:// or http(s) reference to the source video")
    mask_video_ref: str | None = Field(default=None, description="file:// or http(s) reference to the per-frame mask video")
    time_range: TimeRange | None = None
    time_ranges: list[TimeRange] = Field(default_factory=list)
    coordinate_system: Literal["pixel", "normalized"] = "pixel"
    frame_rate: float | None = Field(default=None, gt=0)

    def all_time_ranges(self) -> list[TimeRange]:
        return ([self.time_range] if self.time_range else []) + list(self.time_ranges)

    def has_model_input(self) -> bool:
        return bool(self.video_ref and self.mask_video_ref)


class InferenceOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model_id: str | None = None
    predict_spans: bool | None = None
    reranking_candidates: int | None = Field(default=None, ge=1, le=16)
    residual: Literal["mixture_minus_targets", "model", "none"] = "mixture_minus_targets"
    residual_name: str = Field(default="residual", min_length=1, max_length=200)
    stereo_mode: Literal["duplicate", "per_channel"] | None = None
    chunk_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    chunk_batch: int | None = Field(default=None, ge=1, le=8)
    seed: int | None = Field(default=None, ge=0)
    output_format: Literal["pcm_f32le", "pcm_s16le"] = "pcm_f32le"
    # "base64" inlines WAV bytes (Person 4's current contract). "file" writes stems under
    # AUDIO_OUTPUT_ROOT and returns file:// references instead.
    return_mode: Literal["base64", "file"] = "base64"


class PromptSpec(BaseModel):
    """One separation target. ``prompts`` entries may be plain strings or this object."""

    model_config = ConfigDict(extra="ignore")
    text: str = Field(default="", max_length=500)
    name: str | None = Field(default=None, max_length=500, description="Stem name shown to the user; defaults to the caller's wording of ``text``")
    anchors: list[Anchor] = Field(default_factory=list)
    visual_prompt: VisualPrompt | None = None

    @model_validator(mode="before")
    @classmethod
    def default_name_to_callers_wording(cls, data: Any) -> Any:
        """Unnamed text prompts keep the caller's wording as the stem name.

        Person 3's editor matches stems case-insensitively against the prompt it sent
        (``frontend/src/components/ProjectPanel.tsx``), while ``text`` is normalised for the
        model ("Isolate the dog barking!" -> "dog barking"). So the default stem name is the
        caller's own wording with whitespace collapsed; case and punctuation are preserved.
        Prompts that normalise to nothing (punctuation only) stay unnamed so ``display_name``
        can fall back to the visual label.
        """
        if isinstance(data, dict) and data.get("name") is None:
            text = data.get("text")
            if isinstance(text, str) and normalise_description(text):
                data = {**data, "name": " ".join(text.split())}
        return data

    @field_validator("text")
    @classmethod
    def normalise_text(cls, value: str) -> str:
        return normalise_description(value)

    # A prompt may be text-less when visual guidance is supplied (per prompt or shared through
    # ``metadata.visual_prompt``). Whether that guidance is actually usable is only known after
    # translation, so the engine performs that check and reports why it failed.

    @property
    def display_name(self) -> str:
        return self.name or self.text or (self.visual_prompt.label if self.visual_prompt and self.visual_prompt.label else "visual target")


def normalise_description(value: str) -> str:
    """SAM Audio was trained on lowercase noun/verb phrases ("thunder", "man speaking")."""
    text = " ".join(value.strip().split()).lower()
    while text and text[-1] in ".!?,;:":
        text = text[:-1]
    for prefix in ("isolate the ", "isolate ", "extract the ", "extract ", "the "):
        if text.startswith(prefix) and len(text) > len(prefix):
            text = text[len(prefix):]
            break
    return text


class SourceRef(BaseModel):
    """Where the ORIGINAL uploaded media lives (Contract A subset, proposed).

    Person 4 sends only extracted audio bytes today. Visual prompting additionally needs
    the original video, so this optional block lets the caller point at it: a ``file://``
    path under ``AUDIO_MEDIA_ROOT`` (shared volume) or an allow-listed http(s) URL. ``sha256``
    lets the service confirm Person 2's masks were computed on the same bytes.
    """

    model_config = ConfigDict(extra="ignore")
    asset_id: str | None = None
    original_ref: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    duration: float | None = Field(default=None, ge=0)
    frame_rate: float | None = Field(default=None, gt=0)


class SeparationMetadata(BaseModel):
    """Contract B (proposed): what arrives in the ``metadata`` form field / JSON body."""

    model_config = ConfigDict(extra="allow")
    prompts: list[str | PromptSpec] = Field(default_factory=list)
    source: SourceRef | None = None
    # Legacy/shared visual prompt applied to every prompt (Person 4 forwards the project's
    # visual metadata here unchanged). Accepted as free-form JSON and translated leniently.
    visual_prompt: dict[str, Any] | None = None
    anchors: list[Anchor] = Field(default_factory=list, description="Span anchors applied to every prompt")
    options: InferenceOptions = Field(default_factory=InferenceOptions)
    asset_id: str | None = None
    request_id: str | None = Field(default=None, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")

    def resolved_prompts(self, max_prompts: int) -> list[PromptSpec]:
        specs: list[PromptSpec] = []
        for item in self.prompts:
            # Both prompt forms name their stem identically: ``PromptSpec`` defaults ``name`` to the
            # caller's wording, so the string form needs nothing beyond the text.
            spec = PromptSpec(text=item) if isinstance(item, str) else item
            if self.anchors and not spec.anchors:
                spec = spec.model_copy(update={"anchors": list(self.anchors)})
            specs.append(spec)
        if len(specs) > max_prompts:
            raise ValueError(f"at most {max_prompts} prompts per request")
        return specs


class SeparationRequest(SeparationMetadata):
    """JSON body for ``POST /v1/separate`` (asset-reference variant of the multipart endpoint)."""

    audio_ref: str = Field(min_length=1, description="file:// path under AUDIO_MEDIA_ROOT or allow-listed http(s) URL")


class SourceInfo(BaseModel):
    asset_id: str | None = None
    duration: float
    sample_rate: int
    channels: int
    sample_count: int


class StemResult(BaseModel):
    """Contract C (proposed): one stem. ``name`` + ``audio_base64`` are what Person 4 reads today."""

    name: str
    audio_base64: str | None = None
    audio_ref: str | None = None
    stem_id: str
    role: Literal["target", "residual"]
    prompt: str | None = None
    label: str | None = None
    start_time: float = 0.0
    duration: float
    sample_rate: int
    channels: int
    sample_count: int
    format: str = "wav/pcm_f32le"
    spans: list[list[float]] | None = Field(default=None, description="[[start, end], ...] seconds where the target is active, if known")
    source_asset_id: str | None = None


class ModelInfo(BaseModel):
    backend: str
    model_id: str | None
    device: str | None
    dtype: str | None = None
    predict_spans: bool
    reranking_candidates: int
    stereo_mode: str
    chunked: bool
    elapsed_seconds: float
    chunk_seconds: float = 0
    chunk_overlap_seconds: float = 2
    requested_chunk_batch: int = 1
    effective_chunk_batch: int = 1
    peak_memory_bytes: int | None = None
    is_mock: bool = False


class SeparationResponse(BaseModel):
    stems: list[StemResult]
    model: ModelInfo
    source: SourceInfo
    request_id: str
    warnings: list[str] = Field(default_factory=list)
    contract_version: str = CONTRACT_VERSION
