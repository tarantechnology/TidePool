"""Translate visual guidance (Person 2's documents) into model-ready inputs.

Person 2 owns detection, segmentation, tracking and selection. This module only maps
what they produce onto what SAM Audio can actually consume:

* a time interval            -> ``+`` span anchors (works with text prompts alone)
* per-frame instance masks + the original video -> a masked-video visual prompt

Accepted ``visual_prompt`` shapes (all lenient; unknown fields are ignored with a warning):

1. Person 2 *selection* 1.0 (``kind: "visual_selection"``): ``interval``, ``track_id`` /
   ``detection_id``, ``source_sha256``, one mask for the clicked frame. Alone it yields
   anchors only, because masks for every frame live in the detections document.
2. Person 2 *detections* 1.0 (``frames[].objects[].mask`` RLE). Needs a selected object:
   a ``selection`` alongside it (envelope form) or ``selected_track_id`` /
   ``selected_detection_id`` / ``object_id``; a single masked track is picked automatically.
3. Envelope: ``{"selection": {...}, "detections": {...} | "detections_ref": "...",
   "video_ref": "...", "time_ranges": [...]}``.
4. Generic: ``{"video_ref", "mask_video_ref", "time_range" | "time_ranges", "label", ...}``.

The original video is taken from the envelope's ``video_ref`` / ``original_ref`` or from
``metadata.source.original_ref`` (Person 4). Without it, or without per-frame masks, the
request still runs with text + anchors and the response explains what was missing.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .backends.base import Anchor, VisualInput
from .config import Settings
from .contracts import SourceRef, TimeRange, VisualPrompt
from .errors import AudioServiceError, InvalidRequest
from .media import resolve_ref


@dataclass
class VisualTranslation:
    visual: VisualInput | None
    anchors: list[Anchor]
    warnings: list[str] = field(default_factory=list)


def _is_selection(raw: dict[str, Any]) -> bool:
    return raw.get("kind") == "visual_selection" or ("interval" in raw and ("detection_id" in raw or "track_id" in raw))


def _is_detections(raw: dict[str, Any]) -> bool:
    return isinstance(raw.get("frames"), list) and ("analysis_id" in raw or "fps" in raw or "width" in raw)


def _interval(selection: dict[str, Any], source_duration: float) -> tuple[float, float] | None:
    interval = selection.get("interval") or {}
    start = interval.get("start_seconds", selection.get("timestamp_seconds"))
    end = interval.get("end_seconds")
    if start is None:
        return None
    start = max(0.0, float(start))
    end = source_duration if end is None else min(float(end), source_duration)
    return (start, end) if end > start else None


def _load_detections(raw: dict[str, Any], settings: Settings, warnings: list[str], workdir: Path | None = None) -> dict[str, Any] | None:
    detections = raw.get("detections")
    if isinstance(detections, dict):
        return detections
    ref = raw.get("detections_ref")
    if not ref:
        return None
    try:
        path = resolve_ref(str(ref), settings, workdir=workdir)
        data = json.loads(Path(path).read_text())
    except AudioServiceError as exc:
        warnings.append(f"detections_ref ignored: {exc.message}")
        return None
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"detections_ref ignored: {exc}")
        return None
    return data if isinstance(data, dict) else None


def _select_object(detections: dict[str, Any], selection: dict[str, Any] | None, raw: dict[str, Any], warnings: list[str]) -> tuple[int | None, str | None]:
    """Return (track_id, detection_id) identifying the chosen object."""
    track_id = detection_id = None
    if selection:
        track_id, detection_id = selection.get("track_id"), selection.get("detection_id")
    track_id = raw.get("selected_track_id", track_id)
    detection_id = raw.get("selected_detection_id", detection_id)
    object_id = raw.get("object_id") if not selection else None
    if object_id and isinstance(object_id, str) and object_id.startswith("track:") and object_id[6:].isdigit():
        track_id = int(object_id[6:])
    if track_id is None and detection_id is None:
        masked_tracks = {o.get("track_id") for f in detections.get("frames", []) for o in f.get("objects", []) if o.get("mask") and o.get("track_id") is not None}
        if len(masked_tracks) == 1:
            track_id = masked_tracks.pop()
            warnings.append(f"no selection given; using the only masked track {track_id}")
        else:
            warnings.append("detections given without a selected object (selection, selected_track_id or selected_detection_id)")
    return (int(track_id) if track_id is not None else None), detection_id


def _frame_masks(detections: dict[str, Any], track_id: int | None, detection_id: str | None, warnings: list[str]) -> tuple[dict[int, np.ndarray], tuple[float, float] | None]:
    from .visual import decode_rle_mask  # pure numpy, kept in the visual module next to the frame loader

    masks: dict[int, np.ndarray] = {}
    first = last = None
    bbox_only = 0
    for frame in detections.get("frames", []):
        for obj in frame.get("objects", []):
            if (track_id is not None and obj.get("track_id") == track_id) or (track_id is None and obj.get("detection_id") == detection_id):
                if obj.get("mask"):
                    try:
                        masks[int(frame["frame_index"])] = decode_rle_mask(obj["mask"])
                    except (ValueError, KeyError, TypeError) as exc:
                        warnings.append(f"frame {frame.get('frame_index')}: mask ignored ({exc})")
                        continue
                    ts = float(frame.get("timestamp_seconds", 0.0))
                    first = ts if first is None else min(first, ts)
                    until = float(frame.get("valid_until_seconds", ts))
                    last = until if last is None else max(last, until)
                else:
                    bbox_only += 1
    if not masks and bbox_only:
        warnings.append("selected object has bounding boxes but no instance masks; SAM Audio needs per-frame masks (rerun segmentation at native FPS)")
    span = (first, last) if first is not None and last is not None and last > first else None
    return masks, span


def translate_visual_prompt(raw: dict[str, Any] | None, *, source: SourceRef | None, source_duration: float, settings: Settings, backend_supports_visual: bool, workdir: Path | None = None) -> VisualTranslation:
    if not raw:
        return VisualTranslation(None, [])
    warnings: list[str] = []
    anchors: list[Anchor] = []
    selection = detections = None
    video_ref = raw.get("video_ref") or raw.get("original_ref") or (source.original_ref if source else None)
    mask_video_ref = raw.get("mask_video_ref")
    label = raw.get("label")
    object_id = raw.get("object_id")
    frame_rate = raw.get("frame_rate") or raw.get("fps") or (source.frame_rate if source else None)

    if isinstance(raw.get("selection"), dict) or "detections" in raw or "detections_ref" in raw:
        selection = raw.get("selection") if isinstance(raw.get("selection"), dict) else None
        detections = _load_detections(raw, settings, warnings, workdir)
    elif _is_selection(raw):
        selection = raw
    elif _is_detections(raw):
        detections = raw

    for key in ("time_range", "time_ranges"):
        value = raw.get(key)
        for item in (value if isinstance(value, list) else [value] if value else []):
            try:
                tr = TimeRange.model_validate(item)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{key} ignored: {exc}")
                continue
            if tr.start < source_duration:
                anchors.append(("+", tr.start, min(tr.end, source_duration)))

    interval: tuple[float, float] | None = None
    if selection:
        label = label or selection.get("label") or selection.get("class_name")
        object_id = object_id or selection.get("object_id")
        interval = _interval(selection, source_duration)
        if interval:
            anchors.append(("+", *interval))
        else:
            warnings.append("selection has no usable interval on the source timeline")
        if detections is None and not mask_video_ref:
            warnings.append("selection carries one mask for the clicked frame only; per-frame masks (detections document or mask_video_ref) are needed for visual prompting, so text + span anchors are used")

    if not video_ref and not mask_video_ref and detections is None and not selection:
        # Generic / legacy shapes
        try:
            generic = VisualPrompt.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"visual_prompt ignored: {exc}")
            return VisualTranslation(None, anchors, warnings)
        if not generic.has_model_input():
            ignored = sorted(k for k in raw if k not in {"time_range", "time_ranges"})
            if ignored:
                warnings.append("visual_prompt has no video_ref+mask_video_ref pair; fields " + ", ".join(ignored) + " cannot steer SAM Audio and were ignored" + (" (time ranges were used as span anchors)" if anchors else ""))
            return VisualTranslation(None, anchors, warnings)

    frame_masks: dict[int, np.ndarray] | None = None
    frame_size: tuple[int, int] | None = None
    sha256 = raw.get("source_sha256") or (selection or {}).get("source_sha256") or (detections or {}).get("source_sha256") or (source.sha256 if source else None)
    if detections is not None:
        if selection and selection.get("analysis_id") and detections.get("analysis_id") and selection["analysis_id"] != detections["analysis_id"]:
            warnings.append("selection and detections come from different analyses; visual prompt ignored")
            return VisualTranslation(None, anchors, warnings)
        if source and source.sha256 and detections.get("source_sha256") and detections["source_sha256"] != source.sha256:
            warnings.append("detections were computed on different media bytes than the source; visual prompt ignored")
            return VisualTranslation(None, anchors, warnings)
        track_id, detection_id = _select_object(detections, selection, raw, warnings)
        if track_id is not None or detection_id is not None:
            frame_masks, span = _frame_masks(detections, track_id, detection_id, warnings)
            if frame_masks:
                if interval is None and span:
                    interval = (max(0.0, span[0]), min(span[1], source_duration))
                    anchors.append(("+", *interval))
                frame_size = (int(detections["width"]), int(detections["height"])) if detections.get("width") and detections.get("height") else None
                frame_rate = frame_rate or detections.get("fps")
                object_id = object_id or (f"track:{track_id}" if track_id is not None else detection_id)
            else:
                frame_masks = None

    if not (frame_masks or mask_video_ref):
        if video_ref and not (detections is not None or selection):
            warnings.append("video_ref given without masks; visual prompt ignored")
        return VisualTranslation(None, anchors, warnings)
    if not video_ref:
        warnings.append("per-frame masks are available but the original video was not referenced (metadata.source.original_ref or visual_prompt.video_ref); using text + span anchors only")
        return VisualTranslation(None, anchors, warnings)
    if not backend_supports_visual:
        warnings.append("backend does not support visual prompting; using text and span anchors only")
        return VisualTranslation(None, anchors, warnings)
    if interval is None:
        plus = [(a, b) for _, a, b in anchors]
        # Generic shape: the union of the given time ranges is where the model should look.
        interval = (min(a for a, _ in plus), max(b for _, b in plus)) if plus else (0.0, source_duration)
    video_path = resolve_ref(str(video_ref), settings, workdir=workdir)
    mask_path = resolve_ref(str(mask_video_ref), settings, workdir=workdir) if mask_video_ref else None
    visual = VisualInput(
        video_path=str(video_path), interval=interval, mask_video_path=str(mask_path) if mask_path else None, frame_masks=frame_masks,
        frame_rate=float(frame_rate) if frame_rate else None, frame_size=frame_size, object_id=str(object_id) if object_id else None,
        label=str(label) if label else None, source_sha256=sha256,
    )
    return VisualTranslation(visual, anchors, warnings)
