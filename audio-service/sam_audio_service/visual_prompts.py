"""Translate visual guidance (Person 2's documents) into model-ready inputs.

Person 2 owns detection, segmentation, tracking and selection. This module only maps
what they produce onto what SAM Audio can actually consume:

* explicit auditory time ranges -> ``+`` span anchors (works with text prompts alone)
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
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .backends.base import Anchor, VisualInput
from .config import Settings
from .contracts import SourceRef, TimeRange, VisualPrompt
from .errors import AudioServiceError, InvalidRequest, UnsupportedOperation
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
    start = float(start)
    end = source_duration if end is None else float(end)
    if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or end > source_duration + 1e-6:
        raise InvalidRequest("selection interval must lie within the source audio duration")
    return (start, min(end, source_duration))


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


def _frame_masks(detections: dict[str, Any], track_id: int | None, detection_id: str | None, warnings: list[str], interval: tuple[float, float] | None = None) -> tuple[dict[int, np.ndarray], tuple[float, float] | None]:
    from .visual import decode_rle_mask  # pure numpy, kept in the visual module next to the frame loader

    masks: dict[int, np.ndarray] = {}
    first = last = None
    bbox_only = 0
    for frame in detections.get("frames", []):
        if interval is not None:
            timestamp = float(frame.get("timestamp_seconds", -1))
            if timestamp < interval[0] - 1e-7 or timestamp >= interval[1] - 1e-7:
                continue
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


def _validate_coverage(detections: dict[str, Any], interval: tuple[float, float], fps: float,
                       track_id: int | None, detection_id: str | None) -> None:
    """Missing records are unknown, not evidence that an object is absent."""
    if not math.isfinite(fps) or fps <= 0:
        raise InvalidRequest("native video FPS is required for frame coverage validation")
    first = max(0, math.ceil(interval[0] * fps - 1e-6))
    end = max(first, math.ceil(interval[1] * fps - 1e-6))
    records = {}
    for frame in detections.get("frames", []):
        index = frame.get("frame_index")
        if not isinstance(index, int) or isinstance(index, bool) or index in records:
            raise InvalidRequest("detection frame indices must be unique integers")
        records[index] = frame
    for index in range(first, end):
        frame = records.get(index)
        if frame is None:
            raise InvalidRequest(f"sparse detection metadata: native frame {index} is missing; rerun segmentation at native FPS")
        objects = [obj for obj in frame.get("objects", []) if
                   (obj.get("track_id") == track_id if track_id is not None else obj.get("detection_id") == detection_id)]
        if len(objects) > 1:
            raise InvalidRequest(f"ambiguous selected track at frame {index}")
        if objects and not objects[0].get("mask"):
            raise InvalidRequest(f"selected object has no instance mask at frame {index}")
        if objects:
            try:
                from .visual import decode_rle_mask
                decode_rle_mask(objects[0]["mask"])
            except (ValueError, KeyError, TypeError) as exc:
                raise InvalidRequest(f"invalid selected-object mask at frame {index}: {exc}") from exc


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
        if not interval:
            warnings.append("selection has no usable interval on the source timeline")
        if detections is None and not mask_video_ref:
            warnings.append("selection carries one mask for the clicked frame only; per-frame masks (detections document or mask_video_ref) are needed for visual prompting, so supply text plus explicit sound anchors")

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
        processed = detections.get("processed_duration_seconds")
        if interval and isinstance(processed, (int, float)) and interval[1] > processed:
            # Person 2 only guarantees masks inside the processed window; beyond it every
            # frame would read as "object absent", which would silently bias the model.
            warnings.append(f"selection interval end {interval[1]:g}s clamped to the processed window ({processed:g}s)")
            interval = (interval[0], float(processed))
            anchors = [("+", a, min(b, float(processed))) for t, a, b in anchors if a < processed] 
            if interval[1] <= interval[0]:
                warnings.append("selection interval lies entirely outside the processed window; visual prompt ignored")
                return VisualTranslation(None, anchors, warnings)
        track_id, detection_id = _select_object(detections, selection, raw, warnings)
        if track_id is not None or detection_id is not None:
            if interval is None:
                # Find the implicit track extent without allocating full-resolution masks.
                matching_frames = [f for f in detections.get("frames", []) if any(
                    (o.get("track_id") == track_id if track_id is not None else o.get("detection_id") == detection_id)
                    and o.get("mask") for o in f.get("objects", []))]
                if matching_frames:
                    first = min(float(f.get("timestamp_seconds", 0)) for f in matching_frames)
                    last = max(float(f.get("valid_until_seconds", f.get("timestamp_seconds", 0))) for f in matching_frames)
                    interval = (max(0.0, first), min(last, source_duration))
                    if interval[1] <= interval[0]:
                        raise InvalidRequest("selected track is outside the source audio duration")
                    if interval[1] - interval[0] > settings.max_visual_interval_seconds:
                        raise UnsupportedOperation(f"visual prompt interval is limited to {settings.max_visual_interval_seconds:.0f}s; choose a shorter interval")
                    # Visibility is not evidence of audible sound.
            if interval and interval[1] - interval[0] > settings.max_visual_interval_seconds:
                raise UnsupportedOperation(f"visual prompt interval is limited to {settings.max_visual_interval_seconds:.0f}s; choose a shorter interval")
            frame_masks, span = _frame_masks(detections, track_id, detection_id, warnings, interval)
            if frame_masks:
                if interval is None and span:
                    interval = (max(0.0, span[0]), min(span[1], source_duration))
                    # Visibility is not evidence of audible sound.
                frame_size = (int(detections["width"]), int(detections["height"])) if detections.get("width") and detections.get("height") else None
                frame_rate = frame_rate or detections.get("fps")
                object_id = object_id or (f"track:{track_id}" if track_id is not None else detection_id)
            else:
                if video_ref and not mask_video_ref and backend_supports_visual:
                    raise InvalidRequest("visual input has no usable instance masks; rerun segmentation at native FPS")
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
    if detections is not None and frame_masks is not None:
        from .visual import probe_video
        native_fps = probe_video(str(video_path), settings.ffprobe, min(settings.ffmpeg_timeout, 60))["frame_rate"]
        declared_fps = float(frame_rate or 0)
        if not native_fps or not math.isclose(declared_fps, native_fps, rel_tol=1e-4, abs_tol=1e-5):
            raise InvalidRequest("detection FPS does not match the source video; rerun segmentation at native FPS")
        _validate_coverage(detections, interval, native_fps, track_id, detection_id)
    mask_path = resolve_ref(str(mask_video_ref), settings, workdir=workdir) if mask_video_ref else None
    visual = VisualInput(
        video_path=str(video_path), interval=interval, mask_video_path=str(mask_path) if mask_path else None, frame_masks=frame_masks,
        frame_rate=float(frame_rate) if frame_rate else None, frame_size=frame_size, object_id=str(object_id) if object_id else None,
        label=str(label) if label else None, source_sha256=sha256,
    )
    return VisualTranslation(visual, anchors, warnings)
