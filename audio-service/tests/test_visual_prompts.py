"""Translation of Person 2's visual guidance (selection / detections 1.0) into model inputs.

Uses Person 2's documented RLE codec (vision/docs/CONTRACT.md on their branch): row-major,
alternating zero/one runs starting with zeros.
"""
import json
import subprocess

import numpy as np
import pytest

from sam_audio_service.audio_io import DecodedAudio
from sam_audio_service.config import load_settings
from sam_audio_service.contracts import SeparationMetadata, SourceRef
from sam_audio_service.engine import Engine
from sam_audio_service.visual_prompts import translate_visual_prompt

pytest.importorskip("sam_audio_service.visual")


def encode_mask(mask: np.ndarray) -> dict:
    flat = np.asarray(mask, dtype=np.uint8).ravel()
    edges = np.flatnonzero(np.diff(flat)) + 1
    counts = np.diff(np.r_[0, edges, flat.size]).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)
    return {"encoding": "rle", "order": "C", "size": list(mask.shape), "counts": counts, "foreground_value": 1}


def make_video(path, seconds=1.0, fps=10, size="64x48"):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c=red:s={size}:r={fps}:d={seconds}", "-c:v", "mpeg4", str(path)], check=True)
    return path


def detections_doc(fps=10, frames=10, w=64, h=48, track=7, with_masks=True, start_frame=2, end_frame=8):
    doc = {"schema_version": "1.0", "video_id": "vid-1", "source_sha256": "a" * 64, "analysis_id": "run-1", "width": w, "height": h, "fps": fps, "media_type": "video", "processed_duration_seconds": frames / fps, "frames": []}
    for i in range(frames):
        objects = []
        if start_frame <= i < end_frame:
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[10:30, 20:40] = 1
            objects.append({"object_id": f"track:{track}", "detection_id": f"frame:{i}:det:0", "label": "person", "class_name": "person", "class_id": 0, "confidence": 0.9, "bbox_xyxy": [20, 10, 40, 30], "bbox_normalized": [0.3125, 0.2083, 0.625, 0.625], "track_id": track, "mask": encode_mask(mask) if with_masks else None})
        doc["frames"].append({"frame_index": i, "timestamp_seconds": i / fps, "valid_until_seconds": (i + 1) / fps, "objects": objects})
    return doc


def selection_doc(track=7, start=0.2, end=0.8):
    return {"schema_version": "1.0", "kind": "visual_selection", "video_id": "vid-1", "analysis_id": "run-1", "source": "demo.mp4", "source_sha256": "a" * 64, "frame_width": 64, "frame_height": 48, "frame_index": 2, "timestamp_seconds": 0.2, "clicked_at_seconds": 0.25, "interval": {"start_seconds": start, "end_seconds": end}, "object_id": f"track:{track}", "detection_id": "frame:2:det:0", "label": "person", "class_name": "person", "class_id": 0, "confidence": 0.9, "bbox_xyxy": [20, 10, 40, 30], "bbox_normalized": [0.3125, 0.2083, 0.625, 0.625], "track_id": track, "mask": None, "audio_association": "unverified"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    monkeypatch.setenv("AUDIO_MEDIA_ROOT", str(tmp_path))
    video = make_video(tmp_path / "demo.mp4")
    return load_settings(), video


def test_selection_alone_gives_anchors_only(env):
    settings, _ = env
    t = translate_visual_prompt(selection_doc(), source=None, source_duration=1.0, settings=settings, backend_supports_visual=True)
    assert t.visual is None and t.anchors == [("+", 0.2, 0.8)]
    assert any("per-frame masks" in w for w in t.warnings)


def test_detections_without_video_reference(env):
    settings, _ = env
    t = translate_visual_prompt({"selection": selection_doc(), "detections": detections_doc()}, source=None, source_duration=1.0, settings=settings, backend_supports_visual=True)
    assert t.visual is None and t.anchors == [("+", 0.2, 0.8)]
    assert any("original video was not referenced" in w for w in t.warnings)


def test_envelope_with_source_ref_builds_visual_input(env):
    settings, video = env
    src = SourceRef(asset_id="p1", original_ref=video.as_uri(), sha256="a" * 64)
    t = translate_visual_prompt({"selection": selection_doc(), "detections": detections_doc()}, source=src, source_duration=1.0, settings=settings, backend_supports_visual=True)
    assert t.visual is not None
    assert t.visual.interval == (0.2, 0.8) and t.visual.frame_rate == 10 and t.visual.frame_size == (64, 48)
    assert sorted(t.visual.frame_masks) == list(range(2, 8))
    assert t.visual.frame_masks[2].shape == (48, 64) and t.visual.frame_masks[2][15, 25] == 1 and t.visual.frame_masks[2][0, 0] == 0
    assert t.visual.object_id == "track:7" and t.visual.label == "person" and t.visual.source_sha256 == "a" * 64


def test_detections_only_picks_single_masked_track(env):
    settings, video = env
    raw = {**detections_doc(), "video_ref": video.as_uri()}
    t = translate_visual_prompt(raw, source=None, source_duration=1.0, settings=settings, backend_supports_visual=True)
    assert t.visual is not None and t.visual.interval == (0.2, 0.8)
    assert any("only masked track" in w for w in t.warnings)


def test_bbox_only_detections_fall_back(env):
    settings, video = env
    raw = {"selection": selection_doc(), "detections": detections_doc(with_masks=False), "video_ref": video.as_uri()}
    t = translate_visual_prompt(raw, source=None, source_duration=1.0, settings=settings, backend_supports_visual=True)
    assert t.visual is None and any("no instance masks" in w for w in t.warnings)


def test_identity_mismatch_is_rejected(env):
    settings, video = env
    src = SourceRef(original_ref=video.as_uri(), sha256="b" * 64)
    t = translate_visual_prompt({"selection": selection_doc(), "detections": detections_doc()}, source=src, source_duration=1.0, settings=settings, backend_supports_visual=True)
    assert t.visual is None and any("different media bytes" in w for w in t.warnings)


def test_engine_places_visual_result_on_full_timeline(env):
    settings, video = env
    engine = Engine(settings)
    rate = 48000
    t = np.arange(rate) / rate
    mix = np.stack([np.sin(2 * np.pi * 1000 * t)] * 2).astype(np.float32)
    meta = SeparationMetadata(prompts=["person"], source={"original_ref": video.as_uri(), "sha256": "a" * 64}, visual_prompt={"selection": selection_doc(), "detections": detections_doc()})
    result = engine.separate(DecodedAudio(mix, rate, rate, 2), meta)
    target = result.stems[0].samples
    assert target.shape == mix.shape
    assert np.abs(target[:, : int(0.19 * rate)]).max() == 0 and np.abs(target[:, int(0.81 * rate):]).max() == 0
    assert np.abs(target[:, int(0.3 * rate): int(0.7 * rate)]).max() > 0.5
    assert result.stems[0].spans == [[0.2, 0.8]]
    total = sum(s.samples for s in result.stems)
    assert np.max(np.abs(total - mix)) < 1e-5
    assert result.model["chunked"] is False


def test_visual_interval_longer_than_limit_is_rejected(env, monkeypatch):
    from sam_audio_service.errors import UnsupportedOperation

    monkeypatch.setenv("AUDIO_MAX_VISUAL_INTERVAL_SECONDS", "0.5")
    settings, video = env
    engine = Engine(load_settings())
    rate = 48000
    mix = np.zeros((2, rate), dtype=np.float32)
    meta = SeparationMetadata(prompts=["person"], source={"original_ref": video.as_uri()}, visual_prompt={"selection": selection_doc(), "detections": detections_doc()})
    with pytest.raises(UnsupportedOperation, match="limited to 0s|limited to 1s"):
        engine.separate(DecodedAudio(mix, rate, rate, 2), meta)


def test_interval_is_clamped_to_processed_window(env):
    settings, video = env
    src = SourceRef(original_ref=video.as_uri(), sha256="a" * 64)
    raw = {"selection": selection_doc(start=0.2, end=5.0), "detections": detections_doc()}  # processed window is 1.0 s
    t = translate_visual_prompt(raw, source=src, source_duration=10.0, settings=settings, backend_supports_visual=True)
    assert t.visual is not None and t.visual.interval == (0.2, 1.0)
    assert t.anchors == [("+", 0.2, 1.0)] and any("clamped" in w for w in t.warnings)
