"""Cross-team Test 2 (Vision -> Audio) against REAL Person 2 artefacts.

Unlike ``tests/test_visual_prompts.py``, which hand-builds documents that *look* like
Person 2's schema, this module loads the detections/selection JSON that Person 2's own
code actually emitted (``vision/soundcanvas_cv`` on ``origin/vision/validated-pipeline``,
real YOLO segmentation + ByteTrack on real pedestrian footage) and drives this service
with it end to end:

1. ``translate_visual_prompt`` must find the selected track's per-frame RLE masks,
   the selection's interval and the native frame size.
2. ``prepare_masked_video`` must decode exactly the native frames of that interval and
   build masks in processor polarity from their RLE.
3. ``POST /separate`` must accept Person 4's envelope carrying both documents and return
   full-timeline stems whose spans are the selected interval.

No SAM Audio inference happens here: ``AUDIO_BACKEND=mock``. See
``fixtures/vision/README.md`` for the exact commands, commit and weights that produced
the artefacts, and for how to regenerate them.

The whole module skips when the fixtures are absent (they are produced by Person 2's
pipeline, which needs YOLO weights and the downloaded clip).
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from conftest import have_ffmpeg

from sam_audio_service.config import load_settings
from sam_audio_service.contracts import SourceRef
from sam_audio_service.visual import probe_video, verify_sha256
from sam_audio_service.visual_prompts import translate_visual_prompt

SERVICE_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = SERVICE_ROOT / "fixtures" / "vision"
GENERATED = SERVICE_ROOT / "fixtures" / "generated" / "vision"
DETECTIONS_JSON = FIXTURES / "detections.json"
SELECTION_JSON = FIXTURES / "selection.json"
VIDEO_NAME = "pedestrians.mp4"

pytest.importorskip("torch", reason="prepare_masked_video builds torch tensors")
pytestmark = pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg/ffprobe are required to decode the clip")


def _video_path() -> Path | None:
    """The clip Person 2 analysed: committed next to the JSON, or in the ignored cache."""
    for candidate in (FIXTURES / VIDEO_NAME, GENERATED / VIDEO_NAME):
        if candidate.is_file():
            return candidate
    return None


def _require_fixtures() -> tuple[dict, dict, Path]:
    if not (DETECTIONS_JSON.is_file() and SELECTION_JSON.is_file()):
        pytest.skip(f"Person 2 artefacts not present; see {FIXTURES}/README.md")
    video = _video_path()
    if video is None:
        pytest.skip(f"{VIDEO_NAME} not present; regenerate it per {FIXTURES}/README.md")
    return json.loads(DETECTIONS_JSON.read_text()), json.loads(SELECTION_JSON.read_text()), video


@pytest.fixture(scope="module")
def artefacts() -> tuple[dict, dict, Path]:
    return _require_fixtures()


@pytest.fixture(scope="module")
def probe(artefacts) -> dict:
    return probe_video(str(artefacts[2]))


@pytest.fixture
def settings(artefacts, monkeypatch):
    """Settings whose media root is the directory holding the clip they analysed."""
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    monkeypatch.setenv("AUDIO_MEDIA_ROOT", str(artefacts[2].parent))
    monkeypatch.delenv("AUDIO_ALLOWED_MEDIA_HOSTS", raising=False)
    return load_settings()


@pytest.fixture(scope="module")
def clip_wav(artefacts, tmp_path_factory) -> Path:
    """The 48 kHz stereo audio of the clip, exactly what Person 4's worker would POST."""
    out = tmp_path_factory.mktemp("person2") / "clip48k.wav"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(artefacts[2]), "-vn", "-ac", "2",
         "-ar", "48000", "-c:a", "pcm_s16le", str(out)],
        check=True, capture_output=True, timeout=300,
    )
    return out


def _interval(selection: dict) -> tuple[float, float]:
    return float(selection["interval"]["start_seconds"]), float(selection["interval"]["end_seconds"])


def _native_frames(selection: dict, fps: float) -> int:
    """Frames whose native timestamp k/fps lies in [start, end) - what the engine crops to."""
    start, end = _interval(selection)
    return max(0, math.ceil(end * fps - 1e-6) - max(0, math.ceil(start * fps - 1e-6)))


def _first_native_frame(selection: dict, fps: float) -> int:
    return max(0, math.ceil(_interval(selection)[0] * fps - 1e-6))


def _masked_frame_indices(detections: dict, selection: dict) -> set[int]:
    """Frames inside the interval where the selected track actually carries an instance mask."""
    start, end = _interval(selection)
    return {
        int(frame["frame_index"])
        for frame in detections["frames"]
        for obj in frame["objects"]
        if obj.get("track_id") == selection["track_id"] and obj.get("mask")
        and start - 1e-7 <= frame["timestamp_seconds"] < end - 1e-7
    }


def _translate(detections, selection, video, settings, duration):
    return translate_visual_prompt(
        {"selection": selection, "detections": detections},
        source=SourceRef(original_ref=video.resolve().as_uri(), sha256=detections["source_sha256"]),
        source_duration=duration,
        settings=settings,
        backend_supports_visual=True,
    )


# --------------------------------------------------------------- the artefacts themselves
def test_fixture_bytes_are_the_analysed_bytes(artefacts, probe):
    """The SHA256 their documents claim must be the clip we kept, or masks are meaningless."""
    detections, selection, video = artefacts
    verify_sha256(str(video), detections["source_sha256"])  # raises InvalidMedia on mismatch
    assert selection["source_sha256"] == detections["source_sha256"]
    assert selection["analysis_id"] == detections["analysis_id"]
    assert selection["video_id"] == detections["video_id"] == f"sha256:{detections['source_sha256']}"
    assert (detections["width"], detections["height"]) == (probe["width"], probe["height"])
    assert detections["fps"] == pytest.approx(probe["frame_rate"], abs=1e-6)
    # Masks only exist at every native frame when they sampled at (or above) native FPS.
    assert detections["sampling_fps"] == detections["fps"]
    assert detections["analyzed_frames"] == detections["processed_frames"]


# ------------------------------------------------------------------------ 1. translation
def test_translate_real_documents(artefacts, probe, settings):
    detections, selection, video = artefacts
    duration = float(probe["duration"])
    translation = _translate(detections, selection, video, settings, duration)

    assert translation.visual is not None, translation.warnings
    visual = translation.visual
    assert Path(visual.video_path) == video.resolve()
    assert visual.interval == pytest.approx(_interval(selection))
    assert visual.frame_size == (detections["width"], detections["height"])
    assert visual.frame_rate == pytest.approx(detections["fps"])
    assert visual.source_sha256 == detections["source_sha256"]
    assert visual.object_id == f"track:{selection['track_id']}"
    assert visual.label == selection["label"]
    assert translation.anchors == []

    # Exactly the selected track's masked observations, decoded at native resolution.
    expected = _masked_frame_indices(detections, selection)
    assert expected, "the selected track has no instance masks in the detections document"
    assert set(visual.frame_masks) == expected
    for index, mask in visual.frame_masks.items():
        assert mask.shape == (detections["height"], detections["width"]), index
        assert set(np.unique(mask)).issubset({0, 1})
    assert any(mask.any() for mask in visual.frame_masks.values())


def test_translation_rejects_other_media_bytes(artefacts, probe, settings):
    """Polarity/alignment only mean anything for the analysed bytes; a wrong hash must not pass."""
    detections, selection, video = artefacts
    translation = translate_visual_prompt(
        {"selection": selection, "detections": detections},
        source=SourceRef(original_ref=video.resolve().as_uri(), sha256="b" * 64),
        source_duration=float(probe["duration"]),
        settings=settings,
        backend_supports_visual=True,
    )
    assert translation.visual is None
    assert any("different media bytes" in w for w in translation.warnings)


# ----------------------------------------------------------------- 2. frame/mask decoding
def test_prepare_masked_video_from_real_masks(artefacts, probe, settings):
    from sam_audio_service.visual import prepare_masked_video

    detections, selection, video = artefacts
    visual = _translate(detections, selection, video, settings, float(probe["duration"])).visual
    assert visual is not None

    frames, masks = prepare_masked_video(visual, size=64, mask_mode="keep_object", ffmpeg="ffmpeg", ffprobe="ffprobe")
    expected_frames = _native_frames(selection, float(detections["fps"]))
    assert expected_frames > 1
    assert frames.shape == (expected_frames, 3, 64, 64)
    assert masks.shape == (expected_frames, 1, 64, 64)
    assert frames.dtype == masks.dtype == __import__("torch").uint8

    # keep_object polarity: mask_videos computes video * mask.eq(0), so 0 = keep the object.
    object_mask = 1 - masks.numpy()[:, 0]
    assert object_mask.max() == 1, "no frame in the interval keeps any object pixel"
    per_frame = object_mask.reshape(expected_frames, -1).sum(axis=1)
    assert (per_frame > 0).sum() >= 1
    # Frames where their document has no observation of the track stay empty; never reused.
    masked_indices = _masked_frame_indices(detections, selection)
    first_index = _first_native_frame(selection, float(detections["fps"]))
    for offset, count in enumerate(per_frame):
        if first_index + offset not in masked_indices:
            assert count == 0, f"frame {first_index + offset} has no mask but is not empty"

    # "notebook" polarity is the straight object mask; the two must be complementary.
    _, notebook = prepare_masked_video(visual, size=64, mask_mode="notebook", ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert np.array_equal(notebook.numpy(), 1 - masks.numpy())


# -------------------------------------------------------------------- 3. the HTTP contract
def test_separate_endpoint_with_person2_documents(artefacts, clip_wav, monkeypatch):
    from fastapi.testclient import TestClient

    from sam_audio_service.api import create_app

    detections, selection, video = artefacts
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    monkeypatch.setenv("AUDIO_MEDIA_ROOT", str(video.parent))
    monkeypatch.delenv("AUDIO_ALLOWED_MEDIA_HOSTS", raising=False)
    settings = load_settings()

    metadata = {
        "prompts": ["person"],
        "source": {
            "asset_id": detections["video_id"],
            "original_ref": video.resolve().as_uri(),
            "sha256": detections["source_sha256"],
            "frame_rate": detections["fps"],
        },
        "visual_prompt": {"selection": selection, "detections": detections},
    }
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/separate",
            files={"audio": ("clip48k.wav", clip_wav.read_bytes(), "audio/wav")},
            data={"metadata": json.dumps(metadata)},
        )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["model"]["is_mock"] is True
    assert any("MOCK" in w for w in body["warnings"])
    source = body["source"]
    assert source["sample_rate"] == 48000 and source["channels"] == 2
    assert [s["name"] for s in body["stems"]] == ["person", "residual"]
    start, end = _interval(selection)
    for stem in body["stems"]:
        # Full timeline: visual prompting separates only the interval but pads back out.
        assert stem["start_time"] == 0.0
        assert stem["sample_count"] == source["sample_count"]
        assert stem["duration"] == pytest.approx(source["duration"], abs=1e-6)
        assert stem["sample_rate"] == 48000 and stem["channels"] == 2
    target = body["stems"][0]
    assert target["spans"] == [[round(start, 3), round(end, 3)]]
    assert target["prompt"] == "person"
    # The translator must not have silently fallen back to anchors only.
    assert not any("per-frame masks" in w or "visual prompt ignored" in w for w in body["warnings"]), body["warnings"]
