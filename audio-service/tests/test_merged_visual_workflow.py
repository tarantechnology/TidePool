"""Issue #4, end to end: the MERGED backend workflow must reach SAM Audio's visual inputs.

``tests/test_person2_artifacts.py`` proves Person 2's real documents translate when *we*
build the request. ``tests/test_integration_person4_backend.py`` proves a *synthetic*
selection survives Person 4's backend. Neither proves the claim issue #4 actually tracks:
that the normal product workflow — upload, store detections, click an object, press
process — delivers the original video reference **and** per-frame masks to the model
input, on Person 2's real artefacts.

So this module runs Person 4's real backend (``app.*`` from the repo root, imported and
never modified) against a live instance of this service, with
``INFERENCE_ADAPTER=http``, ``AUDIO_BACKEND=mock`` and ``AUDIO_MEDIA_ROOT`` pointing at
the backend's ``STORAGE_ROOT`` — the single-shared-volume arrangement
``audio-service/compose.audio.yaml`` configures (``./storage`` -> ``/data`` in both
containers) — and asserts on OUR side of the HTTP boundary that:

1. the request carried ``metadata.source.original_ref`` (a ``file://`` URI of the
   project's immutable upload) plus the ``sha256`` of exactly those bytes;
2. ``metadata.visual_prompt`` carried both Person 2 documents, as
   ``{"selection": ..., "detections": ...}``;
3. ``Engine.translate_visual`` produced a real ``VisualInput`` with per-frame masks —
   not the anchors-only fallback — and those masks decode against the uploaded video;
4. the stem ``spans`` are the selected interval, and the stems span the full timeline.

The hook used for (3) is a test-only ``monkeypatch`` of ``Engine.translate_visual`` that
records the translation and then delegates to the real implementation; the server runs in
a thread of this process, so no production code grows a test seam.

``AUDIO_BACKEND=mock`` throughout: no SAM Audio inference happens here (see
``fixtures/vision/README.md``). This proves the handoff, not audio quality — the real
visually guided run still needs a GPU, which is the one remaining item on issue #4.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import uvicorn

from conftest import have_ffmpeg

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO = SERVICE_ROOT.parent
FIXTURES = SERVICE_ROOT / "fixtures" / "vision"
DETECTIONS_JSON = FIXTURES / "detections.json"
SELECTION_JSON = FIXTURES / "selection.json"
VIDEO = FIXTURES / "pedestrians.mp4"

pytest.importorskip("torch", reason="prepare_masked_video builds torch tensors")
pytestmark = [
    pytest.mark.skipif(not (REPO / "app" / "main.py").exists(), reason="Person 4 backend not present"),
    pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg/ffprobe are required"),
    pytest.mark.skipif(
        not (DETECTIONS_JSON.is_file() and SELECTION_JSON.is_file() and VIDEO.is_file()),
        reason=f"Person 2 artefacts not present; see {FIXTURES}/README.md",
    ),
]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def storage_root(tmp_path) -> Path:
    """The shared volume: Person 4's STORAGE_ROOT == our AUDIO_MEDIA_ROOT."""
    root = tmp_path / "storage"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def audio_service(storage_root, monkeypatch):
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    monkeypatch.setenv("AUDIO_MEDIA_ROOT", str(storage_root))
    monkeypatch.delenv("AUDIO_ALLOWED_MEDIA_HOSTS", raising=False)
    from sam_audio_service.api import create_app
    from sam_audio_service.config import load_settings

    settings = load_settings()
    assert settings.media_root == storage_root.resolve()
    port = _free_port()
    config = uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(200):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "audio service did not start"
    yield f"http://127.0.0.1:{port}/separate"
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def backend_client(storage_root, monkeypatch, audio_service):
    """Person 4's FastAPI app, talking to the live audio service over HTTP."""
    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))
    monkeypatch.setenv("INFERENCE_ADAPTER", "http")
    monkeypatch.setenv("SAM_AUDIO_SERVICE_URL", audio_service)
    monkeypatch.syspath_prepend(str(REPO))
    for name in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        sys.modules.pop(name)
    core = importlib.import_module("app.core")
    core.ROOT = storage_root
    core.DB = core.ROOT / "db.sqlite"
    main = importlib.import_module("app.main")
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        yield client, core


@pytest.fixture
def translations(monkeypatch):
    """Test-only hook: record every (metadata, VisualInput, anchors) the engine derives."""
    from sam_audio_service.engine import Engine

    recorded: list[dict] = []
    real = Engine.translate_visual

    def record(self, prompt, shared, metadata, source_duration, warnings, workdir=None):
        before = len(warnings)
        visual, anchors = real(self, prompt, shared, metadata, source_duration, warnings, workdir)
        recorded.append({
            "metadata": metadata,
            "shared": shared,
            "visual": visual,
            "anchors": list(anchors),
            "warnings": list(warnings[before:]),
            "source_duration": source_duration,
            "backend_supports_visual": self.backend.supports_visual,
        })
        return visual, anchors

    monkeypatch.setattr(Engine, "translate_visual", record)
    return recorded


@pytest.fixture
def engine_results(monkeypatch):
    """Record each EngineResult: Person 4 does not forward stem ``spans`` to the editor."""
    from sam_audio_service.engine import Engine

    recorded = []
    real = Engine.separate

    def record(self, audio, metadata, workdir=None):
        result = real(self, audio, metadata, workdir)
        recorded.append(result)
        return result

    monkeypatch.setattr(Engine, "separate", record)
    return recorded


@pytest.fixture
def model_inputs(monkeypatch):
    """Record what the *backend* is actually called with: the model-input boundary itself.

    With ``AUDIO_BACKEND=sam`` this is where ``prepare_masked_video`` runs and the frames
    and masks become processor tensors, so a ``BackendPrompt.visual`` here is the proof
    that the merged workflow reaches SAM Audio's visual inputs.
    """
    from sam_audio_service.backends.mock import MockBackend

    recorded: list[dict] = []
    real = MockBackend.separate

    def record(self, mixture, prompt, options):
        recorded.append({"frames": int(mixture.shape[-1]), "prompt": prompt, "sample_rate": self.sample_rate})
        return real(self, mixture, prompt, options)

    monkeypatch.setattr(MockBackend, "separate", record)
    return recorded


def _wait(client, url, headers):
    for _ in range(1200):
        data = client.get(url, headers=headers).json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(0.05)
    pytest.fail("job timed out")


@pytest.fixture(scope="module")
def artefacts() -> tuple[dict, dict, str]:
    detections = json.loads(DETECTIONS_JSON.read_text())
    selection = json.loads(SELECTION_JSON.read_text())
    sha = hashlib.sha256(VIDEO.read_bytes()).hexdigest()
    assert detections["source_sha256"] == selection["source_sha256"] == sha
    return detections, selection, sha


def test_merged_workflow_delivers_video_and_masks(backend_client, translations, engine_results, model_inputs, artefacts):
    """The product workflow (upload -> detections -> click -> process) reaches VisualInput."""
    from sam_audio_service.audio_io import decode
    from sam_audio_service.visual import prepare_masked_video

    detections, selection, sha = artefacts
    client, core = backend_client

    project = client.post("/api/projects").json()
    pid = project["project_id"]
    headers = {"Authorization": "Bearer " + project["access_token"]}
    with VIDEO.open("rb") as handle:
        upload = client.post(f"/api/projects/{pid}/upload", headers=headers,
                             files={"file": ("pedestrians.mp4", handle, "video/mp4")})
    assert upload.status_code == 200, upload.text
    media = upload.json()["metadata"]
    assert media["source_sha256"] == sha, "the backend stored other bytes than Person 2 analysed"
    assert (media["width"], media["height"]) == (detections["width"], detections["height"])
    assert media["frame_rate"] == pytest.approx(detections["fps"])

    # Person 2's detections document, stored by the backend exactly as Person 3 would PUT it.
    saved = client.put(f"/api/projects/{pid}/visual-metadata", headers=headers, json=detections)
    assert saved.status_code == 200, saved.text
    assert client.get(f"/api/projects/{pid}/visual-metadata", headers=headers).json()["analysis_id"] == detections["analysis_id"]

    # Person 2's selection document, sent the way Person 3's editor sends a click.
    start_job = client.post(f"/api/projects/{pid}/process", headers=headers,
                            json={"prompts": ["person"], "visual_prompt": selection})
    assert start_job.status_code == 202, start_job.text
    job = _wait(client, "/api/jobs/" + start_job.json()["job_id"], headers)
    assert job["status"] == "completed", job

    # ---------------------------------------------------------------- 1. what we received
    assert len(translations) == 1, translations
    record = translations[0]
    metadata = record["metadata"]
    original = Path(core.project_dir(pid) / "original")
    stored_video = next(original.iterdir())

    assert metadata.source is not None, "backend sent no metadata.source; masks cannot be used"
    assert metadata.source.original_ref == stored_video.resolve().as_uri()
    assert metadata.source.original_ref.startswith("file://")
    assert metadata.source.sha256 == sha
    assert metadata.source.asset_id == pid
    assert metadata.source.duration == pytest.approx(media["duration"])
    assert metadata.source.frame_rate == pytest.approx(detections["fps"])
    assert metadata.asset_id == pid
    assert metadata.request_id == start_job.json()["job_id"]
    # The exact merged envelope: both Person 2 documents under metadata.visual_prompt.
    assert set(metadata.visual_prompt) == {"selection", "detections"}
    assert metadata.visual_prompt["selection"] == selection
    assert metadata.visual_prompt["detections"] == detections
    assert record["shared"] is metadata.visual_prompt

    # ------------------------------------------------------- 2. a VisualInput, not anchors
    visual = record["visual"]
    assert visual is not None, record["warnings"]
    assert not record["warnings"], record["warnings"]
    interval = (selection["interval"]["start_seconds"], selection["interval"]["end_seconds"])
    assert visual.interval == pytest.approx(interval)
    assert Path(visual.video_path) == stored_video.resolve()
    assert visual.source_sha256 == sha
    assert visual.frame_size == (detections["width"], detections["height"])
    assert visual.frame_rate == pytest.approx(detections["fps"])
    assert visual.object_id == f"track:{selection['track_id']}"
    assert visual.label == selection["label"]
    assert ("+", *interval) in record["anchors"]

    expected_masked = {
        int(frame["frame_index"])
        for frame in detections["frames"]
        for obj in frame["objects"]
        if obj.get("track_id") == selection["track_id"] and obj.get("mask")
        and interval[0] - 1e-7 <= frame["timestamp_seconds"] < interval[1] - 1e-7
    }
    assert expected_masked, "the selected track carries no instance masks"
    assert set(visual.frame_masks) == expected_masked
    assert any(mask.any() for mask in visual.frame_masks.values())

    # ------------------------------------------- the model-input boundary itself got it
    assert len(model_inputs) == 1, model_inputs
    call = model_inputs[0]
    assert call["prompt"].visual is visual, "the engine dropped the VisualInput before the backend"
    assert call["prompt"].text == "person"
    # Visually prompted requests run the interval as one window, never chunked.
    assert call["frames"] == round((interval[1] - interval[0]) * call["sample_rate"])

    # The file:// reference really resolves inside this service: decode frames + masks.
    frames, masks = prepare_masked_video(visual, size=64, mask_mode="keep_object")
    native = round((interval[1] - interval[0]) * detections["fps"])
    assert frames.shape == (native, 3, 64, 64)
    assert masks.shape == (native, 1, 64, 64)
    assert (1 - masks.numpy()[:, 0]).max() == 1, "no object pixels survived the handoff"

    # ---------------------------------------------------------------- 3. the stems we sent
    inference = job["inference"]
    assert inference["model"]["is_mock"] is True
    assert inference["source"]["asset_id"] == pid
    assert inference["request_id"] == start_job.json()["job_id"]
    assert not [w for w in inference["warnings"] if "per-frame masks" in w or "visual prompt ignored" in w
                or "span anchors only" in w], inference["warnings"]
    assert [s["name"] for s in job["stems"]] == ["person", "residual"]

    audio = decode(data=(core.project_dir(pid) / "extracted" / "audio.wav").read_bytes())
    stems = []
    for stem in job["stems"]:
        assert stem["start_offset"] == 0
        assert stem["duration"] == pytest.approx(audio.duration, abs=1e-3)
        response = client.get(stem["audio_url"], headers=headers)
        assert response.status_code == 200
        samples = decode(data=response.content).samples
        assert samples.shape == audio.samples.shape, "stem is not full length"
        stems.append(samples)
    np.testing.assert_allclose(sum(stems), audio.samples, atol=1e-6)

    # The target's spans are exactly the selected interval on the project timeline.
    assert len(engine_results) == 1
    produced = {stem.name: stem for stem in engine_results[0].stems}
    assert produced["person"].spans == [[round(interval[0], 3), round(interval[1], 3)]]
    assert produced["residual"].spans is None
    assert produced["person"].samples.shape == audio.samples.shape


def test_merged_workflow_spans_are_the_selected_interval(backend_client, translations, artefacts):
    """Separate assertion on the response contract Person 4 forwards to the editor."""
    detections, selection, sha = artefacts
    client, core = backend_client
    project = client.post("/api/projects").json()
    pid = project["project_id"]
    headers = {"Authorization": "Bearer " + project["access_token"]}
    with VIDEO.open("rb") as handle:
        assert client.post(f"/api/projects/{pid}/upload", headers=headers,
                           files={"file": ("pedestrians.mp4", handle, "video/mp4")}).status_code == 200
    assert client.put(f"/api/projects/{pid}/visual-metadata", headers=headers, json=detections).status_code == 200
    narrowed = {**selection, "interval": {"start_seconds": 1.0, "end_seconds": 3.0}}
    started = client.post(f"/api/projects/{pid}/process", headers=headers,
                          json={"prompts": ["person"], "visual_prompt": narrowed})
    assert started.status_code == 202, started.text
    job = _wait(client, "/api/jobs/" + started.json()["job_id"], headers)
    assert job["status"] == "completed", job
    assert translations and translations[0]["visual"] is not None, translations
    assert translations[0]["visual"].interval == pytest.approx((1.0, 3.0))
    # Only the frames of the narrowed interval are handed to the model.
    assert set(translations[0]["visual"].frame_masks) <= set(range(10, 30))

    from sam_audio_service.audio_io import decode

    audio = decode(data=(core.project_dir(pid) / "extracted" / "audio.wav").read_bytes())
    target = decode(data=client.get(job["stems"][0]["audio_url"], headers=headers).content).samples
    assert target.shape == audio.samples.shape
    assert job["inference"]["source"]["duration"] == pytest.approx(audio.duration, abs=1e-3)


def test_backend_rejects_selection_whose_bytes_changed(backend_client, artefacts, tmp_path):
    """The merged workflow's identity guard: Person 4 refuses detections for other bytes."""
    import subprocess

    detections, selection, _ = artefacts
    client, _core = backend_client
    other = tmp_path / "other.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=768x576:r=10:d=7",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(other)], check=True)
    project = client.post("/api/projects").json()
    pid = project["project_id"]
    headers = {"Authorization": "Bearer " + project["access_token"]}
    with other.open("rb") as handle:
        assert client.post(f"/api/projects/{pid}/upload", headers=headers,
                           files={"file": ("other.mp4", handle, "video/mp4")}).status_code == 200
    assert client.put(f"/api/projects/{pid}/visual-metadata", headers=headers, json=detections).status_code == 200
    rejected = client.post(f"/api/projects/{pid}/process", headers=headers,
                           json={"prompts": ["person"], "visual_prompt": selection})
    assert rejected.status_code == 422, rejected.text
    assert "source_sha256" in rejected.text


def test_process_without_a_selection_forwards_detections_that_cannot_steer(backend_client, translations, artefacts):
    """Pinned gap: no click, but stored detections.

    ``prepare_payload`` then takes its ``selection is None`` branch (``app/visual.py``:
    ``payload['visual_prompt'] = raw``), which forwards the whole detections document and
    attaches **no** ``metadata.source``. With no selected object and no video reference we
    can only run text, with no anchors, and we say so in ``warnings``. The request still
    succeeds — this is waste, not breakage.
    """
    detections, _selection, _sha = artefacts
    client, _core = backend_client
    project = client.post("/api/projects").json()
    pid = project["project_id"]
    headers = {"Authorization": "Bearer " + project["access_token"]}
    with VIDEO.open("rb") as handle:
        assert client.post(f"/api/projects/{pid}/upload", headers=headers,
                           files={"file": ("pedestrians.mp4", handle, "video/mp4")}).status_code == 200
    assert client.put(f"/api/projects/{pid}/visual-metadata", headers=headers, json=detections).status_code == 200

    started = client.post(f"/api/projects/{pid}/process", headers=headers, json={"prompts": ["person"]})
    assert started.status_code == 202, started.text
    job = _wait(client, "/api/jobs/" + started.json()["job_id"], headers)
    assert job["status"] == "completed", job

    assert len(translations) == 1
    record = translations[0]
    assert record["metadata"].source is None, "no selection, so the backend sends no source block"
    assert set(record["metadata"].visual_prompt) == set(detections), "the raw detections document is forwarded"
    assert record["visual"] is None, "without a selected object there is nothing to prompt with"
    assert record["anchors"] == []
    assert any("without a selected object" in w for w in record["warnings"]), record["warnings"]
    assert any("without a selected object" in w for w in job["inference"]["warnings"]), job["inference"]["warnings"]
