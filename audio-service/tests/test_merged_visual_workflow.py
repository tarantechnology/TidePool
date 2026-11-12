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
containers), built by the fixtures in ``tests/conftest.py``. On OUR side of the HTTP
boundary it then asserts, one test per claim, that:

1. the request carried ``metadata.source.original_ref`` (a ``file://`` URI of the
   project's immutable upload) plus the ``sha256`` of exactly those bytes, and
   ``metadata.visual_prompt`` carried both Person 2 documents as
   ``{"selection": ..., "detections": ...}``;
2. ``Engine.translate_visual`` produced a real ``VisualInput`` with per-frame masks —
   not the anchors-only fallback;
3. that same ``VisualInput`` reached the backend's ``separate`` call, and its masks
   decode against the uploaded video;
4. the stems are full-timeline and sum back to the source, with the target's spans
   inside the selected interval.

The hooks used are test-only ``monkeypatch`` of ``Engine.translate_visual``,
``Engine.separate`` and ``MockBackend.separate`` that record and then delegate to the real
implementations; the server runs in a thread of this process, so no production code grows
a test seam. The project is uploaded and the detections stored once per module, and each
``/process`` run is recorded once and asserted on by several tests.

``AUDIO_BACKEND=mock`` throughout: no SAM Audio inference happens here (see
``fixtures/vision/README.md``). This proves the handoff, not audio quality — the real
visually guided run still needs a GPU, which is the one remaining item on issue #4.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import REPO, have_ffmpeg, wait_for_job
from test_person2_artifacts import (
    _interval,
    _masked_frame_indices,
    _native_frames,
    _require_fixtures,
)
from test_visual_prompts import make_video

# The fixture facts this module leans on, so a regenerated clip fails loudly here.
MASKED_FRAMES = 49  # of the clip's 60 native frames, inside the selection's 0.0-6.0 s interval
NARROWED = (1.0, 3.0)

pytest.importorskip("torch", reason="prepare_masked_video builds torch tensors")
pytestmark = [
    pytest.mark.skipif(not (REPO / "app" / "main.py").exists(), reason="Person 4 backend not present"),
    pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg/ffprobe are required"),
]


def _install_recorders(patch) -> dict[str, list]:
    """Record the three boundaries this module asserts on: translation, engine, backend."""
    from sam_audio_service.backends.mock import MockBackend
    from sam_audio_service.engine import Engine

    records: dict[str, list] = {"translations": [], "engine_results": [], "model_inputs": []}

    real_translate = Engine.translate_visual

    def translate(self, prompt, shared, metadata, source_duration, warnings, workdir=None):
        before = len(warnings)
        visual, anchors = real_translate(self, prompt, shared, metadata, source_duration, warnings, workdir)
        records["translations"].append({
            "metadata": metadata,
            "visual": visual,
            "anchors": list(anchors),
            "warnings": list(warnings[before:]),
        })
        return visual, anchors

    real_separate = Engine.separate

    def separate(self, audio, metadata, workdir=None):
        result = real_separate(self, audio, metadata, workdir)
        records["engine_results"].append(result)
        return result

    real_backend = MockBackend.separate

    def backend(self, mixture, prompt, options):
        """With ``AUDIO_BACKEND=sam`` this is where ``prepare_masked_video`` runs and the
        frames and masks become processor tensors, so a ``BackendPrompt.visual`` arriving
        here is the proof that the merged workflow reaches SAM Audio's visual inputs."""
        records["model_inputs"].append({"frames": int(mixture.shape[-1]), "prompt": prompt, "sample_rate": self.sample_rate})
        return real_backend(self, mixture, prompt, options)

    patch.setattr(Engine, "translate_visual", translate)
    patch.setattr(Engine, "separate", separate)
    patch.setattr(MockBackend, "separate", backend)
    return records


@pytest.fixture
def recorders(monkeypatch) -> dict[str, list]:
    return _install_recorders(monkeypatch)


@pytest.fixture(scope="module")
def artefacts() -> tuple[dict, dict, Path]:
    """Person 2's real detections/selection documents and the clip they analysed."""
    from sam_audio_service.visual import verify_sha256

    detections, selection, video = _require_fixtures()
    verify_sha256(str(video), detections["source_sha256"])  # raises InvalidMedia on mismatch
    assert selection["source_sha256"] == detections["source_sha256"]
    return detections, selection, video


@pytest.fixture(scope="module")
def pedestrians_project(backend_client, artefacts) -> tuple[str, dict, dict]:
    """The state just before the click: clip uploaded, Person 2's detections stored.

    Done once for the module — the uploads and the ffmpeg audio extraction are the
    expensive part; each test below only adds a ``/process`` run on top.
    """
    detections, _selection, video = artefacts
    client, _core = backend_client
    project = client.post("/api/projects").json()
    pid = project["project_id"]
    headers = {"Authorization": "Bearer " + project["access_token"]}
    with video.open("rb") as handle:
        upload = client.post(f"/api/projects/{pid}/upload", headers=headers,
                             files={"file": (video.name, handle, "video/mp4")})
    assert upload.status_code == 200, upload.text
    media = upload.json()["metadata"]
    assert media["source_sha256"] == detections["source_sha256"], "the backend stored other bytes than Person 2 analysed"
    assert (media["width"], media["height"]) == (detections["width"], detections["height"])
    assert media["frame_rate"] == pytest.approx(detections["fps"])

    # Person 2's detections document, stored by the backend exactly as Person 3 would PUT it.
    saved = client.put(f"/api/projects/{pid}/visual-metadata", headers=headers, json=detections)
    assert saved.status_code == 200, saved.text
    return pid, headers, media


def _process(client, pid: str, headers: dict, body: dict) -> dict:
    started = client.post(f"/api/projects/{pid}/process", headers=headers, json=body)
    assert started.status_code == 202, started.text
    job = wait_for_job(client, "/api/jobs/" + started.json()["job_id"], headers)
    assert job["status"] == "completed", job
    return {"job": job, "job_id": started.json()["job_id"]}


@pytest.fixture(scope="module")
def selection_run(backend_client, pedestrians_project, artefacts) -> dict:
    """The workflow itself, run once: click the tracked person, then press process.

    The recorders are installed only for the duration of the run, so the function-scoped
    ``recorders`` fixture used by the other tests never stacks on top of them.
    """
    _detections, selection, _video = artefacts
    client, _core = backend_client
    pid, headers, _media = pedestrians_project
    with pytest.MonkeyPatch.context() as patch:
        records = _install_recorders(patch)
        run = _process(client, pid, headers, {"prompts": ["person"], "visual_prompt": selection})
    assert len(records["translations"]) == 1, records["translations"]
    return {**run, **records}


# ----------------------------------------------------------------- 1. what we received
def test_merged_workflow_sends_the_original_video_and_both_documents(selection_run, pedestrians_project, backend_client, artefacts):
    """``metadata.source`` points at the project's immutable upload; both documents ride along."""
    detections, selection, _video = artefacts
    _client, core = backend_client
    pid, _headers, media = pedestrians_project
    metadata = selection_run["translations"][0]["metadata"]
    stored_video = next((core.project_dir(pid) / "original").iterdir())

    assert metadata.source is not None, "backend sent no metadata.source; masks cannot be used"
    assert metadata.source.original_ref == stored_video.resolve().as_uri()
    assert metadata.source.sha256 == detections["source_sha256"]
    assert metadata.source.asset_id == pid
    assert metadata.source.duration == pytest.approx(media["duration"])
    assert metadata.source.frame_rate == pytest.approx(detections["fps"])
    assert metadata.asset_id == pid
    assert metadata.request_id == selection_run["job_id"]
    # The exact merged envelope: both Person 2 documents under metadata.visual_prompt.
    assert set(metadata.visual_prompt) == {"selection", "detections"}
    assert metadata.visual_prompt["selection"] == selection
    assert metadata.visual_prompt["detections"] == detections


# ------------------------------------------------------------ 2. a VisualInput, not anchors
def test_merged_workflow_builds_a_visual_input_with_per_frame_masks(selection_run, pedestrians_project, backend_client, artefacts):
    detections, selection, _video = artefacts
    _client, core = backend_client
    pid, _headers, _media = pedestrians_project
    record = selection_run["translations"][0]
    visual = record["visual"]
    stored_video = next((core.project_dir(pid) / "original").iterdir())

    assert visual is not None, record["warnings"]
    assert not record["warnings"], record["warnings"]
    assert visual.interval == pytest.approx(_interval(selection))
    assert Path(visual.video_path) == stored_video.resolve()
    assert visual.source_sha256 == detections["source_sha256"]
    assert visual.frame_size == (detections["width"], detections["height"])
    assert visual.frame_rate == pytest.approx(detections["fps"])
    assert visual.object_id == f"track:{selection['track_id']}"
    assert visual.label == selection["label"]
    assert record["anchors"] == []  # visible presence is not audible presence
    assert set(visual.frame_masks) == _masked_frame_indices(detections, selection)
    assert len(visual.frame_masks) == MASKED_FRAMES


# ------------------------------------------- 3. the model-input boundary itself got it
def test_merged_workflow_reaches_the_model_input_boundary(selection_run, artefacts):
    from sam_audio_service.visual import prepare_masked_video

    detections, selection, _video = artefacts
    visual = selection_run["translations"][0]["visual"]
    assert len(selection_run["model_inputs"]) == 1, selection_run["model_inputs"]
    call = selection_run["model_inputs"][0]
    assert call["prompt"].visual is visual, "the engine dropped the VisualInput before the backend"
    assert call["prompt"].text == "person"
    # Visually prompted requests run the interval as one window, never chunked. The sample
    # bounds are ``Engine._separate_interval``'s, not (end - start) * sr.
    start, end = _interval(selection)
    sample_rate = call["sample_rate"]
    assert call["frames"] == round(end * sample_rate) - round(start * sample_rate)

    # The file:// reference really resolves inside this service: decode frames + masks.
    # ``prepare_masked_video`` raises InvalidMedia when no mask survives, so reaching the
    # shape assertions already proves object pixels made it through the handoff.
    frames, masks = prepare_masked_video(visual, size=64, mask_mode="keep_object")
    native = _native_frames(selection, float(detections["fps"]))
    assert frames.shape == (native, 3, 64, 64)
    assert masks.shape == (native, 1, 64, 64)


# ---------------------------------------------------------------- 4. the stems we sent
def test_merged_workflow_returns_full_timeline_stems(selection_run, pedestrians_project, backend_client, artefacts):
    from sam_audio_service.audio_io import decode

    _detections, selection, _video = artefacts
    client, core = backend_client
    pid, headers, _media = pedestrians_project
    job = selection_run["job"]

    inference = job["inference"]
    assert inference["model"]["is_mock"] is True
    assert inference["source"]["asset_id"] == pid
    assert inference["request_id"] == selection_run["job_id"]
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

    # The target's spans land on the selected interval of the project timeline. The mock
    # reports energy-based spans whenever a '+' anchor is present, so the engine's
    # "fall back to the interval" branch does not run here; what is ours to assert is that
    # every span it does report was mapped back into the selected window.
    produced = {stem.name: stem for stem in selection_run["engine_results"][0].stems}
    start, end = _interval(selection)
    assert produced["person"].spans
    assert all(start - 1e-3 <= s <= e <= end + 1e-3 for s, e in produced["person"].spans), produced["person"].spans
    assert produced["residual"].spans is None
    assert produced["person"].samples.shape == audio.samples.shape


def test_merged_workflow_narrows_to_the_selected_subinterval(backend_client, pedestrians_project, recorders, artefacts):
    """A click on a 1.0-3.0 s window must crop the frames *and* rebase the spans onto it."""
    from sam_audio_service.audio_io import decode
    from sam_audio_service.visual import _frame_index_range

    detections, selection, _video = artefacts
    client, core = backend_client
    pid, headers, _media = pedestrians_project
    narrowed = {**selection, "interval": {"start_seconds": NARROWED[0], "end_seconds": NARROWED[1]}}
    job = _process(client, pid, headers, {"prompts": ["person"], "visual_prompt": narrowed})["job"]

    visual = recorders["translations"][0]["visual"]
    assert visual is not None, recorders["translations"]
    assert visual.interval == pytest.approx(NARROWED)
    # Only the native frames of the narrowed interval are handed to the model.
    first, last_exclusive = _frame_index_range(*NARROWED, float(detections["fps"]))
    assert set(visual.frame_masks) == _masked_frame_indices(detections, narrowed)
    assert set(visual.frame_masks) <= set(range(first, last_exclusive))

    # ``Engine._separate_interval`` rebases the backend's spans by s0/sr; this is the only
    # case where a missing offset would show, since the full-clip interval starts at 0.
    produced = {stem.name: stem for stem in recorders["engine_results"][0].stems}
    assert produced["person"].spans
    assert all(NARROWED[0] - 1e-3 <= s <= e <= NARROWED[1] + 1e-3 for s, e in produced["person"].spans), produced["person"].spans

    audio = decode(data=(core.project_dir(pid) / "extracted" / "audio.wav").read_bytes())
    target = decode(data=client.get(job["stems"][0]["audio_url"], headers=headers).content).samples
    assert target.shape == audio.samples.shape
    assert job["inference"]["source"]["duration"] == pytest.approx(audio.duration, abs=1e-3)


def test_backend_rejects_selection_whose_bytes_changed(backend_client, artefacts, tmp_path):
    """The merged workflow's identity guard: Person 4 refuses detections for other bytes."""
    detections, selection, _video = artefacts
    client, _core = backend_client
    # Any other bytes will do: ``app/visual.py::validate_selection`` calls
    # ``validate_identity`` (the source_sha256 comparison) before it ever compares
    # dimensions or FPS, so a 1 s 64x64 clip is enough — and cheap.
    other = make_video(tmp_path / "other.mp4", seconds=1.0, fps=10, size="64x64", audio=True)
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


@pytest.fixture(scope="module")
def no_selection_run(backend_client, pedestrians_project) -> dict:
    """Process with stored detections but *no* click, recorded once for the two tests below."""
    client, _core = backend_client
    pid, headers, _media = pedestrians_project
    with pytest.MonkeyPatch.context() as patch:
        records = _install_recorders(patch)
        run = _process(client, pid, headers, {"prompts": ["person"]})
    assert len(records["translations"]) == 1, records["translations"]
    return {**run, **records}


def test_process_without_a_selection_runs_text_only(no_selection_run):
    """No click, but stored detections: we can only run text, with no anchors, and we say so.

    ``prepare_payload`` takes its ``selection is None`` branch and attaches **no**
    ``metadata.source``. With no selected object and no video reference there is nothing to
    prompt with, so the request still succeeds with a warning — waste, not breakage.
    """
    record = no_selection_run["translations"][0]
    assert record["metadata"].source is None, "no selection, so the backend sends no source block"
    assert record["visual"] is None, "without a selected object there is nothing to prompt with"
    assert record["anchors"] == []
    assert any("without a selected object" in w for w in record["warnings"]), record["warnings"]
    warnings = no_selection_run["job"]["inference"]["warnings"]
    assert any("without a selected object" in w for w in warnings), warnings


@pytest.mark.xfail(strict=True, reason="Person 4 still forwards detections without a selection")
def test_process_without_a_selection_should_not_forward_the_detections_document(no_selection_run, artefacts):
    """The waste, pinned as a wish rather than as a fact.

    ``app/visual.py`` currently does ``payload['visual_prompt'] = raw``, so the whole
    detections document (every frame's RLE) crosses the wire for a request that cannot use
    any of it. Pinning the *current* behaviour would make Person 4's fix break CI; pinning
    the behaviour we want makes it flip this xfail to XPASS instead.
    """
    detections, _selection, _video = artefacts
    forwarded = no_selection_run["translations"][0]["metadata"].visual_prompt
    assert set(forwarded or {}) != set(detections), "the raw detections document is still forwarded"
