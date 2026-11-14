"""Router decision table (routing.route), exercised without audio or models."""
import pytest

from sam_audio_service.contracts import SeparationMetadata
from sam_audio_service.errors import InvalidRequest, UnsupportedOperation
from sam_audio_service.routing import route

FOUR = ("drums", "bass", "other", "vocals")
MUSIC = {"music": 0.9, "speech": 0.05}
SPEECH = {"music": 0.1, "speech": 0.95}


def decide(meta: dict, *, scores=None, available=FOUR, default_mode="auto", threshold=0.5):
    metadata = SeparationMetadata.model_validate(meta)
    calls = []

    def classify():
        calls.append(1)
        return scores

    decision = route(
        metadata.resolved_prompts(8), metadata.options, metadata.hints,
        available=available, classify=classify, default_mode=default_mode, threshold=threshold,
        prompted_engine=("audiosep", "audiosep-base-4M"), music_engine=("demucs", "htdemucs_ft") if available else None,
    )
    assert len(calls) <= 1  # the classifier runs at most once
    return decision, len(calls)


@pytest.mark.parametrize("meta,scores,mode,reason", [
    # explicit mode wins
    ({"prompts": ["vocals"], "options": {"mode": "prompted"}}, MUSIC, "prompted", "options.mode=prompted"),
    ({"prompts": ["vocals"], "options": {"mode": "music_stems"}}, SPEECH, "music_stems", "options.mode=music_stems"),
    ({"prompts": [], "options": {"mode": "music_stems"}}, None, "music_stems", "options.mode=music_stems"),
    # auto
    ({"prompts": []}, None, "music_stems", "no prompts"),
    ({"prompts": []}, SPEECH, "music_stems", "no prompts"),
    ({"prompts": ["vocals", "drums"]}, MUSIC, "music_stems", "content is music"),
    ({"prompts": ["vocals"]}, SPEECH, "prompted", "not a song"),
    ({"prompts": ["singing"]}, None, "prompted", "classifier is unavailable"),
    ({"prompts": ["singing"], "hints": {"source_kind": "video"}}, None, "prompted", "classifier is unavailable"),
    ({"prompts": ["singing"], "hints": {"source_kind": "audio"}}, None, "music_stems", "audio-only"),
    ({"prompts": ["dog barking"]}, MUSIC, "prompted", "not music stems"),
    ({"prompts": ["vocals", "dog barking"]}, MUSIC, "prompted", "mixed prompts"),
    ({"prompts": ["guitar"]}, MUSIC, "prompted", "not music stems"),  # a 4-stem splitter has no guitar
    ({"prompts": ["instrumental", "drums"]}, MUSIC, "prompted", "overlaps"),
    ({"prompts": [{"text": "vocals", "anchors": [{"start": 0, "end": 1}]}]}, MUSIC, "prompted", "span or visual guidance"),
    ({"options": {"stems": ["bass"]}}, MUSIC, "music_stems", "content is music"),
    ({"prompts": ["vocals"], "options": {"mode": "music_stems", "stems": ["drums"]}}, None, "music_stems", "options.mode"),
])
def test_decision_table(meta, scores, mode, reason):
    decision, _ = decide(meta, scores=scores)
    assert decision.mode == mode
    assert reason in decision.reason
    assert decision.engine == ("demucs" if mode == "music_stems" else "audiosep")


def test_default_mode_comes_from_settings():
    decision, _ = decide({"prompts": ["vocals"]}, default_mode="prompted", scores=MUSIC)
    assert decision.mode == "prompted" and decision.reason == "AUDIO_ROUTING_DEFAULT=prompted"


def test_auto_without_splitter_is_prompted_and_explicit_music_stems_is_unsupported():
    decision, calls = decide({"prompts": []}, available=None)
    assert decision.mode == "prompted" and calls == 0
    with pytest.raises(UnsupportedOperation, match="AUDIO_STEM_SPLITTER"):
        decide({"prompts": [], "options": {"mode": "music_stems"}}, available=None)


def test_unknown_stem_lists_valid_names():
    with pytest.raises(InvalidRequest) as exc:
        decide({"prompts": ["vocals", "dog"], "options": {"mode": "music_stems"}})
    assert exc.value.details["unknown"] == ["dog"]
    assert exc.value.details["valid_stems"] == ["vocals", "drums", "bass", "other", "instrumental"]
    with pytest.raises(InvalidRequest, match="overlaps"):
        decide({"options": {"mode": "music_stems", "stems": ["instrumental", "bass"]}})


def test_classifier_only_runs_when_a_row_needs_it():
    assert decide({"prompts": ["dog"]}, scores=MUSIC)[1] == 0
    assert decide({"prompts": ["vocals"], "options": {"mode": "music_stems"}}, scores=MUSIC)[1] == 0
    assert decide({"prompts": ["vocals"]}, scores=MUSIC)[1] == 1


def test_music_threshold_is_inclusive():
    assert decide({"prompts": ["vocals"]}, scores={"music": 0.5, "speech": 0.0})[0].mode == "music_stems"
    assert decide({"prompts": ["vocals"]}, scores={"music": 0.49, "speech": 0.0})[0].mode == "prompted"


def test_stem_names_residual_and_scores():
    decision, _ = decide({"prompts": ["Lead vocal", "the beat"]}, scores=MUSIC)
    assert [(r.name, r.canonical) for r in decision.requests] == [("Lead vocal", "vocals"), ("the beat", "drums")]
    assert decision.residual == ["bass", "other"]
    assert decision.to_dict() == {"mode": "music_stems", "engine": "demucs", "model_id": "htdemucs_ft", "reason": decision.reason, "scores": MUSIC, "stems": ["vocals", "drums"]}
    full, _ = decide({"prompts": []})
    assert [r.name for r in full.requests] == ["Vocals", "Drums", "Bass", "Other"] and full.residual == []
    karaoke, _ = decide({"options": {"mode": "music_stems", "stems": ["vocals", "karaoke"]}})
    assert [r.name for r in karaoke.requests] == ["Vocals", "Instrumental"] and karaoke.residual == []


def test_warnings():
    not_music, _ = decide({"prompts": []}, scores=SPEECH)
    assert any("does not sound like music" in w for w in not_music.warnings)
    repeated, _ = decide({"prompts": ["vocals", "singing"], "options": {"mode": "music_stems"}})
    assert [r.name for r in repeated.requests] == ["vocals"] and any("returned once" in w for w in repeated.warnings)
    ignored, _ = decide({"prompts": ["dog"], "options": {"stems": ["vocals"], "mode": "prompted"}})
    assert any("options.stems" in w for w in ignored.warnings)
    guided, _ = decide({"prompts": [{"text": "vocals", "anchors": [{"start": 0, "end": 1}]}], "options": {"mode": "music_stems"}})
    assert any("ignores span anchors" in w for w in guided.warnings)


# ---------------------------------------------------------------- classifier and config
def test_classifier_helpers():
    import numpy as np

    from sam_audio_service.classifier import resample_fft, windows

    t = np.arange(48_000) / 48_000
    down = resample_fft(np.sin(2 * np.pi * 440 * t), 48_000, 16_000)
    assert len(down) == 16_000
    assert np.argmax(np.abs(np.fft.rfft(down))) == 440  # 1 Hz bins over one second
    long = np.zeros(16_000 * 60)
    clips = windows(long, 160_000, 3)
    assert len(clips) == 3 and all(len(c) == 160_000 for c in clips)
    assert len(windows(np.zeros(1000), 160_000, 3)) == 1
    assert len(windows(np.zeros(16_000 * 240), 160_000, 8)) == 8  # a 4 min song: 8 windows
    assert len(windows(np.zeros(16_000 * 25), 160_000, 8)) == 3  # no near-duplicate windows
    assert len(windows(np.zeros(16_000 * 13), 160_000, 8)) == 2


def test_song_score_is_median_music_minus_speech():
    from sam_audio_service.classifier import decision_score, song_score

    # a song: music on every window, sung vocals barely register as speech
    assert song_score([0.6, 0.55, 0.7, 0.5], [0.05, 0.1, 0.2, 0.05]) == pytest.approx(0.475)
    # speech over a music bed (vlog): AST says "music" just as loudly, but also "speech"
    assert song_score([0.65, 0.6, 0.7], [0.7, 0.65, 0.6]) == 0.0
    # a spoken intro (one window of eight) does not sink a song
    assert song_score([0.05] + [0.6] * 7, [0.8] + [0.05] * 7) == pytest.approx(0.55)
    assert song_score([], []) == 0.0
    assert decision_score({"music": 0.9, "speech": 0.8, "song": 0.42}) == 0.42  # classifier's own
    assert decision_score({"music": 0.9, "speech": 0.1}) == pytest.approx(0.8)  # fallback
    assert decision_score({"music": 0.2, "speech": 0.8}) == 0.0


def test_speech_over_music_is_not_routed_to_the_splitter():
    from sam_audio_service.config import Settings

    threshold = Settings(backend="mock", router_classifier="panns").router_music_threshold
    vlog = {"music": 0.88, "speech": 0.62, "song": 0.03}  # what PANNs says about a vlog
    song = {"music": 0.85, "speech": 0.04, "song": 0.82}  # ... and about a 4 min song
    decision, _ = decide({"prompts": ["vocals"]}, scores=vlog, threshold=threshold)
    assert decision.mode == "prompted" and "not a song" in decision.reason and "song score 0.03" in decision.reason
    decision, _ = decide({"prompts": ["singing"]}, scores=song, threshold=threshold)
    assert decision.mode == "music_stems" and "song score 0.82" in decision.reason
    assert song["song"] >= 2 * threshold  # a full song clears the threshold comfortably


def test_ast_score_with_a_fake_model(monkeypatch):
    """AstClassifier.score end to end with a fake torch/model/extractor (no download, no torch)."""
    import contextlib
    import sys
    import types

    import numpy as np

    from sam_audio_service.classifier import AstClassifier

    class T:  # the bits of a torch tensor score() touches
        def __init__(self, a):
            self.a = np.asarray(a, dtype=np.float32)

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext, sigmoid=lambda t: T(1 / (1 + np.exp(-t.a))))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    logit = lambda p: float(np.log(p / (1 - p)))  # noqa: E731
    seen = {}

    class Batch(dict):
        def to(self, device):
            return self

    def extractor(clips, sampling_rate, return_tensors):
        seen["clips"], seen["rate"] = len(clips), sampling_rate
        return Batch(n=len(clips))

    def model(n):
        logits = np.full((n, 3), -10.0)
        logits[:, 0] = logit(0.6)  # Music
        logits[:, 1] = logit(0.1)  # Speech
        logits[0, 1] = logit(0.9)  # a spoken intro in the first window
        return types.SimpleNamespace(logits=T(logits))

    ast = AstClassifier(device="cpu")
    monkeypatch.setattr(ast, "load", lambda: None)
    ast.model, ast.extractor, ast.label_ids = model, extractor, {"music": 0, "speech": 1}
    scores = ast.score(np.zeros((2, 48_000 * 240), dtype=np.float32), 48_000)
    assert seen == {"clips": 8, "rate": 16_000}
    assert scores["song"] == pytest.approx(0.5, abs=1e-3)
    assert scores["music"] == pytest.approx(0.6, abs=1e-3) and scores["speech"] == pytest.approx(0.2, abs=1e-3)


def test_create_classifier_and_defaults(monkeypatch):
    from sam_audio_service.classifier import AstClassifier, MockClassifier, NoClassifier, PannsClassifier, create_classifier
    from sam_audio_service.config import Settings

    monkeypatch.delenv("AUDIO_ROUTER_CLASSIFIER", raising=False)
    assert Settings(backend="mock", stem_splitter="off").router_classifier == "off"
    assert Settings(backend="mock", stem_splitter="mock").router_classifier == "off"
    assert Settings(backend="sam", stem_splitter="demucs").router_classifier == "panns"
    assert Settings(backend="sam", stem_splitter="demucs").router_music_threshold == 0.4
    assert Settings(backend="mock", router_classifier="ast").router_music_threshold == 0.2
    assert Settings(backend="mock", router_classifier="ast", router_music_threshold=0.3).router_music_threshold == 0.3
    panns = create_classifier(Settings(router_classifier="panns"))
    assert isinstance(panns, PannsClassifier) and not panns.loaded and panns.rate == 32_000
    assert isinstance(create_classifier(Settings(router_classifier="off")), NoClassifier)
    assert isinstance(create_classifier(Settings(router_classifier="mock")), MockClassifier)
    ast = create_classifier(Settings(router_classifier="ast"))
    assert isinstance(ast, AstClassifier) and not ast.loaded  # nothing imported until first use


def test_failing_classifier_is_disabled_and_routing_falls_back(settings, monkeypatch):
    import numpy as np

    from sam_audio_service.audio_io import DecodedAudio
    from sam_audio_service.classifier import ContentClassifier
    from sam_audio_service.config import load_settings
    from sam_audio_service.engine import Engine

    class Broken(ContentClassifier):
        name = "ast"
        calls = 0

        def score(self, samples, sample_rate):
            Broken.calls += 1
            raise ImportError("No module named 'transformers'")

    monkeypatch.setenv("AUDIO_STEM_SPLITTER", "mock")
    engine = Engine(load_settings(), classifier=Broken())
    audio = DecodedAudio(np.zeros((2, 4800), dtype=np.float32), 48_000, 48_000, 2)
    meta = SeparationMetadata.model_validate({"prompts": ["vocals"]})
    for _ in range(2):
        decision = engine.route(audio, meta)
        assert decision.mode == "prompted" and "classifier is unavailable" in decision.reason
    assert Broken.calls == 1 and "transformers" in engine.classifier_status()["load_error"]


@pytest.mark.parametrize("env,message", [
    ({"AUDIO_STEM_SPLITTER": "spleeter"}, "AUDIO_STEM_SPLITTER"),
    ({"AUDIO_ROUTING_DEFAULT": "magic"}, "AUDIO_ROUTING_DEFAULT"),
    ({"AUDIO_ROUTING_DEFAULT": "music_stems"}, "needs AUDIO_STEM_SPLITTER"),
    ({"AUDIO_ROUTER_CLASSIFIER": "yamnet"}, "AUDIO_ROUTER_CLASSIFIER"),
    ({"AUDIO_ROUTER_MUSIC_THRESHOLD": "1.5"}, "AUDIO_ROUTER_MUSIC_THRESHOLD"),
    ({"AUDIO_STEM_SHIFTS": "-1"}, "AUDIO_STEM_SHIFTS"),
    ({"AUDIO_STEM_OVERLAP": "1"}, "AUDIO_STEM_OVERLAP"),
    ({"AUDIO_STEM_SEGMENT": "0"}, "AUDIO_STEM_SEGMENT"),
])
def test_music_settings_are_validated(monkeypatch, env, message):
    from sam_audio_service.config import load_settings

    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match=message):
        load_settings()


def test_route_answers_while_a_split_is_running(settings, monkeypatch):
    """/v1/route must not queue behind the separation lock: the classifier has its own."""
    import json
    import threading

    import numpy as np
    from fastapi.testclient import TestClient

    from sam_audio_service.api import create_app
    from sam_audio_service.audio_io import encode_wav_pcm16
    from sam_audio_service.classifier import MockClassifier
    from sam_audio_service.config import load_settings
    from sam_audio_service.engine import Engine
    from sam_audio_service.stems.mock import MockStemSplitter

    started, release = threading.Event(), threading.Event()

    class SlowSplitter(MockStemSplitter):
        def split(self, mixture, progress=None, cancel=None):
            started.set()
            assert release.wait(10), "test did not release the split"
            return super().split(mixture, progress, cancel)

    monkeypatch.setenv("AUDIO_STEM_SPLITTER", "mock")
    music_settings = load_settings()
    engine = Engine(music_settings, splitter=SlowSplitter(), classifier=MockClassifier(music=0.9, speech=0.05))
    wav = encode_wav_pcm16(np.zeros((2, 48_000), dtype=np.float32) + 0.1, 48_000)

    def post(client, path, prompts):
        return client.post(path, files={"audio": ("a.wav", wav, "audio/wav")}, data={"metadata": json.dumps({"prompts": prompts})})

    with TestClient(create_app(music_settings, engine)) as client:
        results = {}
        split = threading.Thread(target=lambda: results.update(split=post(client, "/separate", [])))
        split.start()
        try:
            assert started.wait(10), "the split never started"
            routed = post(client, "/v1/route", ["vocals"])  # the split still holds the separation lock
            assert routed.status_code == 200, routed.text
            assert routed.json()["routing"]["mode"] == "music_stems" and not release.is_set()
        finally:
            release.set()
            split.join(10)
        assert results["split"].status_code == 200
