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
    ({"prompts": ["vocals"]}, SPEECH, "prompted", "not music"),
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


def test_create_classifier_and_defaults(monkeypatch):
    from sam_audio_service.classifier import AstClassifier, MockClassifier, NoClassifier, create_classifier
    from sam_audio_service.config import Settings

    monkeypatch.delenv("AUDIO_ROUTER_CLASSIFIER", raising=False)
    assert Settings(backend="mock", stem_splitter="off").router_classifier == "off"
    assert Settings(backend="mock", stem_splitter="mock").router_classifier == "off"
    assert Settings(backend="sam", stem_splitter="demucs").router_classifier == "ast"
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
    ({"AUDIO_ROUTER_CLASSIFIER": "panns"}, "AUDIO_ROUTER_CLASSIFIER"),
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
