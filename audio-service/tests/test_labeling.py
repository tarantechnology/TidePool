"""Instrument tags and descriptive names for music stems (labeling.py), with a fake tagger."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sam_audio_service import labeling as L
from sam_audio_service.api import create_app
from sam_audio_service.classifier import ContentClassifier
from sam_audio_service.config import load_settings
from sam_audio_service.engine import Engine

from test_music_stems_api import post, song_wav

SR = 48_000
NAMES = ["Speech", "Music", *L.DISPLAY]


def probs(**scores: float) -> np.ndarray:
    p = np.zeros(len(NAMES))
    for name, score in scores.items():
        p[NAMES.index(name.replace("_", " "))] = score
    return p


# What the fake "hears" in each mock-splitter band (bass < 150 Hz < vocals < 3.5 kHz < other < 8 kHz < drums).
BAND_PROBS = [
    (150, probs(**{"Bass guitar": 0.7, "Synthesizer": 0.2})),
    (3500, probs(**{"Singing": 0.8, "Female singing": 0.6, "Male singing": 0.1})),
    (8000, probs(**{"Synthesizer": 0.5, "String section": 0.35, "Bowed string instrument": 0.2, "Musical instrument": 0.9, "Female singing": 0.4})),
    (np.inf, probs(**{"Drum": 0.7, "Drum kit": 0.6, "Drum machine": 0.3})),
]


class FakeTagger(ContentClassifier):
    """Says "music"; tags each clip by its loudest frequency (see BAND_PROBS)."""

    name = "fake-tagger"
    can_tag = True

    def __init__(self, fail: bool = False):
        self.fail, self.calls = fail, 0

    def score(self, samples, sample_rate):
        return {"music": 1.0, "speech": 0.0}

    def class_names(self):
        return list(NAMES)

    def tag(self, clips, sample_rate):
        self.calls += 1
        if self.fail:
            raise RuntimeError("tagger exploded")
        out = []
        for clip in clips:
            peak = np.argmax(np.abs(np.fft.rfft(clip))) * sample_rate / len(clip)
            out.append(next(p for limit, p in BAND_PROBS if peak < limit))
        return np.stack(out)


# --------------------------------------------------------------------- pure functions
def test_whitelist_merges_classes_with_the_same_display_name():
    classes = L.whitelist(NAMES)
    assert "Speech" not in classes and "Music" not in classes
    assert [NAMES[i] for i in classes["Strings"]] == ["Bowed string instrument", "String section", "Pizzicato"]
    assert L.candidates(probs(**{"String section": 0.3, "Pizzicato": 0.5}), classes, 0.1) == [("Strings", 0.5)]


def test_candidates_drop_generic_parents_and_resolve_parent_child_pairs():
    classes = L.whitelist(NAMES)
    # Generic "Instruments"/"Keys" only survive when nothing specific clears the threshold.
    assert L.candidates(probs(**{"Musical instrument": 0.9, "Keyboard (musical)": 0.5, "Synthesizer": 0.2}), classes, 0.1) == [("Synth", 0.2)]
    assert L.candidates(probs(**{"Musical instrument": 0.9, "Keyboard (musical)": 0.5, "Synthesizer": 0.05}), classes, 0.1) == [("Instruments", 0.9), ("Keys", 0.5)]
    # A confident child replaces its parent; an unsure child is dropped in favour of it.
    assert L.candidates(probs(Guitar=0.6, **{"Electric guitar": 0.4}), classes, 0.1) == [("Electric guitar", 0.4)]
    assert L.candidates(probs(Guitar=0.6, **{"Electric guitar": 0.15}), classes, 0.1) == [("Guitar", 0.6)]
    assert L.candidates(probs(Piano=0.05), classes, 0.1) == []


def test_describe_uses_top_tag_or_a_pair():
    assert L.describe([("Synth", 0.5), ("Strings", 0.31)]) == "Synth & Strings"
    assert L.describe([("Synth", 0.5), ("Strings", 0.29)]) == "Synth"
    assert L.describe([("Instruments", 0.9), ("Keys", 0.2)]) == "Keys"
    assert L.describe([("Instruments", 0.9)]) is None
    assert L.describe([]) is None


def test_active_windows_skip_silence_cap_the_count_and_normalise_level():
    t = np.arange(10 * SR) / SR
    loud, quiet = np.sin(2 * np.pi * 440 * t) * 0.01, np.zeros(10 * SR)
    stem = np.concatenate([loud, quiet, loud, quiet] * 5).astype(np.float32)[None]  # 200 s, active half the time
    _, mix = L.window_energies(stem[0] * 0 + 0.01, 10 * SR)
    clips = L.active_windows(stem, mix, SR, max_windows=4)
    assert len(clips) == 4 and all(len(c) == 10 * SR for c in clips)
    assert all(abs(20 * np.log10(np.sqrt(np.mean(c.astype(np.float64) ** 2))) - L.TARGET_RMS_DB) < 0.1 for c in clips)
    # A stem far below the mixture in every window still yields its loudest window.
    assert len(L.active_windows(stem * 1e-4, mix, SR, max_windows=4)) == 1
    assert L.active_windows(np.zeros((2, SR), np.float32), mix, SR, 4) == []


def bands(seconds=12.0):
    t = np.arange(int(seconds * SR)) / SR
    tone = lambda f, a: (a * np.sin(2 * np.pi * f * t)).astype(np.float32)[None].repeat(2, 0)  # noqa: E731
    return {"bass": tone(80, 0.3), "vocals": tone(440, 0.3), "other": tone(5000, 0.1), "drums": tone(10000, 0.05)}


def test_label_stems_tags_names_and_trivial_classes():
    stems = bands()
    mix = sum(stems.values())
    tagger = FakeTagger()
    labels = dict(zip(stems, L.label_stems(tagger, list(stems.items()), mix, SR, min_score=0.1)))
    assert tagger.calls == 1  # every window in one call
    assert labels["other"].name == "Synth & Strings"
    assert labels["other"].tags == [{"label": "Synth", "score": 0.5}, {"label": "Strings", "score": 0.35}]  # no "Instruments", no sung bleed
    assert labels["vocals"].tags == [{"label": "Female singing", "score": 0.6}]
    assert labels["drums"].tags == [{"label": "Drum kit", "score": 0.6}, {"label": "Drum machine", "score": 0.3}]
    assert labels["bass"].tags == [{"label": "Bass guitar", "score": 0.7}, {"label": "Synth", "score": 0.2}]
    assert all(len(lb.tags) <= L.MAX_TAGS for lb in labels.values())
    high = dict(zip(stems, L.label_stems(tagger, list(stems.items()), mix, SR, min_score=0.55)))
    assert high["other"].name is None and high["other"].tags == []


def test_near_silent_stem_is_not_tagged():
    stems = bands()
    stems["other"] = stems["other"] * 1e-3  # -72 dB vs the mix
    labels = L.label_stems(FakeTagger(), list(stems.items()), sum(stems.values()), SR, min_score=0.1)
    other = labels[list(stems).index("other")]
    assert other.name is None and other.tags == [] and other.windows == 0 and other.energy_db < L.SILENT_DB


# --------------------------------------------------------------------- engine + API
@pytest.fixture
def tagged_settings(settings, monkeypatch):
    monkeypatch.setenv("AUDIO_STEM_SPLITTER", "mock")
    return load_settings()


def client_for(settings, tagger):
    return TestClient(create_app(settings, Engine(settings, classifier=tagger)))


def test_full_split_names_other_and_tags_every_stem(tagged_settings):
    with client_for(tagged_settings, FakeTagger()) as c:
        body = post(c, {"prompts": []}).json()
    stems = {s["label"]: s for s in body["stems"]}
    assert [s["name"] for s in body["stems"]] == ["Vocals", "Drums", "Bass", "Synth & Strings"]
    assert stems["other"]["label"] == "other"
    assert [t["label"] for t in stems["other"]["tags"]] == ["Synth", "Strings"]
    assert stems["vocals"]["tags"] == [{"label": "Female singing", "score": 0.6}]
    assert stems["bass"]["tags"][0] == {"label": "Bass guitar", "score": 0.7}
    assert not any("labeling" in w for w in body["warnings"])


def test_caller_wording_is_kept_but_options_stems_names_are_replaced(tagged_settings):
    with client_for(tagged_settings, FakeTagger()) as c:
        worded = post(c, {"prompts": ["other instruments", "vocals"]}).json()
        listed = post(c, {"options": {"mode": "music_stems", "stems": ["other", "bass"], "residual_name": "the rest"}}).json()
    assert [(s["name"], s["label"]) for s in worded["stems"]][0] == ("other instruments", "other")
    assert worded["stems"][0]["tags"][0]["label"] == "Synth"
    assert [(s["name"], s["role"]) for s in listed["stems"]] == [("Synth & Strings", "target"), ("Bass", "target"), ("the rest", "residual")]
    # The residual (vocals + drums here; the fake hears its loudest part) is tagged too, singing included.
    assert listed["stems"][2]["tags"] == [{"label": "Female singing", "score": 0.6}]


def test_labeling_failure_never_fails_the_split(tagged_settings):
    with client_for(tagged_settings, FakeTagger(fail=True)) as c:
        r = post(c, {"prompts": []})
    assert r.status_code == 200
    body = r.json()
    assert [s["name"] for s in body["stems"]][-1] == "Other"
    assert all(s["tags"] is None for s in body["stems"])
    assert any("stem labeling failed" in w for w in body["warnings"])


def test_labels_off_or_no_tagger_gives_null_tags(tagged_settings, settings, monkeypatch):
    with TestClient(create_app(tagged_settings)) as c:  # mock classifier cannot tag
        body = post(c, {"prompts": []}).json()
    assert [s["name"] for s in body["stems"]][-1] == "Other" and all("tags" in s and s["tags"] is None for s in body["stems"])
    monkeypatch.setenv("AUDIO_STEM_LABELS", "off")
    tagger = FakeTagger()
    with client_for(load_settings(), tagger) as c:
        body = post(c, {"prompts": []}).json()
    assert tagger.calls == 0 and [s["name"] for s in body["stems"]][-1] == "Other"


def test_silent_other_falls_back_to_other(tagged_settings):
    t = np.arange(SR) / SR
    from sam_audio_service.audio_io import encode_wav_pcm16

    mono = 0.3 * np.sin(2 * np.pi * 80 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)  # nothing in the "other" band
    with client_for(tagged_settings, FakeTagger()) as c:
        body = post(c, {"prompts": []}, audio=encode_wav_pcm16(np.stack([mono, mono]).astype(np.float32), SR)).json()
    other = body["stems"][-1]
    assert other["name"] == "Other" and other["tags"] == []


def test_prompted_results_have_null_tags(settings):
    with TestClient(create_app(settings, Engine(settings, classifier=FakeTagger()))) as c:
        r = c.post("/separate", files={"audio": ("a.wav", song_wav(), "audio/wav")}, data={"metadata": json.dumps({"prompts": ["dog barking"]})})
    assert r.status_code == 200, r.text
    assert all(s["tags"] is None for s in r.json()["stems"])


def test_label_settings_validate(monkeypatch):
    monkeypatch.setenv("AUDIO_STEM_LABELS", "sometimes")
    with pytest.raises(ValueError, match="AUDIO_STEM_LABELS"):
        load_settings()
    monkeypatch.setenv("AUDIO_STEM_LABELS", "auto")
    monkeypatch.setenv("AUDIO_STEM_LABEL_MIN_SCORE", "1.5")
    with pytest.raises(ValueError, match="AUDIO_STEM_LABEL_MIN_SCORE"):
        load_settings()
    monkeypatch.setenv("AUDIO_STEM_LABEL_MIN_SCORE", "0.2")
    assert load_settings().stem_label_min_score == 0.2


def test_audioset_tagger_tag_resamples_pads_and_batches():
    from sam_audio_service.classifier import _AudioSetTagger

    class Shapes(_AudioSetTagger):
        rate = 16_000

        def load(self):
            self.model = object()

        def _class_names(self):
            return ["a", "b"]

        def probs(self, clips):
            self.seen.append([len(c) for c in clips])
            return np.array([[float(np.sqrt(np.mean(c**2))), 0.0] for c in clips])

    tagger = Shapes(window_seconds=1.0)
    tagger.seen, tagger.tag_batch = [], 2
    t = np.arange(SR) / SR
    clips = [np.sin(2 * np.pi * 440 * t).astype(np.float32)] * 3 + [np.ones(SR // 2, np.float32) * 0.5]
    out = tagger.tag(clips, SR)
    assert tagger.seen == [[16_000, 16_000], [16_000, 16_000]]
    assert out.shape == (4, 2) and abs(out[0, 0] - np.sqrt(0.5)) < 0.01  # level survives resampling
    assert tagger.tag([], SR).shape == (0, 2) and tagger.can_tag
