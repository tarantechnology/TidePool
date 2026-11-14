"""Nested split ("Separate the rest", options.mode = music_substems) with fakes: CPU only, no weights."""
import base64
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sam_audio_service.api import create_app
from sam_audio_service.audio_io import decode
from sam_audio_service.classifier import ContentClassifier
from sam_audio_service.config import load_settings
from sam_audio_service.engine import Engine
from sam_audio_service.stems import nested as N

from test_music_stems_api import song_wav

CLASSES = ["Music", "Speech", "Piano", "Electric piano", "Guitar", "Electric guitar", "Organ", "Synthesizer", "Violin, fiddle"]


class FixedTagger(ContentClassifier):
    """Hears the same instruments in every clip (``probs``: AudioSet class -> probability)."""

    name = "fixed-tagger"
    can_tag = True

    def __init__(self, **probs: float):
        self.probs = probs
        self.calls = 0

    def score(self, samples, sample_rate):
        return {"music": 1.0, "speech": 0.0}

    def class_names(self):
        return list(CLASSES)

    def tag(self, clips, sample_rate):
        self.calls += 1
        row = np.array([self.probs.get(c, 0.0) for c in CLASSES])
        return np.stack([row] * len(clips))


class FakeSeparator(N.SubstemSeparator):
    """Returns fixed fractions of the input per output (so energies are predictable)."""

    name = "fake"
    families = {"guitar": "guitar", "piano": "piano", "keys": "piano"}

    def __init__(self, gains=None):
        self.gains = gains or {"guitar": 0.3, "piano": 0.5, "other": 0.2}
        self._loaded = False
        self.load_calls = 0
        self.split_calls = 0

    def load(self):
        self.load_calls += 1
        self._loaded = True

    @property
    def loaded(self):
        return self._loaded

    def info(self):
        return {"model_id": "fake-6s", "device": "cpu", "dtype": "float32", "license": "test"}

    def split(self, mixture, progress=None, cancel=None):
        self.split_calls += 1
        return {name: (g * mixture).astype(np.float32) for name, g in self.gains.items()}


@pytest.fixture
def mock_settings(settings, monkeypatch):
    monkeypatch.setenv("AUDIO_STEM_SPLITTER", "mock")
    return load_settings()


def make_client(settings, separator=None, tagger=None):
    return TestClient(create_app(settings, Engine(settings, classifier=tagger, substems=separator)))


def post(client, options, audio=None):
    meta = {"options": {"mode": "music_substems", **options}}
    return client.post("/separate", files={"audio": ("other.wav", audio or song_wav(), "audio/wav")}, data={"metadata": json.dumps(meta)})


def samples(stem):
    return decode(data=base64.b64decode(stem["audio_base64"])).samples


def mix():
    return decode(data=song_wav()).samples


# --------------------------------------------------------------------- vocabulary
def test_wordings_map_to_families():
    assert N.canonical_substem("Rhodes") == "keys"
    assert N.canonical_substem("piano") == "piano"
    assert N.canonical_substem(" Acoustic Guitar ") == "guitar"
    assert N.canonical_substem("sax") == "woodwinds"
    assert N.canonical_substem("kazoo") is None


def test_pick_detected_keeps_present_offered_families_once_per_model_output():
    from sam_audio_service.labeling import Instrument

    families = {"guitar": "guitar", "piano": "piano", "keys": "piano"}
    found = [Instrument("strings", "Strings", 0.9, True), Instrument("keys", "Electric piano", 0.5, True),
             Instrument("piano", "Piano", 0.3, True), Instrument("guitar", "Guitar", 0.2, True), Instrument("organ", "Organ", 0.4, False)]
    # strings not offered; piano shares keys' output; organ not present.
    assert N.pick_detected(found, families, 4) == [("keys", "Electric piano"), ("guitar", "Guitar")]
    assert N.pick_detected(found, families, 1) == [("keys", "Electric piano")]
    assert N.pick_detected([Instrument("guitar", "Guitar", 0.05, False)], families, 4) == []


# --------------------------------------------------------------------- config
def test_substem_model_follows_the_stem_splitter(settings, monkeypatch):
    assert settings.substem_model == "off"
    monkeypatch.setenv("AUDIO_STEM_SPLITTER", "mock")
    assert load_settings().substem_model == "mock"
    monkeypatch.setenv("AUDIO_STEM_SPLITTER", "demucs")
    s = load_settings()
    assert s.substem_model == "htdemucs_6s" and s.router_classifier == "panns"
    monkeypatch.setenv("AUDIO_STEM_SPLITTER", "off")
    monkeypatch.setenv("AUDIO_SUBSTEM_MODEL", "htdemucs_6s")
    assert load_settings().router_classifier == "panns"  # detection needs the tagger


@pytest.mark.parametrize("name,value", [("AUDIO_SUBSTEM_MODEL", "banquet"), ("AUDIO_SUBSTEM_MAX", "0"), ("AUDIO_SUBSTEM_MAX", "9"),
                                        ("AUDIO_SUBSTEM_MIN_SCORE", "1.5"), ("AUDIO_SUBSTEM_MIN_SCORE", "-0.1"), ("AUDIO_SUBSTEM_SILENT_DB", "5"), ("AUDIO_SUBSTEM_PARENTS", " , ")])
def test_invalid_substem_settings_are_rejected(settings, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        load_settings()


def test_real_separator_is_created_lazily(settings, monkeypatch):
    monkeypatch.setenv("AUDIO_SUBSTEM_MODEL", "htdemucs_6s")
    engine = Engine(load_settings())
    assert isinstance(engine.substems, N.DemucsSubstemSeparator) and not engine.substems.loaded
    assert engine.substems.offered == ("guitar", "piano", "keys")
    assert engine.substems_status()["loaded"] is False


# --------------------------------------------------------------------- HTTP, mock separator
def test_requested_instruments_and_rest_sum_to_the_input(mock_settings):
    with TestClient(create_app(mock_settings)) as client:
        r = post(client, {"parent_label": "other", "stems": ["piano", "guitar"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [(s["name"], s["role"], s["label"]) for s in body["stems"]] == [("Piano", "target", "piano"), ("Guitar", "target", "guitar"), ("Rest", "residual", "rest")]
    assert body["routing"]["mode"] == "music_substems" and body["routing"]["engine"] == "mock" and body["routing"]["stems"] == ["piano", "guitar"]
    assert any("MOCK nested splitter" in w for w in body["warnings"])
    total = sum(samples(s) for s in body["stems"])
    assert np.max(np.abs(total - mix())) < 1e-4


def test_capabilities_advertise_parents_and_instruments(mock_settings):
    with TestClient(create_app(mock_settings)) as client:
        caps = client.get("/capabilities").json()["music_substems"]
        health = client.get("/health").json()["substems"]
    assert caps["available"] and caps["model_id"] == "mock-substems"
    assert caps["instruments"] == ["guitar", "piano", "keys"] and caps["parents"] == ["other"]
    assert caps["residual"] == {"name": "Rest", "label": "rest"} and caps["aliases"]["rhodes"] == "keys"
    assert health["backend"] == "mock" and health["instruments"] == ["guitar", "piano", "keys"]


@pytest.mark.parametrize("parent", ["vocals", "drums", "bass", "guitar"])
def test_other_parents_are_refused(mock_settings, parent):
    with TestClient(create_app(mock_settings)) as client:
        r = post(client, {"parent_label": parent})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "UNSUPPORTED_OPERATION" and err["details"]["parents"] == ["other"] and err["retryable"] is False


def test_parents_are_configurable(settings, monkeypatch):
    monkeypatch.setenv("AUDIO_STEM_SPLITTER", "mock")
    monkeypatch.setenv("AUDIO_SUBSTEM_PARENTS", "other,instrumental")
    with TestClient(create_app(load_settings())) as client:
        assert post(client, {"parent_label": "Instrumental", "stems": ["piano"]}).status_code == 200


def test_instruments_the_model_cannot_separate_are_400(mock_settings):
    with TestClient(create_app(mock_settings)) as client:
        r = post(client, {"stems": ["piano", "strings"]})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "INVALID_REQUEST" and err["details"]["unknown"] == ["strings"] and err["details"]["valid_stems"] == ["guitar", "piano", "keys"]


def test_unavailable_when_no_nested_splitter(client):
    r = post(client, {"stems": ["piano"]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "UNSUPPORTED_OPERATION"


def test_route_dry_run(mock_settings):
    with TestClient(create_app(mock_settings)) as client:
        meta = {"options": {"mode": "music_substems", "stems": ["rhodes"]}}
        r = client.post("/v1/route", files={"audio": ("o.wav", song_wav(), "audio/wav")}, data={"metadata": json.dumps(meta)})
    assert r.status_code == 200, r.text
    assert r.json()["routing"]["mode"] == "music_substems" and r.json()["routing"]["stems"] == ["keys"]


# --------------------------------------------------------------------- HTTP, fakes
def test_detection_picks_present_instruments_with_tag_names(mock_settings):
    sep = FakeSeparator()
    tagger = FixedTagger(Piano=0.05, **{"Electric piano": 0.45, "Electric guitar": 0.3, "Violin, fiddle": 0.6})
    with make_client(mock_settings, sep, tagger) as client:
        assert sep.load_calls == 0  # lazy: nothing loads before the first nested request
        r = post(client, {"parent_label": "other"})
    assert r.status_code == 200, r.text
    body = r.json()
    # Strings are heard but not offered; keys (Electric piano) beats guitar; names come from the tags.
    assert [(s["name"], s["label"]) for s in body["stems"]] == [("Electric piano", "keys"), ("Electric guitar", "guitar"), ("Rest", "rest")]
    assert body["routing"]["stems"] == ["keys", "guitar"] and body["model"]["model_id"] == "fake-6s"
    assert all(s["tags"] is not None for s in body["stems"])
    total = sum(samples(s) for s in body["stems"])
    assert np.max(np.abs(total - mix())) < 1e-4
    assert sep.split_calls == 1


def test_detection_respects_max_and_threshold(settings, monkeypatch):
    monkeypatch.setenv("AUDIO_STEM_SPLITTER", "mock")
    monkeypatch.setenv("AUDIO_SUBSTEM_MAX", "1")
    monkeypatch.setenv("AUDIO_SUBSTEM_MIN_SCORE", "0.2")
    with make_client(load_settings(), FakeSeparator(), FixedTagger(Piano=0.5, Guitar=0.4)) as client:
        body = post(client, {}).json()
    assert [s["label"] for s in body["stems"]] == ["piano", "rest"]


def test_nothing_detected_is_a_clear_400(mock_settings):
    sep = FakeSeparator()
    with make_client(mock_settings, sep, FixedTagger(Synthesizer=0.8, Organ=0.5)) as client:
        r = post(client, {"parent_label": "other"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "INVALID_REQUEST" and err["retryable"] is False
    assert err["message"] == "No separable instruments found in this track"
    assert err["details"]["reason"] == "no_separable_instruments" and err["details"]["separable"] == ["guitar", "piano", "keys"]
    assert sep.split_calls == 0  # no model run for nothing


def test_near_silent_instruments_fold_into_rest(mock_settings):
    sep = FakeSeparator({"guitar": 0.001, "piano": 0.6, "other": 0.399})  # guitar at -60 dB
    with make_client(mock_settings, sep, FixedTagger()) as client:
        r = post(client, {"stems": ["piano", "guitar"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [s["label"] for s in body["stems"]] == ["piano", "rest"]
    assert any("Guitar came back near-silent" in w and "folded into Rest" in w for w in body["warnings"])
    m = mix()
    stems = {s["label"]: samples(s) for s in body["stems"]}
    np.testing.assert_allclose(stems["piano"], 0.6 * m, atol=2e-4)
    np.testing.assert_allclose(stems["rest"], 0.4 * m, atol=2e-4)


def test_everything_silent_is_the_same_400(mock_settings):
    sep = FakeSeparator({"guitar": 0.0, "piano": 0.0, "other": 1.0})
    with make_client(mock_settings, sep, FixedTagger()) as client:
        r = post(client, {"stems": ["piano"]})
    assert r.status_code == 400 and r.json()["error"]["details"]["reason"] == "no_separable_instruments"


def test_without_a_tagger_every_output_is_tried_once(mock_settings):
    sep = FakeSeparator({"guitar": 0.001, "piano": 0.7, "other": 0.299})
    with make_client(mock_settings, sep, None) as client:  # mock settings: classifier off
        body = post(client, {}).json()
    assert [s["label"] for s in body["stems"]] == ["piano", "rest"]
    assert any("instrument detection unavailable" in w for w in body["warnings"])


def test_piano_and_keys_share_one_output(mock_settings):
    with make_client(mock_settings, FakeSeparator(), FixedTagger()) as client:
        body = post(client, {"stems": ["piano", "keys", "guitar"]}).json()
    assert [s["label"] for s in body["stems"]] == ["piano", "guitar", "rest"]
    assert any("'keys' is separated together with 'Piano'" in w for w in body["warnings"])


def test_too_many_requested_instruments(settings, monkeypatch):
    monkeypatch.setenv("AUDIO_STEM_SPLITTER", "mock")
    monkeypatch.setenv("AUDIO_SUBSTEM_MAX", "1")
    with make_client(load_settings(), FakeSeparator(), FixedTagger()) as client:
        r = post(client, {"stems": ["piano", "guitar"]})
    assert r.status_code == 400 and "AUDIO_SUBSTEM_MAX" in r.json()["error"]["message"]


def test_shares_the_engine_lock(mock_settings):
    sep = FakeSeparator()
    engine = Engine(mock_settings, classifier=FixedTagger(), substems=sep)
    seen = []
    original = sep.split

    def split(mixture, progress=None, cancel=None):
        seen.append(engine._lock.locked())
        return original(mixture, progress, cancel)

    sep.split = split
    with TestClient(create_app(mock_settings, engine)) as client:
        assert post(client, {"stems": ["piano"]}).status_code == 200
    assert seen == [True]


# --------------------------------------------------------------------- benchmark harness (pure parts)
def test_benchmark_policy_counts_returned_and_phantom_tracks(tmp_path):
    from types import SimpleNamespace

    from benchmarks import nested_stems as B

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "detect_panns.json").write_text(json.dumps([
        {"clip": "c1", "cond": "B", "families": {"piano": 0.5}, "present": ["piano"]}, {"clip": "c2", "cond": "B", "families": {"piano": 0.02}, "present": []}]))
    rows = [
        {"tag": "t", "cond": "B", "clip": "c1", "family": "piano", "state": "present", "est_db": -5.0, "usdr": 4.0, "sdri": 6.0, "csdr": 3.0},
        {"tag": "t", "cond": "B", "clip": "c2", "family": "piano", "state": "absent", "est_db": -10.0},
    ]
    policy = B._score_policy(SimpleNamespace(work=str(tmp_path)), rows)
    by = {(r["min_score"], r["silent_db"]): r for r in policy}
    assert by[(None, -20.0)]["absent_returned"] == 1.0  # no tagger gate: the phantom piano is returned
    assert by[(0.1, -20.0)]["absent_returned"] == 0.0 and by[(0.1, -20.0)]["present_returned"] == 1.0
    assert by[(0.1, -20.0)]["returned_usdr_mean"] == 4.0
    assert by[(None, -15.0)]["absent_returned"] == 1.0 and by[(0.2, -30.0)]["present_returned"] == 1.0
    assert by[("present", -20.0)]["absent_returned"] == 0.0 and by[("present", -20.0)]["present_returned"] == 1.0
    assert set(B.SLAKH_FAMILY.values()) - {None} == set(B.FAMILIES)
