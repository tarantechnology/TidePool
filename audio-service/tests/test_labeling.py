"""Instrument tags, descriptive names and presence for music stems (labeling.py), with a fake tagger."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sam_audio_service import labeling as L
from sam_audio_service.api import create_app
from sam_audio_service.classifier import ClapTagger, ContentClassifier, create_stem_tagger
from sam_audio_service.config import load_settings
from sam_audio_service.engine import Engine

from test_music_stems_api import post, song_wav

SR = 48_000
NAMES = ["Speech", "Music", "Gong", *dict.fromkeys(c for lab in L.LABELS for c in lab.audioset if c != "Speech")]


def probs(**scores: float) -> np.ndarray:
    p = np.zeros(len(NAMES))
    for name, score in scores.items():
        p[NAMES.index(name.replace("_", " "))] = score
    return p


def scores_for(vocab, *rows):
    """(windows, len(vocab)) label scores from AudioSet probability rows."""
    return L.label_scores(np.stack(rows), NAMES, vocab)


# What the fake "hears" in each mock-splitter band (bass < 150 Hz < vocals < 3.5 kHz < other < 8 kHz < drums).
BAND_PROBS = [
    (150, probs(**{"Bass guitar": 0.7, "Synthesizer": 0.2})),
    # Separated vocals: AudioSet spreads singing over odd classes ("Chant", "A capella"); never shown.
    (3500, probs(**{"Singing": 0.1, "A capella": 0.4, "Chant": 0.3, "Female singing": 0.6, "Male singing": 0.1, "Speech": 0.3})),
    (8000, probs(**{"Synthesizer": 0.5, "String section": 0.35, "Bowed string instrument": 0.2, "Female singing": 0.4, "Gong": 0.3})),
    (np.inf, probs(**{"Drum": 0.7, "Drum kit": 0.6, "Drum machine": 0.3, "Hi-hat": 0.2})),
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
def test_every_label_maps_to_known_audioset_names_and_vocabularies_are_closed():
    assert len({lab.id for lab in L.LABELS}) == len(L.LABELS)
    for vocab in [*L.STEM_VOCAB.values(), L.INSTRUMENT_IDS]:
        assert all(v in L.LABEL for v in vocab)
    # No voice label can describe an instrument stem, no instrument a vocal stem.
    assert not {L.LABEL[v].family for v in L.vocabulary("other")} & {"vocals", "drums", "bass"}
    assert {L.LABEL[v].family for v in L.vocabulary("vocals")} == {"vocals"}
    assert {L.LABEL[v].family for v in L.vocabulary("bass")} == {"bass"}
    assert set(L.PRESENCE["panns"]) <= set(L.INSTRUMENT_FAMILIES)


def test_label_scores_take_the_max_of_mapped_classes_and_ignore_missing_ones():
    s = scores_for(("strings", "violin", "piano"), probs(**{"String section": 0.3, "Pizzicato": 0.5, "Violin, fiddle": 0.2}))
    assert s.tolist() == [[0.5, 0.2, 0.0]]
    assert L.label_scores(np.ones((1, 2)), ["Piano", "x"], ("piano", "organ")).tolist() == [[1.0, 0.0]]
    assert L.label_scores(np.zeros((0, 2)), ["Piano", "x"], ("piano",)).shape == (0, 1)


def test_zero_shot_scores_are_a_softmax_over_the_stem_vocabulary_plus_noise():
    names = ["piano", "organ", "male_singing", "noise"]
    sims = np.array([[0.30, 0.20, 0.90, 0.10]])  # the voice label is not in the "other" vocabulary
    s = L.label_scores(sims, names, ("piano", "organ", "guitar"), zero_shot=True, temperature=0.05)
    e = np.exp(np.array([0.30, 0.20, 0.10]) / 0.05)
    assert np.allclose(s[0], [e[0] / e.sum(), e[1] / e.sum(), 0.0])
    fams = L.family_scores(np.array([[0.2, 0.3, 0.1]]), ("guitar", "electric_guitar", "piano"), zero_shot=True)
    assert np.isclose(fams["guitar"][0], 0.5) and np.isclose(fams["piano"][0], 0.1)


def test_instruments_use_per_family_thresholds_and_specific_names():
    vocab = L.vocabulary("other")
    # Brass at 0.08 is audible (calibrated 0.03); guitar at 0.1 is not (0.15); 0.5 Electric guitar names the family.
    s = scores_for(vocab, probs(**{"Brass instrument": 0.08, "Guitar": 0.1}), probs(**{"Brass instrument": 0.08, "Guitar": 0.1}))
    inst = {i.family: i for i in L.instruments_from_scores(s, vocab, "panns")}
    assert inst["brass"].present and not inst["guitar"].present and inst["brass"].coverage == 1.0
    s = scores_for(vocab, probs(Guitar=0.6, **{"Electric guitar": 0.4}))
    top = L.instruments_from_scores(s, vocab, "panns")[0]
    assert (top.family, top.name, top.present) == ("guitar", "Electric guitar", True)
    s = scores_for(vocab, probs(Guitar=0.6, **{"Electric guitar": 0.1}))
    assert L.instruments_from_scores(s, vocab, "panns")[0].name == "Guitar"
    # AUDIO_STEM_LABEL_MIN_SCORE is a floor on top of the calibrated values.
    assert L.presence_threshold("panns", "brass") == 0.03 and L.presence_threshold("panns", "brass", 0.2) == 0.2
    assert L.presence_threshold("panns", "guitar", 0.0) == 0.15 and L.presence_threshold("clap", "harp") == L.DEFAULT_PRESENCE["clap"]


@pytest.mark.parametrize(("row", "expected"), [
    ({"Singing": 0.05, "Chant": 0.4, "Mantra": 0.3, "A capella": 0.3}, ["Singing"]),  # "Chant" is evidence, not a tag
    ({"Male singing": 0.3, "Female singing": 0.27, "A capella": 0.2}, ["Singing"]),  # gender too close to call
    ({"Male singing": 0.3, "Female singing": 0.1}, ["Male singing"]),
    ({"Male singing": 0.05, "Female singing": 0.0, "Singing": 0.2}, ["Singing"]),  # below GENDER_MIN
    ({"Rapping": 0.15, "Speech": 0.45, "Male singing": 0.1}, ["Rapping"]),
    ({"Speech": 0.8, "Singing": 0.05}, ["Spoken word"]),
    ({"Choir": 0.4, "Female singing": 0.2}, ["Choir"]),
])
def test_vocal_tags(row, expected):
    vocab = L.vocabulary("vocals")
    tags = [t for t, _ in L._vocal_tags(scores_for(vocab, probs(**row)), vocab, "panns")]
    assert tags[: len(expected)] == expected and "Chant" not in tags


def test_vocal_duet_needs_each_voice_to_win_windows():
    vocab = L.vocabulary("vocals")
    m, f = probs(**{"Male singing": 0.5, "Female singing": 0.1}), probs(**{"Male singing": 0.1, "Female singing": 0.45})
    assert {t for t, _ in L._vocal_tags(scores_for(vocab, m, f), vocab, "panns")} == {"Male singing", "Female singing"}


def test_drums_and_bass_always_get_their_family_tag():
    vocab = L.vocabulary("drums")
    tags = L._family_first_tags(scores_for(vocab, probs(Drum=0.04)), vocab, ("drum_kit", "drum_machine"), ("hihat",), 0.15, "drum_kit")
    assert tags == [("Drum kit", 0.04)]
    tags = L._family_first_tags(scores_for(vocab, probs(**{"Drum machine": 0.5, "Drum kit": 0.2, "Hi-hat": 0.3})), vocab, ("drum_kit", "drum_machine"), ("hihat",), 0.15, "drum_kit")
    assert tags == [("Drum machine", 0.5), ("Hi-hat", 0.3)]
    vocab = L.vocabulary("bass")
    assert L._family_first_tags(scores_for(vocab, probs(Synthesizer=0.4, **{"Bass guitar": 0.2})), vocab, L.BASS_IDS, (), 1.0, "bass_guitar")[0][0] == "Synth bass"
    assert L._family_first_tags(scores_for(vocab, probs(Gong=0.9)), vocab, L.BASS_IDS, (), 1.0, "bass_guitar")[0][0] == "Bass guitar"


def test_describe_uses_top_tag_or_a_pair():
    assert L.describe([("Synth", 0.5), ("Strings", 0.31)]) == "Synth & Strings"
    assert L.describe([("Synth", 0.5), ("Strings", 0.29)]) == "Synth"
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
    assert len(L.active_windows(stem, None, SR, max_windows=20)) == 10


def bands(seconds=12.0):
    t = np.arange(int(seconds * SR)) / SR
    tone = lambda f, a: (a * np.sin(2 * np.pi * f * t)).astype(np.float32)[None].repeat(2, 0)  # noqa: E731
    return {"bass": tone(80, 0.3), "vocals": tone(440, 0.3), "other": tone(5000, 0.1), "drums": tone(10000, 0.05)}


def test_label_stems_uses_per_stem_vocabularies():
    stems = bands()
    mix = sum(stems.values())
    tagger = FakeTagger()
    labels = dict(zip(stems, L.label_stems(tagger, list(stems.items()), mix, SR)))
    assert tagger.calls == 1  # every window in one call
    assert labels["other"].name == "Synth & Strings"
    assert labels["other"].tags == [{"label": "Synth", "score": 0.5}, {"label": "Strings", "score": 0.35}]  # no sung bleed
    assert [i.family for i in labels["other"].instruments if i.present] == ["synth", "strings"]
    assert labels["vocals"].tags == [{"label": "Female singing", "score": 0.6}]
    assert labels["drums"].tags == [{"label": "Drum kit", "score": 0.7}, {"label": "Hi-hat", "score": 0.2}]
    assert labels["bass"].tags == [{"label": "Bass guitar", "score": 0.7}]
    assert ("Synth", 0.5) in labels["other"].candidates
    assert all(len(lb.tags) <= L.MAX_TAGS for lb in labels.values())
    high = dict(zip(stems, L.label_stems(tagger, list(stems.items()), mix, SR, min_score=0.9)))
    # Nothing clears a 0.9 floor; the top family still names the stem at >= NAME_RATIO of it, alone.
    assert high["other"].name == "Synth" and high["other"].tags == [{"label": "Synth", "score": 0.5}] and high["other"].candidates == []
    assert [i.family for i in high["other"].instruments if i.present] == []
    assert high["bass"].tags[0]["label"] == "Bass guitar"  # the family tag does not depend on the floor


def test_near_silent_stem_is_not_tagged():
    stems = bands()
    stems["other"] = stems["other"] * 1e-3  # -72 dB vs the mix
    labels = L.label_stems(FakeTagger(), list(stems.items()), sum(stems.values()), SR)
    other = labels[list(stems).index("other")]
    assert other.name is None and other.tags == [] and other.windows == 0 and other.energy_db < L.SILENT_DB and other.instruments == []


def test_stem_tagger_selection(monkeypatch):
    router = FakeTagger()
    monkeypatch.setenv("AUDIO_STEM_LABEL_MODEL", "router")
    assert create_stem_tagger(load_settings(), router) is router
    monkeypatch.setenv("AUDIO_STEM_LABEL_MODEL", "clap")
    tagger = create_stem_tagger(load_settings(), router)
    assert isinstance(tagger, ClapTagger) and not tagger.loaded and tagger.zero_shot and tagger.can_tag
    assert "noise" in tagger.class_names() and "male_singing" in tagger.class_names() and L.tagger_kind(tagger) == "clap"
    assert L.tagger_kind(router) == "panns"
    monkeypatch.setenv("AUDIO_STEM_LABEL_MODEL", "essentia")
    with pytest.raises(ValueError, match="AUDIO_STEM_LABEL_MODEL"):
        load_settings()


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
    assert stems["bass"]["tags"] == [{"label": "Bass guitar", "score": 0.7}]
    assert stems["drums"]["tags"][0] == {"label": "Drum kit", "score": 0.7}
    assert not any("labeling" in w for w in body["warnings"])


def test_caller_wording_is_kept_but_options_stems_names_are_replaced(tagged_settings):
    with client_for(tagged_settings, FakeTagger()) as c:
        worded = post(c, {"prompts": ["other instruments", "vocals"]}).json()
        listed = post(c, {"options": {"mode": "music_stems", "stems": ["other", "bass"], "residual_name": "the rest"}}).json()
    assert [(s["name"], s["label"]) for s in worded["stems"]][0] == ("other instruments", "other")
    assert worded["stems"][0]["tags"][0]["label"] == "Synth"
    assert [(s["name"], s["role"]) for s in listed["stems"]] == [("Synth & Strings", "target"), ("Bass", "target"), ("the rest", "residual")]
    # The residual (vocals + drums here; the fake hears its loudest part) is tagged too, singing included.
    assert listed["stems"][2]["tags"][0]["label"] == "Female singing"


def test_labeling_failure_never_fails_the_split(tagged_settings):
    with client_for(tagged_settings, FakeTagger(fail=True)) as c:
        r = post(c, {"prompts": []})
    assert r.status_code == 200
    body = r.json()
    assert [s["name"] for s in body["stems"]][-1] == "Other"
    assert all(s["tags"] is None for s in body["stems"])
    assert any("stem labeling failed" in w for w in body["warnings"])


def test_a_separate_tagger_that_fails_is_disabled_with_its_own_error(tagged_settings):
    engine = Engine(tagged_settings)  # mock router classifier (cannot tag)
    engine.stem_tagger = FakeTagger(fail=True)  # e.g. AUDIO_STEM_LABEL_MODEL=clap whose weights fail
    with TestClient(create_app(tagged_settings, engine)) as c:
        first, second = post(c, {"prompts": []}).json(), post(c, {"prompts": []}).json()
    assert engine.stem_tagger.calls == 1  # not retried on the next request
    assert any("stem labeling failed" in w for w in first["warnings"])
    assert any("stem tagger failed earlier (RuntimeError: tagger exploded)" in w for w in second["warnings"])
    assert engine._classifier_error is None  # routing is unaffected


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
    monkeypatch.delenv("AUDIO_STEM_LABEL_MIN_SCORE")
    assert load_settings().stem_label_min_score == 0.0  # calibrated thresholds only
    assert load_settings().stem_label_model == "router"


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
