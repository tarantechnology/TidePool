import numpy as np
from sam_audio_service.explode import explode, kind_of


def _sep(samples, prompt):
    return samples * 0.5


def test_music_words():
    assert kind_of("orchestral music") == "music"
    assert kind_of("film score") == "music"
    assert kind_of("people talking") == "voice"


def test_music_layer_splits_into_audible_instruments_and_sums_exactly():
    rng = np.random.default_rng(0)
    mix = rng.standard_normal((2, 4800)).astype(np.float32) * 0.1
    def split(music):
        return {"drums": music * 0.6, "bass": music * 0.4 - 1e-9, "piano": music * 1e-6}
    out = explode(mix, 48000, ["orchestral music", "high-pitched screech"], _sep, None, split_music=split)
    names = [l.name for l in out.layers]
    assert names[:2] == ["orchestral music: Drums", "orchestral music: Bass"]
    assert "orchestral music: Piano" not in names           # -120 dB: folded into the loudest
    music = sum(l.samples for l in out.layers if l.name.startswith("orchestral music"))
    assert np.allclose(music, mix * 0.5, atol=1e-6)
    total = sum(l.samples for l in out.layers) + out.residual
    assert np.allclose(total, mix, atol=1e-6)


def test_failed_split_keeps_the_music_whole():
    mix = np.ones((1, 100), dtype=np.float32)
    def boom(music):
        raise RuntimeError("no gpu")
    out = explode(mix, 48000, ["music"], _sep, None, split_music=boom)
    assert [l.name for l in out.layers] == ["music"] and any("kept whole" in w for w in out.warnings)


def test_without_split_music_nothing_changes():
    mix = np.ones((1, 100), dtype=np.float32)
    out = explode(mix, 48000, ["orchestral music"], _sep, None)
    assert [l.name for l in out.layers] == ["orchestral music"]
