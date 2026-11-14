"""Stem splitters, mixture consistency and the stem vocabulary. No torch or demucs needed."""
import contextlib
import sys
import types

import numpy as np
import pytest

from sam_audio_service.audio_io import encode_wav_pcm16
from sam_audio_service.config import Settings
from sam_audio_service.stems import create_splitter, mixture_consistency
from sam_audio_service.stems.chain import ChainSplitter
from sam_audio_service.stems.demucs import DemucsSplitter
from sam_audio_service.stems.mock import MockStemSplitter
from sam_audio_service.stems.vocabulary import canonical_stem, components, requestable

SR = 48_000


def song(seconds: float = 1.0, channels: int = 2) -> np.ndarray:
    """One partial per mock band: 80 Hz bass, 440 Hz voice, 5 kHz other, 10 kHz cymbal."""
    t = np.arange(int(seconds * SR)) / SR
    mono = 0.3 * np.sin(2 * np.pi * 80 * t) + 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 5000 * t) + 0.05 * np.sin(2 * np.pi * 10000 * t)
    return np.stack([mono * (1.0 - 0.2 * c) for c in range(channels)]).astype(np.float32)


# ---------------------------------------------------------------- mock splitter
@pytest.mark.parametrize("channels", [1, 2])
def test_mock_splitter_bands_shapes_and_determinism(channels):
    mix = song(channels=channels)
    splitter = MockStemSplitter()
    splitter.load()
    stems = splitter.split(mix)
    assert splitter.is_mock and set(stems) == {"vocals", "drums", "bass", "other"}
    assert all(s.shape == mix.shape and s.dtype == np.float32 for s in stems.values())
    energy = {n: float(np.mean(s ** 2)) for n, s in stems.items()}
    assert max(energy, key=energy.get) in {"bass", "vocals"} and energy["drums"] > 0
    again = splitter.split(mix)
    assert all(np.array_equal(stems[n], again[n]) for n in stems)
    assert np.max(np.abs(sum(stems.values()) - mix)) < 1e-3  # bands partition the spectrum


def test_mock_splitter_six_stems_and_short_input():
    splitter = MockStemSplitter(six_stems=True)
    assert set(splitter.stems) == {"vocals", "drums", "bass", "guitar", "piano", "other"}
    short = song(seconds=0.01)
    assert all(s.shape == short.shape for s in splitter.split(short).values())


def test_create_splitter_follows_settings():
    assert create_splitter(Settings(backend="mock", stem_splitter="off")) is None
    assert isinstance(create_splitter(Settings(backend="mock", stem_splitter="mock")), MockStemSplitter)
    assert len(create_splitter(Settings(backend="mock", stem_splitter="mock", stem_model="htdemucs_6s")).stems) == 6
    demucs = create_splitter(Settings(backend="mock", stem_splitter="demucs", stem_shifts=2, stem_overlap=0.5, stem_segment=7.0))
    assert isinstance(demucs, DemucsSplitter) and not demucs.loaded
    assert (demucs.model_name, demucs.shifts, demucs.overlap, demucs.segment) == ("htdemucs_ft", 2, 0.5, 7.0)
    assert isinstance(create_splitter(Settings(backend="mock", stem_splitter="chain")), ChainSplitter)


# ---------------------------------------------------------------- consistency
def test_mixture_consistency_sums_exactly_and_follows_energy():
    rng = np.random.default_rng(0)
    mix = rng.standard_normal((2, 48_000)).astype(np.float32) * 0.3
    stems = {"vocals": mix * 0.5, "drums": mix * 0.2, "silent": np.zeros_like(mix)}  # 0.3 of the mix is missing
    before = {n: s.copy() for n, s in stems.items()}
    fixed = mixture_consistency(mix, stems)
    assert np.max(np.abs(sum(fixed.values()) - mix)) < 1e-4
    assert np.max(np.abs(fixed["silent"])) < 1e-6  # a silent stem takes ~none of the error
    assert np.mean(np.abs(fixed["vocals"] - before["vocals"])) > np.mean(np.abs(fixed["drums"] - before["drums"]))
    assert fixed["vocals"] is stems["vocals"]  # float32 stems are corrected in place, not copied


def test_mixture_consistency_uses_short_time_energy():
    rng = np.random.default_rng(1)
    n = 96_000
    vocals = rng.standard_normal((2, n)).astype(np.float32) * 0.2
    vocals[:, : n // 2] = 0.0  # instrumental break in the first half
    drums = rng.standard_normal((2, n)).astype(np.float32) * 0.2
    mix = (vocals + drums) * 1.1  # 10% missing everywhere
    fixed = mixture_consistency(mix, {"vocals": vocals.copy(), "drums": drums.copy()})
    assert np.max(np.abs(fixed["vocals"][:, : n // 2 - 2048])) < 1e-6  # no leaked error during the break
    assert np.max(np.abs(fixed["vocals"] + fixed["drums"] - mix)) < 1e-4


def test_mixture_consistency_long_input_stays_float32():
    rng = np.random.default_rng(2)
    n = 48_000 * 40 + 123  # several processing blocks plus a ragged tail
    mix = rng.standard_normal((2, n)).astype(np.float32) * 0.3
    stems = {name: (mix * w).astype(np.float32) for name, w in (("vocals", 0.4), ("drums", 0.3), ("bass", 0.1), ("other", 0.1))}
    fixed = mixture_consistency(mix, stems)
    assert all(s.dtype == np.float32 and s.shape == mix.shape for s in fixed.values())
    total = fixed["vocals"].copy()
    for name in ("drums", "bass", "other"):
        total += fixed[name]
    assert np.max(np.abs(total - mix)) < 1e-4


def test_mixture_consistency_all_silent_splits_equally():
    mix = np.ones((1, 10), dtype=np.float32)
    fixed = mixture_consistency(mix, {"a": np.zeros_like(mix), "b": np.zeros_like(mix)})
    assert np.allclose(fixed["a"], 0.5) and np.allclose(fixed["b"], 0.5)


# ---------------------------------------------------------------- vocabulary
@pytest.mark.parametrize("wording,canonical", [
    ("vocals", "vocals"), ("Lead Vocal", "vocals"), ("singer", "vocals"), ("Isolate the singing!", "vocals"), ("voice", "vocals"),
    ("the drums", "drums"), ("percussion", "drums"), ("beat", "drums"), ("Bassline", "bass"), ("bass guitar", "bass"),
    ("acoustic guitar", "guitar"), ("keys", "piano"), ("karaoke", "instrumental"), ("Backing track", "instrumental"),
    ("accompaniment", "instrumental"), ("other", "other"),
    ("dog barking", None), ("speech", None), ("narrator", None), ("music", None), ("", None),
])
def test_canonical_stem(wording, canonical):
    assert canonical_stem(wording) == canonical


def test_instrumental_is_every_non_vocal_stem():
    four = ("drums", "bass", "other", "vocals")
    assert components("instrumental", four) == ["drums", "bass", "other"]
    assert components("guitar", four) == []
    assert requestable(four) == ["vocals", "drums", "bass", "other", "instrumental"]


# ---------------------------------------------------------------- demucs with a fake module
class FakeDemucsModel:
    samplerate = 44_100
    audio_channels = 2
    sources = ["drums", "bass", "other", "vocals"]
    weights = (0.1, 0.2, 0.3, 0.4)

    def eval(self):
        return self


def tone(seconds: float, channels: int) -> np.ndarray:
    """Low enough that the fake's linear-interpolation resampler is accurate."""
    t = np.arange(int(seconds * SR)) / SR
    return np.stack([0.3 * np.sin(2 * np.pi * 100 * t) * (1.0 - 0.2 * c) for c in range(channels)]).astype(np.float32)


def _interp(x: np.ndarray, n_out: int) -> np.ndarray:
    old = np.linspace(0.0, 1.0, x.shape[-1])
    new = np.linspace(0.0, 1.0, n_out)
    flat = x.reshape(-1, x.shape[-1])
    return np.stack([np.interp(new, old, row) for row in flat]).reshape(*x.shape[:-1], n_out)


@pytest.fixture
def fake_demucs(monkeypatch):
    calls: dict = {"convert": []}

    def get_model(name):
        calls["model"] = name
        return FakeDemucsModel()

    def convert_audio(wav, from_rate, to_rate, channels):
        assert wav.shape[-2] == channels == 2
        calls["convert"].append((from_rate, to_rate, wav.shape))
        return _interp(np.asarray(wav), int(round(wav.shape[-1] * to_rate / from_rate)))

    def apply_model(model, mix, **kwargs):
        calls["apply"] = {"shape": mix.shape, **kwargs}
        return np.stack([mix * w for w in model.weights], axis=1)  # (batch, sources, channels, frames)

    demucs = types.ModuleType("demucs")
    modules = {
        "demucs": demucs,
        "demucs.pretrained": types.SimpleNamespace(get_model=get_model),
        "demucs.apply": types.SimpleNamespace(apply_model=apply_model),
        "demucs.audio": types.SimpleNamespace(convert_audio=convert_audio),
        "torch": types.SimpleNamespace(
            from_numpy=lambda a: a, inference_mode=contextlib.nullcontext,
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return calls


def test_demucs_splitter_resamples_and_keeps_shape(fake_demucs):
    splitter = DemucsSplitter("htdemucs", device="auto", shifts=2, overlap=0.5, segment=7.0)
    splitter.load()
    assert splitter.loaded and splitter.device == "cpu" and fake_demucs["model"] == "htdemucs"
    mix = tone(1.0, 2)
    stems = splitter.split(mix)
    assert fake_demucs["apply"]["shape"] == (1, 2, 44_100)  # the model sees its own rate
    assert {k: fake_demucs["apply"][k] for k in ("shifts", "overlap", "segment", "device", "split")} == {"shifts": 2, "overlap": 0.5, "segment": 7.0, "device": "cpu", "split": True}
    assert [c[:2] for c in fake_demucs["convert"]] == [(48_000, 44_100), (44_100, 48_000)]
    assert set(stems) == {"drums", "bass", "other", "vocals"}
    assert all(s.shape == mix.shape and s.dtype == np.float32 for s in stems.values())
    # The model scales the normalised mix; undoing the normalisation gives w * mix + (1 - w) * mean.
    assert np.max(np.abs(stems["vocals"] - 0.4 * mix)) < 5e-3
    assert np.max(np.abs(sum(stems.values()) - mix)) < 5e-3


def test_demucs_splitter_mono_in_mono_out(fake_demucs):
    splitter = DemucsSplitter("htdemucs")
    splitter.load()
    mix = tone(0.5, 1)
    stems = splitter.split(mix)
    assert fake_demucs["apply"]["shape"][1] == 2  # duplicated to stereo for the model
    assert all(s.shape == (1, mix.shape[1]) for s in stems.values())
    assert np.max(np.abs(stems["drums"] - 0.1 * mix)) < 5e-3


def test_demucs_split_before_load_fails():
    with pytest.raises(RuntimeError, match="not loaded"):
        DemucsSplitter().split(song())


# ---------------------------------------------------------------- chain with a fake audio-separator
def test_chain_splitter_vocal_model_then_demucs(monkeypatch):
    class FakeSeparator:
        def __init__(self, output_dir, output_format, **kwargs):
            self.output_dir = output_dir

        def load_model(self, model_filename):
            self.model = model_filename

        def separate(self, path):
            from sam_audio_service.audio_io import decode

            mix = decode(path=path).samples
            names = {"Vocals": 0.25 * mix, "Instrumental": 0.75 * mix}
            out = []
            for key, audio in names.items():
                name = f"mix_({key})_{self.model}.wav"
                with open(f"{self.output_dir}/{name}", "wb") as f:
                    f.write(encode_wav_pcm16(audio, SR))
                out.append(name)
            return out

    monkeypatch.setitem(sys.modules, "audio_separator", types.ModuleType("audio_separator"))
    monkeypatch.setitem(sys.modules, "audio_separator.separator", types.SimpleNamespace(Separator=FakeSeparator))

    class StubDemucs(DemucsSplitter):
        def load(self):
            self.model = object()

        def split(self, mixture):
            self.seen = mixture
            return {"drums": mixture * 0.5, "bass": mixture * 0.3, "other": mixture * 0.1, "vocals": mixture * 0.1}

    demucs = StubDemucs("htdemucs")
    chain = ChainSplitter("vocals.ckpt", demucs)
    chain.load()
    mix = song(seconds=0.5)
    stems = chain.split(mix)
    assert set(stems) == {"drums", "bass", "other", "vocals"}
    assert np.max(np.abs(demucs.seen - 0.75 * mix)) < 1e-3  # Demucs ran on the instrumental
    assert np.max(np.abs(stems["vocals"] - 0.25 * mix)) < 1e-3
    assert np.max(np.abs(stems["other"] - 0.15 * mix)) < 1e-3  # Demucs's vocal leakage folded into other
    assert not any(chain.output_dir.iterdir())  # per-request files are cleaned up
    assert chain.info()["model_id"] == "vocals.ckpt+htdemucs"
    chain.unload()
