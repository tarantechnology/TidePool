"""Overlap-add windowing and the native BS-RoFormer splitter, with fake torch / bs_roformer."""
import contextlib
import sys
import threading
import types

import numpy as np
import pytest

from sam_audio_service.config import Settings
from sam_audio_service.errors import Cancelled
from sam_audio_service.stems import create_splitter
from sam_audio_service.stems.overlap import overlap_add, triangular_window
from sam_audio_service.stems.roformer import RoformerSplitter, constructor_kwargs, unwrap_state


def gains(*g):
    return lambda batch: np.stack([batch * k for k in g], axis=1)


# ---------------------------------------------------------------- overlap-add
@pytest.mark.parametrize("n", [1, 7, 999, 1000, 1001, 4321])
@pytest.mark.parametrize("overlap", [1, 2, 4, 3])
@pytest.mark.parametrize("batch", [1, 3])
def test_identity_reconstructs_exactly(n, overlap, batch):
    x = np.random.default_rng(n).standard_normal((2, n)).astype(np.float32)
    y = overlap_add(x, gains(1.0), chunk=1000, overlap=overlap, batch=batch)
    assert y.shape == (1, 2, n) and y.dtype == np.float32
    assert np.max(np.abs(y[0] - x)) < 1e-5


def test_gain_model_scales_each_stem_exactly():
    x = np.random.default_rng(0).standard_normal((2, 2500)).astype(np.float32)
    y = overlap_add(x, gains(0.5, 2.0, -1.0), chunk=512, overlap=2)
    for stem, g in zip(y, (0.5, 2.0, -1.0), strict=True):
        assert np.max(np.abs(stem - g * x)) < 1e-5


def test_window_is_strictly_positive_and_no_clipping():
    assert triangular_window(8).min() > 0
    x = np.full((1, 300), 3.0, dtype=np.float32)  # far above full scale: left alone
    assert np.allclose(overlap_add(x, gains(1.0), chunk=128, overlap=2), 3.0)


def test_progress_per_batch_and_cancel():
    x = np.zeros((2, 5000), dtype=np.float32)
    seen = []
    overlap_add(x, gains(1.0), chunk=1000, overlap=2, batch=2, progress=lambda f, s: seen.append(f))
    assert seen == sorted(seen) and seen[-1] == 1.0 and len(seen) > 2
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled):
        overlap_add(x, gains(1.0), chunk=1000, overlap=2, cancel=cancel)


def test_bad_model_output_is_rejected():
    with pytest.raises(ValueError, match="model returned"):
        overlap_add(np.zeros((2, 100), np.float32), lambda b: b, chunk=50, overlap=2)


# ---------------------------------------------------------------- construction helpers
def test_constructor_kwargs_drop_unknown_keys_and_tuple_lists():
    class Model:
        def __init__(self, dim, freqs_per_bands=(), stereo=False):
            pass

    kwargs, dropped = constructor_kwargs(Model, {"dim": 8, "freqs_per_bands": [2, [3, 4]], "mlp_expansion_factor": 4})
    assert kwargs == {"dim": 8, "freqs_per_bands": (2, (3, 4))} and dropped == ["mlp_expansion_factor"]
    assert unwrap_state({"state_dict": {"a": 1}}) == {"a": 1} and unwrap_state({"a": 1}) == {"a": 1}


# ---------------------------------------------------------------- splitter with fakes
class FakeBSRoformer:
    created: list = []
    keys = ("layers.0.weight", "layers.1.weight", "final.weight")

    def __init__(self, dim, depth, stereo=False, num_stems=1, freqs_per_bands=()):
        self.kwargs = {"dim": dim, "depth": depth, "stereo": stereo, "num_stems": num_stems, "freqs_per_bands": freqs_per_bands}
        self.gains = (0.1, 0.2, 0.3, 0.4)[:num_stems]
        FakeBSRoformer.created.append(self)

    def state_dict(self):
        return dict.fromkeys(self.keys, 0)

    def load_state_dict(self, state, strict=True):
        return [k for k in self.keys if k not in state], [k for k in state if k not in self.keys]

    def to(self, device):
        return self

    def eval(self):
        return self

    def half(self):
        self.halved = True
        return self

    def __call__(self, x):
        return np.stack([x * g for g in self.gains], axis=1)


@pytest.fixture
def fake_roformer(monkeypatch, tmp_path):
    state = {"state": {"model": dict.fromkeys(FakeBSRoformer.keys, 0)}}

    def from_numpy(arr):
        return types.SimpleNamespace(to=lambda *a, **k: arr)

    torch = types.SimpleNamespace(
        load=lambda path, map_location=None, weights_only=None: state["state"], from_numpy=from_numpy,
        inference_mode=contextlib.nullcontext, float16="f16", float32="f32", autocast=lambda *a, **k: contextlib.nullcontext(),
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "bs_roformer", types.SimpleNamespace(BSRoformer=FakeBSRoformer))
    single = tmp_path / "vocals.yaml"
    single.write_text(
        "audio:\n  chunk_size: 4000\n  sample_rate: 44100\n"
        "model:\n  dim: 8\n  depth: 1\n  stereo: true\n  num_stems: 1\n  freqs_per_bands: !!python/tuple\n  - 2\n  - 2\n"
        "training:\n  instruments: [Vocals, Instrumental]\n  target_instrument: Vocals\n"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "audio:\n  chunk_size: 4000\n  sample_rate: 44100\n"
        "model:\n  dim: 8\n  depth: 1\n  stereo: true\n  num_stems: 4\n  freqs_per_bands: [2, 2]\n  mlp_expansion_factor: 4\n"
        "training:\n  instruments: [drums, bass, other, vocals]\n"
    )
    calls = []

    def resampler(x, src, dst):
        calls.append((src, dst))
        n_out = int(round(x.shape[-1] * dst / src))
        old, new = np.linspace(0, 1, x.shape[-1]), np.linspace(0, 1, n_out)
        return np.stack([np.interp(new, old, row) for row in x]).astype(np.float32)

    return types.SimpleNamespace(config=str(config), single=str(single), state=state, resample_calls=calls, resampler=resampler)


def tone(seconds, channels):
    t = np.arange(int(seconds * 48_000)) / 48_000
    return np.stack([0.3 * np.sin(2 * np.pi * 100 * t) * (1 - 0.2 * c) for c in range(channels)]).astype(np.float32)


@pytest.mark.parametrize("channels", [1, 2])
def test_roformer_splitter_end_to_end(fake_roformer, channels):
    splitter = RoformerSplitter("/w/bs_roformer_4stem.ckpt", fake_roformer.config, overlap=2, fp16=True, batch=2, resampler=fake_roformer.resampler)
    splitter.load()
    model = FakeBSRoformer.created[-1]
    assert model.kwargs["freqs_per_bands"] == (2, 2) and splitter.device == "cpu" and not hasattr(model, "halved")  # fp16 = autocast, weights stay fp32
    assert splitter.stems == ("drums", "bass", "other", "vocals") and splitter.chunk == 4000
    mix = tone(0.5, channels)
    progress = []
    stems = splitter.split(mix, progress=lambda f, s: progress.append(f))
    assert fake_roformer.resample_calls[0] == (48_000, 44_100) and fake_roformer.resample_calls[-1] == (44_100, 48_000)
    assert all(s.shape == mix.shape and s.dtype == np.float32 for s in stems.values())
    assert np.max(np.abs(stems["vocals"] - 0.4 * mix)) < 5e-3
    assert progress[0] == 0.0 and progress[-1] == 1.0
    info = splitter.info()
    assert info["license"].startswith("MIT") and info["chunk_size"] == 4000 and info["overlap"] == 2 and info["model_id"] == "bs_roformer_4stem.ckpt"


def test_roformer_fails_loudly_on_a_mismatched_checkpoint(fake_roformer):
    fake_roformer.state["state"] = {"something.else": 0}
    with pytest.raises(RuntimeError, match="do not match"):
        RoformerSplitter("x.ckpt", fake_roformer.config, resampler=fake_roformer.resampler).load()


def test_roformer_settings(monkeypatch):
    with pytest.raises(ValueError, match="AUDIO_STEM_ROFORMER_CKPT"):
        Settings(stem_splitter="roformer").validate()
    s = Settings(stem_splitter="roformer", stem_roformer_ckpt="a.ckpt", stem_roformer_config="a.yaml", stem_roformer_overlap=4, stem_roformer_batch=2)
    s.validate()
    splitter = create_splitter(s)
    assert isinstance(splitter, RoformerSplitter) and (splitter.overlap, splitter.batch, splitter.fp16) == (4, 2, False) and not splitter.loaded
    monkeypatch.delenv("AUDIO_ROUTER_CLASSIFIER", raising=False)
    assert Settings(stem_splitter="roformer").router_classifier == "panns"
    for field, value in (("stem_roformer_overlap", 0), ("stem_roformer_batch", 17), ("stem_vocal_overlap", 17)):
        with pytest.raises(ValueError):
            Settings(stem_splitter="mock", **{field: value}).validate()


def test_single_target_model_returns_target_and_complement(fake_roformer):
    splitter = RoformerSplitter("/w/vocals.ckpt", fake_roformer.single, overlap=2, batch=3, resampler=fake_roformer.resampler)
    splitter.load()
    assert splitter.instruments == ["Vocals"] and splitter.stems == ("vocals", "other")
    assert FakeBSRoformer.created[-1].kwargs["freqs_per_bands"] == (2, 2)  # audio-separator YAMLs tag tuples
    x = tone(0.3, 2)
    est = splitter.separate(x)  # model rate in, model rate out
    assert set(est) == {"vocals", "other"} and np.max(np.abs(est["vocals"] - 0.1 * x)) < 1e-5
    assert np.max(np.abs(est["vocals"] + est["other"] - x)) < 1e-6
    stems = splitter.split(tone(0.3, 1))
    assert set(stems) == {"vocals", "other"} and all(s.shape[0] == 1 for s in stems.values())


def test_enable_flash_turns_on_the_flash_kernel_for_every_attend():
    from collections import namedtuple

    from sam_audio_service.stems.roformer import enable_flash

    Config = namedtuple("Config", ["enable_flash", "enable_math", "enable_mem_efficient"])
    attends = [types.SimpleNamespace(cuda_config=Config(False, True, False)) for _ in range(3)]
    model = types.SimpleNamespace(modules=lambda: [object(), *attends, types.SimpleNamespace(cuda_config=None)])
    assert enable_flash(model) == 3
    assert all(a.cuda_config == Config(True, True, True) for a in attends)


def test_fast_rotary_patches_each_shared_embedding_once_and_skips_unsupported():
    from sam_audio_service.stems.roformer import FastRotary, fast_rotary

    def rotary(**kw):
        return types.SimpleNamespace(freqs=np.ones(32), rotate_queries_or_keys=lambda t, **k: ("original", t), **kw)

    shared, xpos = rotary(), rotary(use_xpos=True)
    layers = [types.SimpleNamespace(rotary_embed=shared), types.SimpleNamespace(rotary_embed=shared),
              types.SimpleNamespace(rotary_embed=xpos), types.SimpleNamespace(rotary_embed=None)]
    model = types.SimpleNamespace(modules=lambda: layers)
    assert fast_rotary(model) == 1 and fast_rotary(model) == 0  # idempotent
    assert shared._fast_rotary and not hasattr(xpos, "_fast_rotary")
    assert not FastRotary.supports(rotary(learned_freq=True)) and not FastRotary.supports(rotary(interpolate_factor=2.0))
    # anything but the plain call bs_roformer makes goes to the library's own implementation
    t = types.SimpleNamespace(shape=(1, 8, 10, 64))
    assert shared.rotate_queries_or_keys(t, offset=3) == ("original", t)
