import json

import numpy as np

from conftest import synth
from sam_audio_service.explode import explode, highpass, kind_of


def test_kinds():
    assert kind_of('people talking') == 'voice'
    assert kind_of('deep male voice speaking') == 'voice'
    assert kind_of('man shouting over the rain') == 'voice'
    assert kind_of('heavy rain') == 'rain'
    assert kind_of('thunder rumbling') == 'thunder'
    assert kind_of('orchestral music') == 'music'
    assert kind_of('high-pitched screech') == 'sound'


def test_highpass_splits_low_from_high():
    sr = 8000
    t = np.arange(sr) / sr
    low, high = np.sin(2 * np.pi * 50 * t), np.sin(2 * np.pi * 2000 * t)
    out = highpass((low + high)[None].astype(np.float32), sr, 200.0)[0]
    rest = (low + high) - out
    def level(x, hz):  # amplitude of the hz bin over the settled second half
        tail = x[sr // 2:]
        return 2 * abs(np.fft.rfft(tail)[int(hz * len(tail) / sr)]) / len(tail)
    assert level(out, 2000) > 0.95 and level(out, 50) < 0.1
    assert level(rest, 50) > 0.9 and level(rest, 2000) < 0.1
    # The low band is what the high band leaves, exactly.
    assert np.allclose(out + rest, low + high, atol=1e-5)


def _fake_separate(calls):
    def separate(audio, prompt):
        calls.append(prompt)
        return (audio * 0.25).astype(np.float32)
    return separate


def test_recipe_order_and_exact_sum():
    rng = np.random.default_rng(0)
    mix = rng.standard_normal((2, 4800)).astype(np.float32) * 0.1
    calls = []
    vocals_calls = []

    def vocals(audio):
        vocals_calls.append(audio.copy())
        return (audio * 0.5).astype(np.float32)

    out = explode(mix, 48000, ['people talking', 'heavy rain', 'thunder rumbling', 'orchestral music', 'high-pitched screech'],
                  _fake_separate(calls), vocals)
    assert [l.name for l in out.layers] == ['dialogue', 'heavy rain', 'thunder rumbling', 'orchestral music', 'high-pitched screech']
    assert [l.how for l in out.layers] == ['demucs', 'crossover-high', 'crossover-low', 'audiosep', 'audiosep']
    # One storm pass for both weather layers, using the storm prompt.
    assert calls == ['rain', 'orchestral music', 'high-pitched screech']
    assert len(vocals_calls) == 1 and np.allclose(vocals_calls[0], mix)
    total = sum(l.samples for l in out.layers) + out.residual
    assert np.max(np.abs(total - mix)) < 1e-5
    rain, thunder = out.layers[1].samples, out.layers[2].samples
    assert np.max(np.abs(rain + thunder - mix * 0.5 * 0.25)) < 1e-5


def test_extra_voices_merge_and_no_demucs_goes_last():
    mix = np.ones((1, 100), dtype=np.float32)
    calls = []
    out = explode(mix, 48000, ['people talking', 'woman speaking', 'heavy rain'], _fake_separate(calls), None)
    assert [l.name for l in out.layers] == ['heavy rain', 'people talking']
    assert any('woman speaking' in w for w in out.warnings) and any('Demucs' in w for w in out.warnings)


def test_explode_endpoint_mock(client, tmp_path, monkeypatch):
    monkeypatch.setenv('AUDIO_EXPLODE_DEMUCS', '0')
    path = synth(tmp_path / 'clip.wav', seconds=2.0)
    with path.open('rb') as f:
        r = client.post('/explode', files={'audio': ('clip.wav', f, 'audio/wav')},
                        data={'metadata': json.dumps({'prompts': ['heavy rain', 'thunder rumbling', 'orchestral music']})})
    assert r.status_code == 200, r.text
    body = r.json()
    assert [s['name'] for s in body['stems']] == ['heavy rain', 'thunder rumbling', 'orchestral music', 'residual']
    assert body['stems'][-1]['role'] == 'residual'
    assert any(w.startswith('Explode recipe:') for w in body['warnings'])
    caps = client.get('/capabilities').json()
    assert caps['prompting_modes']['explode'] is True and caps['explode']['crossover_hz'] == 200.0
