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


class _PitchEncoder:
    """Stand-in voice model: the embedding is which of two pitches dominates the window."""
    same = 0.7

    def embed(self, windows):
        out = []
        for w in windows:
            spec = np.abs(np.fft.rfft(w))
            low, high = spec[150:250].sum(), spec[550:650].sum()  # 100-ish Hz bins vs 400-ish Hz (1.5 s windows)
            v = np.array([low, high, 0.05 * (low + high)])
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype=np.float32)


def _turns(sr, plan):
    t = np.arange(int(sr * sum(d for d, _ in plan))) / sr
    x = np.zeros_like(t)
    at = 0
    for dur, hz in plan:
        n = int(dur * sr)
        x[at:at + n] = 0.3 * np.sin(2 * np.pi * hz * t[at:at + n])
        at += n
    return x[None].astype(np.float32)


def test_speakers_split_by_turns_and_sum_exactly():
    from sam_audio_service.speakers import split_speakers
    sr = 16000
    x = _turns(sr, [(4, 110), (0.5, 0), (4, 400), (0.5, 0), (4, 110), (4, 400)])
    out = split_speakers(x, sr, _PitchEncoder())
    assert len(out.layers) == 2
    assert np.max(np.abs(sum(out.layers) - x)) < 1e-5
    p1, p2 = out.layers
    energy = lambda a, s, e: float(np.mean(a[0, int(s * sr):int(e * sr)] ** 2))  # noqa: E731
    # Person 1 speaks first (low voice); the middles of each turn belong wholly to one person.
    assert energy(p1, 1, 3) > 100 * energy(p2, 1, 3)
    assert energy(p2, 5.5, 7.5) > 100 * energy(p1, 5.5, 7.5)
    assert energy(p1, 10, 12) > 100 * energy(p2, 10, 12)
    assert energy(p2, 14, 16) > 100 * energy(p1, 14, 16)
    assert [s for *_, s in out.turns][:4] == [0, 1, 0, 1]


def test_one_speaker_stays_whole():
    from sam_audio_service.speakers import split_speakers
    sr = 16000
    x = _turns(sr, [(4, 110), (0.5, 0), (6, 110)])
    out = split_speakers(x, sr, _PitchEncoder())
    assert len(out.layers) == 1 and out.layers[0] is not None and np.allclose(out.layers[0], x)


def test_explode_names_people_and_keeps_the_sum():
    rng = np.random.default_rng(1)
    mix = rng.standard_normal((2, 4800)).astype(np.float32) * 0.1

    def two_people(d):
        return [d * 0.7, d * 0.3]

    out = explode(mix, 48000, ['people talking', 'heavy rain'], _fake_separate([]), lambda a: (a * 0.5).astype(np.float32),
                  speakers=two_people)
    assert [l.name for l in out.layers] == ['person 1', 'person 2', 'heavy rain']
    assert [l.prompt for l in out.layers[:2]] == ['people talking', 'people talking']
    assert np.max(np.abs(out.layers[0].samples + out.layers[1].samples - mix * 0.5)) < 1e-5
    assert np.max(np.abs(sum(l.samples for l in out.layers) + out.residual - mix)) < 1e-5


def test_explode_keeps_dialogue_when_speakers_fail_or_one_person():
    mix = np.ones((1, 100), dtype=np.float32)

    def boom(d):
        raise RuntimeError('no model')

    out = explode(mix, 48000, ['people talking'], _fake_separate([]), lambda a: a * 0.5, speakers=boom)
    assert [l.name for l in out.layers] == ['dialogue'] and any('one layer' in w for w in out.warnings)
    out = explode(mix, 48000, ['people talking'], _fake_separate([]), lambda a: a * 0.5, speakers=lambda d: [d])
    assert [l.name for l in out.layers] == ['dialogue'] and not out.warnings


def _clusters_with(cos, n=20):
    """Two clusters of unit embeddings whose centroids have cosine ``cos``."""
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([cos, np.sqrt(1 - cos ** 2), 0.0])
    emb = np.array([a] * n + [b] * n, dtype=np.float32)
    return emb, np.array([0] * n + [1] * n)


def test_merge_uses_pitch_between_maybe_and_same():
    from sam_audio_service.speakers import _merge_close
    emb, labels = _clusters_with(0.72)
    man_woman = np.array([140.0] * 20 + [290.0] * 20)
    one_man = np.array([140.0] * 20 + [150.0] * 20)
    # A man and a woman, fairly alike voices: two people.
    assert len(np.unique(_merge_close(emb, labels, 0.82, 0.60, man_woman))) == 2
    # One man, fairly alike: one person.
    assert len(np.unique(_merge_close(emb, labels, 0.82, 0.60, one_man))) == 1
    # Very alike merges whatever the pitch; unlike never merges.
    emb_hi, _ = _clusters_with(0.85)
    assert len(np.unique(_merge_close(emb_hi, labels, 0.82, 0.60, man_woman))) == 1
    emb_lo, _ = _clusters_with(0.4)
    assert len(np.unique(_merge_close(emb_lo, labels, 0.82, 0.60, one_man))) == 2
    # No pitch (unvoiced): only the strict line applies.
    assert len(np.unique(_merge_close(emb, labels, 0.82, 0.60, np.full(40, np.nan)))) == 2


def test_window_pitch_reads_a_voice():
    from sam_audio_service.speakers import window_pitch
    sr = 16000
    t = np.arange(sr * 3) / sr
    for hz in (110, 260):
        x = sum(np.sin(2 * np.pi * hz * k * t) / k for k in range(1, 6)).astype(np.float32)
        assert abs(window_pitch(x, [0], 150)[0] - hz) < 0.05 * hz
