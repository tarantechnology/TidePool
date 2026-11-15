import json
import subprocess

import numpy as np
import pytest
import soundfile as sf

from app import moonpull as mp
from tests.test_app import client, wait  # noqa: F401  (fixture)


def tone(path, seconds=4.0, freq=220, sr=48000, channels=1):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', f'sine=frequency={freq}:duration={seconds}',
                    '-ar', str(sr), '-ac', str(channels), str(path)], check=True)
    return path


def boom_track(path, sr=48000, seconds=6.0, boom_at=4.0):
    """Quiet noise with one loud decaying burst: a stand-in for a thunder layer."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.01, int(seconds * sr)).astype(np.float32)
    n = int(1.0 * sr)
    burst = rng.normal(0, 0.5, n).astype(np.float32) * np.exp(-np.linspace(0, 4, n)).astype(np.float32)
    i = int(boom_at * sr); x[i:i + n] += burst
    sf.write(path, x, sr, subtype='FLOAT')
    return path


def test_effect_over_several_ranges_leaves_the_rest_alone(tmp_path):
    src = tone(tmp_path / 'a.wav')
    out = mp.apply_effect_ranges(src, 'muffle', [(0.5, 1.0), (2.5, 3.0)], 1.0)
    a, sr = sf.read(src, always_2d=True); b, _ = sf.read(out, always_2d=True)
    assert len(a) == len(b)
    for lo, hi in [(0, 0.45), (1.05, 2.45), (3.05, 4.0)]:
        sl = slice(int(lo * sr), int(hi * sr))
        assert np.allclose(b[sl, 0], a[sl, 0], atol=1e-4)
    for lo, hi in [(0.6, 0.9), (2.6, 2.9)]:
        sl = slice(int(lo * sr), int(hi * sr))
        assert np.sqrt(np.mean(b[sl, 0] ** 2)) < 0.8 * np.sqrt(np.mean(a[sl, 0] ** 2))


def test_ranges_convert_to_stem_time_and_clip():
    assert mp.to_stem_ranges([(1.0, 2.0), (9.0, 12.0)], 0.5, 10.0) == [(0.5, 1.5), (8.5, 10.0)]
    with pytest.raises(ValueError): mp.to_stem_ranges([(20.0, 21.0)], 0.0, 10.0)
    with pytest.raises(ValueError): mp.to_stem_ranges([(2.0, 1.0)], 0.0, 10.0)


def test_loudest_window_finds_the_boom(tmp_path):
    a, b = mp.loudest_window(boom_track(tmp_path / 't.wav'), 1.5)
    assert 3.7 <= a <= 4.0 and b - a == pytest.approx(1.5, abs=0.01)


def test_echo_places_the_motif_at_each_time_and_never_clips(tmp_path):
    thunder = boom_track(tmp_path / 't.wav')
    out = mp.apply_echo(thunder, thunder, (4.0, 5.0), [0.5, 2.0], gain=1.0)
    a, sr = sf.read(thunder, always_2d=True); b, _ = sf.read(out, always_2d=True)
    assert len(a) == len(b)
    rms = lambda x, lo, hi: float(np.sqrt(np.mean(x[int(lo * sr):int(hi * sr), 0] ** 2)))
    assert rms(b, 0.5, 0.8) > 10 * rms(a, 0.5, 0.8)
    assert rms(b, 2.0, 2.3) > 10 * rms(a, 2.0, 2.3)
    # Untouched between the echoes and before the first one.
    assert np.allclose(b[:int(0.49 * sr)], a[:int(0.49 * sr)], atol=1e-6)
    assert np.allclose(b[int(1.55 * sr):int(1.99 * sr)], a[int(1.55 * sr):int(1.99 * sr)], atol=1e-6)
    assert np.abs(b).max() < 1.0


def test_echo_onto_another_layer_matches_rate_and_channels(tmp_path):
    thunder = boom_track(tmp_path / 't.wav', sr=44100)
    dialogue = tone(tmp_path / 'd.wav', seconds=3, sr=48000, channels=2)
    out = mp.apply_echo(dialogue, thunder, (4.0, 5.0), [1.0])
    info = sf.info(out)
    assert info.samplerate == 48000 and info.channels == 2 and info.frames == sf.info(dialogue).frames
    with pytest.raises(ValueError): mp.apply_echo(dialogue, thunder, (4.0, 5.0), [10.0])
    with pytest.raises(ValueError): mp.apply_echo(dialogue, thunder, (6.5, 9.0), [1.0])
    # A motif that runs to the very end of its layer (the loudest boom often does) is clipped, not refused.
    a, b = mp.loudest_window(thunder, 2.0)
    assert mp.apply_echo(dialogue, thunder, (5.0, 6.0004), [1.0]).exists()


def test_soft_limit_is_transparent_below_the_knee():
    x = np.array([0.1, -0.5, 0.84, 1.5, -3.0], dtype=np.float32)
    y = mp.soft_limit(x)
    assert np.allclose(y[:3], x[:3]) and np.all(np.abs(y) < 1.0) and y[3] > 0.85 and y[4] < -0.85


def test_parse_transcript_offsets_and_skips_bad_items():
    data = {'text': ' Here I am. ', 'segments': [{'start': 1.0, 'end': 2.5, 'text': ' Here I am.'}, {'start': 'x'}],
            'words': [{'start': 1.0, 'end': 1.3, 'word': 'Here'}, {'start': 3, 'end': 2, 'word': 'bad'}]}
    t = mp.parse_transcript(data, 0.5)
    assert t['lines'] == [{'start': 1.5, 'end': 3.0, 'text': 'Here I am.'}]
    assert t['words'] == [{'start': 1.5, 'end': 1.8, 'text': 'Here'}]


def project_with_layers(client, tmp_path):
    p = client.post('/api/projects').json(); pid = p['project_id']; h = {'Authorization': 'Bearer ' + p['access_token']}
    audio = boom_track(tmp_path / 'mix.wav', seconds=6.0, boom_at=4.0)
    with audio.open('rb') as f: assert client.post(f'/api/projects/{pid}/upload', headers=h, files={'file': ('mix.wav', f, 'audio/wav')}).status_code == 200
    job = wait(client, '/api/jobs/' + client.post(f'/api/projects/{pid}/process', headers=h, json={'prompts': ['thunder', 'people talking']}).json()['job_id'], h)
    assert job['status'] == 'completed', job
    stems = {s['name']: s for s in client.get(f'/api/projects/{pid}/stems', headers=h).json()}
    return pid, h, stems


def test_echo_effect_revert_round_trip(client, tmp_path):
    pid, h, stems = project_with_layers(client, tmp_path)
    thunder = next(s for s in stems.values() if s['name'] != 'residual')
    sid, rev = thunder['id'], thunder['revision']
    base = f'/api/projects/{pid}/stems/{sid}'
    original = client.get(base + '/audio', headers=h).content

    motif = client.get(base + '/moonpull/motif', headers=h).json()
    assert motif['end'] > motif['start']

    r = client.post(base + '/moonpull/echo', headers=h, json={'source_stem_id': sid, 'times': [0.5, 2.0], 'expected_revision': rev, 'rule': 'Storm answers a lie'})
    assert r.status_code == 200, r.text
    echoed = r.json(); assert echoed['revision'] == rev + 1
    assert client.get(base + '/audio', headers=h).content != original

    # A stale revision is refused, and so is a revert of someone else's change.
    assert client.post(base + '/moonpull/effect', headers=h, json={'effect': 'muffle', 'ranges': [[1, 2]], 'expected_revision': rev}).status_code == 409
    r = client.post(base + '/moonpull/effect', headers=h, json={'effect': 'muffle', 'ranges': [[1, 2], [3, 3.5]], 'expected_revision': rev + 1})
    assert r.status_code == 200, r.text
    
    muffled = r.json()
    # Only the latest change can be reverted; fader edits in between do not matter.
    r = client.post(base + '/moonpull/revert', headers=h, json={'file': echoed['moonpull_file']})
    assert r.status_code == 409 and 'changed after' in r.json()['detail']
    assert client.patch(base, headers=h, json={'volume': 0.8}).status_code == 200
    assert client.post(base + '/moonpull/revert', headers=h, json={'file': muffled['moonpull_file']}).status_code == 200
    assert client.post(base + '/moonpull/revert', headers=h, json={'file': echoed['moonpull_file']}).status_code == 200
    assert client.get(base + '/audio', headers=h).content == original
    r = client.post(base + '/moonpull/revert', headers=h, json={'file': echoed['moonpull_file']})
    assert r.status_code == 409


def test_moonpull_validation_and_auth(client, tmp_path):
    pid, h, stems = project_with_layers(client, tmp_path)
    sid = next(iter(stems.values()))['id']
    base = f'/api/projects/{pid}/stems/{sid}/moonpull'
    assert client.post(base + '/effect', json={'effect': 'muffle', 'ranges': [[0, 1]]}).status_code == 401
    assert client.post(base + '/effect', headers=h, json={'effect': 'shout', 'ranges': [[0, 1]]}).status_code == 422
    assert client.post(base + '/effect', headers=h, json={'effect': 'muffle', 'ranges': []}).status_code == 422
    assert client.post(base + '/effect', headers=h, json={'effect': 'muffle', 'ranges': [[50, 60]]}).status_code == 422
    assert client.post(base + '/echo', headers=h, json={'source_stem_id': 'missing', 'times': [1]}).status_code == 404


def test_transcript_route_uses_openai_and_caches(client, tmp_path, monkeypatch):
    pid, h, stems = project_with_layers(client, tmp_path)
    sid = next(iter(stems.values()))['id']
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    calls = []

    class Response:
        status_code = 200; is_error = False
        def json(self): return {'text': 'Here I am.', 'segments': [{'start': 0.2, 'end': 1.1, 'text': 'Here I am.'}], 'words': [{'start': 0.2, 'end': 0.5, 'word': 'Here'}]}

    def fake_post(url, **kwargs):
        calls.append((url, kwargs['data']))
        return Response()
    monkeypatch.setattr(mp.httpx, 'post', fake_post)
    url = f'/api/projects/{pid}/stems/{sid}/moonpull/transcript'
    r = client.post(url, headers=h); assert r.status_code == 200, r.text
    assert r.json()['lines'] == [{'start': 0.2, 'end': 1.1, 'text': 'Here I am.'}]
    assert calls[0][0].endswith('/audio/transcriptions') and calls[0][1]['response_format'] == 'verbose_json'
    assert client.post(url, headers=h).status_code == 200 and len(calls) == 1

    monkeypatch.delenv('OPENAI_API_KEY')
    other = [s for s in stems.values() if s['id'] != sid][0]['id']
    r = client.post(f'/api/projects/{pid}/stems/{other}/moonpull/transcript', headers=h)
    assert r.status_code == 503 and 'not configured' in r.json()['detail']


def test_place_sound_restarts_loops_and_stays_silent_between(tmp_path):
    sound = tone(tmp_path / 's.wav', seconds=0.5, sr=44100)
    out = mp.place_sound(sound, [(1.0, 2.2), (4.0, 4.3)], 6.0, tmp_path / 'layer.wav')
    b, sr = sf.read(out, always_2d=True)
    assert sr == 48000 and b.shape == (6 * 48000, 2)
    rms = lambda lo, hi: float(np.sqrt(np.mean(b[int(lo * sr):int(hi * sr)] ** 2)))
    assert rms(0, 0.99) == 0 and rms(2.21, 3.99) == 0 and rms(4.31, 6) == 0
    # Loops to fill the 1.2 s range from a 0.5 s sound, and plays again at the second moment.
    assert rms(1.6, 1.7) > 0.05 and rms(4.1, 4.15) > 0.05
    assert np.abs(b).max() < 1.0
    with pytest.raises(ValueError): mp.place_sound(sound, [(7.0, 8.0)], 6.0, tmp_path / 'x.wav')


def test_layer_route_adds_a_layer_that_deletes_cleanly(client, tmp_path):
    pid, h, stems = project_with_layers(client, tmp_path)
    music = tone(tmp_path / 'theme.wav', seconds=20, freq=330)
    with music.open('rb') as f:
        assert client.post(f'/api/projects/{pid}/sound-effects/upload', headers=h, files={'file': ('theme.wav', f, 'audio/wav')}).status_code == 422
    with music.open('rb') as f:
        r = client.post(f'/api/projects/{pid}/sound-effects/upload?long=true', headers=h, files={'file': ('theme.wav', f, 'audio/wav')})
    assert r.status_code == 200, r.text
    aid = r.json()['asset_id']
    url = f'/api/projects/{pid}/moonpull/layer'
    assert client.post(url, json={'asset_id': aid, 'name': 'Theme', 'ranges': [[0.5, 1.5]]}).status_code == 401
    assert client.post(url, headers=h, json={'asset_id': '0' * 32, 'name': 'Theme', 'ranges': [[0.5, 1.5]]}).status_code == 404
    assert client.post(url, headers=h, json={'asset_id': aid, 'name': 'Theme', 'ranges': [[2.0, 1.0]]}).status_code == 422
    r = client.post(url, headers=h, json={'asset_id': aid, 'name': 'Theme', 'ranges': [[0.5, 1.5], [2.5, 3.0]], 'rule': 'Hero on screen'})
    assert r.status_code == 200, r.text
    added = r.json()
    assert added['stem']['label'] == 'Theme' and added['stem']['role'] == 'added' and added['stem']['start_offset'] == 0
    job = client.get('/api/jobs/' + added['job_id'], headers=h).json()
    assert added['stem']['id'] in {s['id'] for s in job['stems']}
    audio = client.get(f"/api/projects/{pid}/stems/{added['stem']['id']}/audio", headers=h)
    assert audio.status_code == 200
    names = {s['name'] for s in client.get(f'/api/projects/{pid}/stems', headers=h).json()}
    assert 'Theme' in names and len(names) == len(stems) + 1
    r = client.delete(f"/api/projects/{pid}/stems/{added['stem']['id']}?fold=false", headers=h)
    assert r.status_code == 200
    assert {s['name'] for s in client.get(f'/api/projects/{pid}/stems', headers=h).json()} == set(stems)
