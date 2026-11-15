"""The http adapter posts FLAC (not the float WAV) and asks for 16-bit stems: a split's transfer shrinks several-fold."""
import base64
import json
import subprocess

import httpx
import pytest


@pytest.fixture
def tone(tmp_path):
    path = tmp_path / 'tone.wav'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'sine=duration=0.5', '-ac', '2', '-ar', '48000', '-c:a', 'pcm_f32le', str(path)], check=True)
    return path


def pcm16(tmp_path, source):
    out = tmp_path / 'stem16.wav'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-c:a', 'pcm_s16le', str(out)], check=True)
    return base64.b64encode(out.read_bytes()).decode()


def capture(monkeypatch, tmp_path, tone, sent):
    import app.media as media
    monkeypatch.setenv('INFERENCE_ADAPTER', 'http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL', 'http://mock.invalid/separate')
    raw = pcm16(tmp_path, tone)
    def post(url, files, data, **kwargs):
        name, f, ctype = files['audio']
        body = f.read()
        sent.append({'name': name, 'type': ctype, 'magic': body[:4], 'size': len(body), 'metadata': json.loads(data['metadata'])})
        return httpx.Response(200, request=httpx.Request('POST', url), json={'stems': [{'name': 'Vocals', 'role': 'target', 'audio_base64': raw}]})
    monkeypatch.setattr(media.httpx, 'post', post)
    return media


@pytest.mark.parametrize('options', [{'mode': 'music_stems', 'stems': ['vocals']}, {'mode': 'prompted'}, {}])
def test_separate_uploads_flac_and_requests_16_bit_stems(monkeypatch, tmp_path, tone, options):
    sent = []
    media = capture(monkeypatch, tmp_path, tone, sent)
    payload = {'prompts': ['vocals'], 'options': dict(options)}
    stems = media.separate(tone, tmp_path / 'out', payload)
    assert sent[0]['name'] == 'audio.flac' and sent[0]['type'] == 'audio/flac' and sent[0]['magic'] == b'fLaC'
    assert sent[0]['size'] < tone.stat().st_size
    assert sent[0]['metadata']['options'] == {**options, 'output_format': 'pcm_s16le'}
    assert payload['options'] == options  # The caller's payload is not mutated.
    # Stored as received; 16-bit keeps the same length and rate as the float source.
    from app.core import probe
    info, source = probe(stems[0]['path']), probe(tone)
    assert stems[0]['path'].read_bytes()[:4] == b'RIFF'
    assert (info['sample_rate'], info['channels']) == (source['sample_rate'], source['channels'])
    assert abs(info['duration'] - source['duration']) < 1e-6
    samples, _ = media.read_wav(stems[0]['path'])
    assert samples.shape[0] == media.read_wav(tone)[0].shape[0]


def test_an_explicit_output_format_wins(monkeypatch, tmp_path, tone):
    sent = []
    media = capture(monkeypatch, tmp_path, tone, sent)
    media.separate(tone, tmp_path / 'out', {'prompts': ['x'], 'options': {'mode': 'prompted', 'output_format': 'pcm_f32le'}})
    assert sent[0]['metadata']['options']['output_format'] == 'pcm_f32le'


def test_route_and_timing_calls_upload_flac(monkeypatch, tmp_path, tone):
    import app.media as media
    import app.sfx_insert as sfx_insert
    monkeypatch.setenv('INFERENCE_ADAPTER', 'http'); monkeypatch.setenv('SAM_AUDIO_SERVICE_URL', 'http://mock.invalid/separate')
    seen = []
    def post(url, files, data, **kwargs):
        name, f, ctype = files['audio']; seen.append((name, ctype, f.read()[:4]))
        return httpx.Response(200, request=httpx.Request('POST', url), json={'routing': {'mode': 'prompted'}, 'words': []})
    monkeypatch.setattr(httpx, 'post', post); monkeypatch.setattr(media.httpx, 'post', post)
    media.route(tone, 'audio')
    sfx_insert._service_post('/v1/words', tone, {'word_timestamps': True}, 5)
    assert seen == [('audio.flac', 'audio/flac', b'fLaC')] * 2


def test_upload_falls_back_to_the_original_when_ffmpeg_cannot_encode(tmp_path):
    from app.media import service_upload
    junk = tmp_path / 'junk.wav'; junk.write_bytes(b'not audio')
    with service_upload(junk) as (name, f, ctype):
        assert (name, ctype, f.read()) == ('audio.wav', 'audio/wav', b'not audio')


def float_wav(tmp_path, peak):
    import numpy as np
    raw = tmp_path / f'raw{peak}.f32'; wav = tmp_path / f'peak{peak}.wav'
    raw.write_bytes((peak * np.sin(2 * np.pi * 440 * np.arange(24000) / 48000)).astype('<f4').tobytes())
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', '48000', '-ac', '1', '-i', str(raw), '-c:a', 'pcm_f32le', str(wav)], check=True)
    return wav


def test_upload_sends_the_float_wav_when_the_source_peaks_above_full_scale(tmp_path):
    from app.media import service_upload, source_peak
    hot = float_wav(tmp_path, 1.4)
    assert source_peak(hot) == pytest.approx(1.4, abs=1e-3)
    with service_upload(hot) as (name, f, ctype):
        assert (name, ctype) == ('audio.wav', 'audio/wav') and f.read() == hot.read_bytes()  # FLAC would clip 1.40 to 1.00
    safe = float_wav(tmp_path, 0.9)
    with service_upload(safe) as (name, f, ctype):
        assert (name, ctype, f.read()[:4]) == ('audio.flac', 'audio/flac', b'fLaC')
