"""/v1/words, /v1/enhance, /v1/beats with mock/fake models (CPU, no weights, no downloads)."""
import base64
import json
import subprocess
import sys
import threading
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sam_audio_service.api import create_app
from sam_audio_service.audio_io import decode, encode_wav_pcm16
from sam_audio_service.config import Settings, load_settings
from sam_audio_service.speech import SpeechTools
from sam_audio_service.speech.beats import BeatTracker, MockBeatTracker, summarise
from sam_audio_service.speech.lexicon import ProfanityList, find_fillers, find_pauses, find_profanity, normalize
from sam_audio_service.speech.words import FILLER_PROMPT, FasterWhisperTranscriber, Transcript, Wav2Vec2Aligner, ctc_viterbi


def bursts(onsets, length=0.25, seconds=None, rate=48000, channels=2, freq=300.0, noise=0.0):
    """Tone bursts at ``onsets`` (s); the mock transcriber turns each one into a word."""
    seconds = seconds or (max(onsets) + length + 0.5)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    x = np.zeros(n)
    for o in onsets:
        a, b = int(o * rate), int((o + length) * rate)
        x[a:b] = 0.5 * np.sin(2 * np.pi * freq * t[a:b])
    if noise:
        x = x + noise * np.random.default_rng(1).standard_normal(n)
    return encode_wav_pcm16(np.stack([x] * channels).astype(np.float32), rate)


ONSETS = [0.5 + 0.5 * i for i in range(12)]
ONSETS[6:] = [o + 1.0 for o in ONSETS[6:]]  # a 1.25 s pause after word 5


def post(client, path, audio, metadata=None):
    return client.post(path, files={"audio": ("a.wav", audio, "audio/wav")}, data={"metadata": json.dumps(metadata or {})})


@pytest.fixture
def speech_settings(settings, monkeypatch):
    monkeypatch.setenv("AUDIO_WORDS_BACKEND", "mock")
    monkeypatch.setenv("AUDIO_ENHANCE_BACKEND", "mock")
    monkeypatch.setenv("AUDIO_BEATS_BACKEND", "mock")
    return load_settings()


@pytest.fixture
def speech_client(speech_settings):
    with TestClient(create_app(speech_settings)) as c:
        yield c


# ----------------------------------------------------------------------------------- config
def test_speech_tools_are_off_by_default(client, wav_path):
    caps = client.get("/capabilities").json()
    for block in ("words", "enhance", "beats"):
        assert caps[block] == {"available": False, "backend": None, "model_id": None}
    for path in ("/v1/words", "/v1/enhance", "/v1/beats"):
        r = post(client, path, wav_path.read_bytes())
        assert r.status_code == 400, r.text
        assert r.json()["error"]["code"] == "UNSUPPORTED_OPERATION"
    assert client.get("/health").json()["speech"] == {"words": None, "aligner": None, "enhance": None, "beats": None}


@pytest.mark.parametrize("env,value", [
    ("AUDIO_WORDS_BACKEND", "whisperx"), ("AUDIO_WORDS_ALIGN", "mfa"), ("AUDIO_WORDS_BEAM_SIZE", "0"), ("AUDIO_WORDS_COMPUTE_TYPE", "fp8"),
    ("AUDIO_WORDS_PROFANITY_FILE", "/does/not/exist.txt"), ("AUDIO_WORDS_MIN_PAUSE_SECONDS", "0"), ("AUDIO_ENHANCE_BACKEND", "rnnoise"),
    ("AUDIO_ENHANCE_ATTEN_LIMIT_DB", "-3"), ("AUDIO_ENHANCE_MODEL", "Demucs"), ("AUDIO_ENHANCE_CHUNK_SECONDS", "2"), ("AUDIO_BEATS_BACKEND", "madmom"),
    ("AUDIO_BEATS_FALLBACK", "aubio"), ("AUDIO_SPEECH_DEVICE", "tpu"),
])
def test_invalid_speech_settings_are_rejected(monkeypatch, env, value):
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    monkeypatch.setenv(env, value)
    with pytest.raises(ValueError, match=env):
        load_settings()


def test_speech_defaults(monkeypatch):
    for k in ("AUDIO_WORDS_MODEL", "AUDIO_WORDS_ALIGN", "AUDIO_ENHANCE_ATTEN_LIMIT_DB", "AUDIO_BEATS_MODEL"):
        monkeypatch.delenv(k, raising=False)
    s = Settings(backend="mock")
    assert (s.words_backend, s.words_model, s.words_align, s.words_vad) == ("off", "large-v3-turbo", "wav2vec2", True)
    assert (s.enhance_backend, s.enhance_model, s.enhance_atten_limit_db) == ("off", "DeepFilterNet3", None)
    assert (s.beats_backend, s.beats_model, s.beats_fallback) == ("off", "final0", "librosa")


# ----------------------------------------------------------------------------------- /v1/words
def test_words_endpoint_returns_timed_words_fillers_profanity_and_pauses(speech_client):
    r = post(speech_client, "/v1/words", bursts(ONSETS), {"request_id": "w1", "asset_id": "a1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["request_id"] == "w1" and body["model"]["is_mock"] is True and any("MOCK" in w for w in body["warnings"])
    words = body["words"]
    assert [w["word"] for w in words] == ["Um", "so", "what", "the", "fuck,", "like,", "is", "this", "uh", "thing", "damn", "what"]
    for w, onset in zip(words, ONSETS):
        assert abs(w["start"] - onset) <= 0.02 and abs(w["end"] - (onset + 0.25)) <= 0.02
    assert words[4]["norm"] == "fuck" and words[5]["norm"] == "like"
    assert body["fillers"] == [0, 5, 8]
    assert body["profanity"] == [4, 10]
    assert body["pauses"] == [{"start": words[5]["end"], "end": words[6]["start"], "after_word": 5}]
    assert body["segments"][0]["word_start"] == 0 and body["segments"][0]["word_end"] == 12
    assert body["source"] == {"asset_id": "a1", "duration": pytest.approx(ONSETS[-1] + 0.75, abs=1e-3), "sample_rate": 48000, "channels": 2}
    assert body["aligned"] is False and body["profanity_list"]["source"] == "builtin:profanity_en.txt"


def test_words_extra_profanity_min_pause_and_validation(speech_client):
    r = post(speech_client, "/v1/words", bursts(ONSETS), {"extra_profanity": ["what"], "min_pause_seconds": 2.0, "align": True})
    body = r.json()
    assert r.status_code == 200, r.text
    assert body["profanity"] == [2, 4, 10, 11]
    assert body["pauses"] == []
    assert any("forced alignment is not configured" in w for w in body["warnings"])
    for bad in ({"language": "english!"}, {"min_pause_seconds": 0}, {"extra_profanity": "fuck"}):
        r = post(speech_client, "/v1/words", bursts(ONSETS), bad)
        assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_REQUEST", bad
    r = speech_client.post("/v1/words", files={"audio": ("a.wav", bursts(ONSETS), "audio/wav")}, data={"metadata": "{not json"})
    assert r.status_code == 400


def test_words_rejects_empty_and_too_long_audio(speech_client, monkeypatch):
    r = speech_client.post("/v1/words", files={"audio": ("a.wav", b"", "audio/wav")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_MEDIA"
    speech_client.app.state.speech.settings.max_duration_seconds = 1.0
    r = post(speech_client, "/v1/words", bursts(ONSETS))
    assert r.status_code == 400 and r.json()["error"]["code"] == "UNSUPPORTED_OPERATION"


def test_capabilities_and_health_describe_speech_tools(speech_client):
    caps = speech_client.get("/capabilities").json()
    assert caps["words"]["available"] and caps["words"]["backend"] == "mock" and caps["words"]["model_id"] == "mock-words"
    assert caps["words"]["profanity_list"]["entries"] > 50 and caps["words"]["align"]["available"] is False
    assert caps["enhance"]["available"] and caps["enhance"]["model_sample_rate"] == 48000
    assert caps["beats"]["available"] and caps["beats"]["downbeats"] is True
    health = speech_client.get("/health").json()["speech"]
    assert health["words"]["backend"] == "mock" and health["words"]["load_error"] is None


def test_speech_requests_do_not_wait_for_the_separation_lock(speech_client):
    engine = speech_client.app.state.engine
    results = {}

    def call():
        results["words"] = post(speech_client, "/v1/words", bursts(ONSETS[:3])).status_code
        results["beats"] = post(speech_client, "/v1/beats", bursts([0.5, 1.0])).status_code

    with engine._lock:  # a long separation in progress
        t = threading.Thread(target=call)
        t.start()
        t.join(timeout=30)
        assert not t.is_alive(), "speech request blocked behind the separation lock"
    assert results == {"words": 200, "beats": 200}


def test_model_load_failure_is_missing_model_and_retried(speech_settings):
    class Flaky(MockBeatTracker):
        attempts = 0

        def load(self):
            Flaky.attempts += 1
            if Flaky.attempts == 1:
                raise RuntimeError("weights not reachable")
            super().load()

    speech = SpeechTools(speech_settings, beat_tracker=Flaky())
    with TestClient(create_app(speech_settings, speech=speech)) as c:  # startup preload fails once
        assert c.get("/health").json()["speech"]["beats"]["load_error"] == "RuntimeError: weights not reachable"
        r = post(c, "/v1/beats", bursts([0.5]))
        assert r.status_code == 200, r.text
        assert c.get("/health").json()["speech"]["beats"]["load_error"] is None


def test_out_of_memory_maps_to_insufficient_gpu_memory(speech_settings):
    class Oom(MockBeatTracker):
        def track(self, samples, sample_rate):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    with TestClient(create_app(speech_settings, speech=SpeechTools(speech_settings, beat_tracker=Oom()))) as c:
        r = post(c, "/v1/beats", bursts([0.5]))
    assert r.status_code == 503 and r.json()["error"]["code"] == "INSUFFICIENT_GPU_MEMORY" and r.json()["error"]["retryable"] is True


# ----------------------------------------------------------------------------------- /v1/enhance
def _samples(body):
    return decode(data=base64.b64decode(body["audio_base64"]))


def test_enhance_keeps_length_channels_and_removes_noise(speech_client):
    audio = bursts([0.5, 1.5], seconds=3.0, noise=0.02)
    r = post(speech_client, "/v1/enhance", audio, {"request_id": "e1"})
    assert r.status_code == 200, r.text
    body = r.json()
    inp, out = decode(data=audio), _samples(body)
    assert (body["sample_rate"], body["channels"], body["sample_count"]) == (48000, 2, inp.frames)
    assert out.samples.shape == inp.samples.shape and body["format"] == "wav/pcm_f32le"
    quiet = slice(int(2.2 * 48000), int(2.9 * 48000))  # noise only
    assert np.sqrt(np.mean(out.samples[:, quiet] ** 2)) < 0.3 * np.sqrt(np.mean(inp.samples[:, quiet] ** 2))
    tone = slice(int(0.55 * 48000), int(0.7 * 48000))
    assert np.corrcoef(out.samples[0, tone], inp.samples[0, tone])[0, 1] > 0.99
    assert body["levels"]["rms_dbfs_out"] < body["levels"]["rms_dbfs_in"] and body.get("atten_lim_db") is None
    # A 6 dB limit removes at most ~6 dB of the noise.
    gentle = _samples(post(speech_client, "/v1/enhance", audio, {"atten_lim_db": 6}).json()).samples
    ratio = np.sqrt(np.mean(gentle[:, quiet] ** 2) / np.mean(inp.samples[:, quiet] ** 2))
    assert 0.45 < ratio < 0.6


def test_enhance_preserves_source_rate_mono_and_formats(speech_client, settings):
    audio = bursts([0.3], seconds=1.0, rate=44100, channels=1, noise=0.01)
    body = post(speech_client, "/v1/enhance", audio, {"output_format": "pcm_s16le"}).json()
    assert (body["sample_rate"], body["channels"], body["sample_count"]) == (44100, 1, 44100) and body["format"] == "wav/pcm_s16le"
    assert body["source"]["sample_rate"] == 44100
    body48 = post(speech_client, "/v1/enhance", audio, {"preserve_sample_rate": False}).json()
    assert (body48["sample_rate"], body48["sample_count"]) == (48000, 48000)
    ref = post(speech_client, "/v1/enhance", audio, {"return_mode": "file", "request_id": "enh-file"}).json()
    assert "audio_base64" not in ref and ref["audio_ref"].endswith("/enh-file/enhanced.wav")
    path = settings.output_root / "enh-file" / "enhanced.wav"
    assert decode(path=path, target_rate=44100).frames == 44100
    r = post(speech_client, "/v1/enhance", audio, {"atten_lim_db": 0})
    assert r.status_code == 400


def test_enhance_rejects_wrong_output_shape(speech_settings):
    class Bad(SpeechTools):
        pass

    from sam_audio_service.speech.enhance import MockEnhancer

    class Short(MockEnhancer):
        def enhance(self, samples, atten_lim_db):
            return samples[:, :-10]

    with TestClient(create_app(speech_settings, speech=Bad(speech_settings, enhancer=Short()))) as c:
        r = post(c, "/v1/enhance", bursts([0.2], seconds=0.5))
    assert r.status_code == 500 and r.json()["error"]["code"] == "INFERENCE_FAILURE"


def test_deepfilternet_chunking_stitches_exactly():
    from sam_audio_service.speech.enhance import DeepFilterNetEnhancer

    e = DeepFilterNetEnhancer(chunk_seconds=5.0)
    e.model = object()  # loaded
    e._run = lambda x, lim: x * 0.5  # a causal-free stand-in: chunk borders must not show
    x = np.random.default_rng(0).standard_normal((2, 48000 * 12 + 123)).astype(np.float32)
    np.testing.assert_allclose(e.enhance(x, None), x * 0.5)


# ----------------------------------------------------------------------------------- /v1/beats
def test_beats_endpoint_mock_grid(speech_client):
    audio = bursts([0.5], seconds=10.0, length=9.0)
    r = post(speech_client, "/v1/beats", audio, {"request_id": "b1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tempo_bpm"] == 120.0 and body["beats_per_bar"] == 4 and body["downbeats_available"] is True
    assert body["beats"][0] == pytest.approx(0.5, abs=0.01) and body["downbeats"] == body["beats"][::4]
    assert all(0 <= b <= 10.0 for b in body["beats"]) and body["source"]["duration"] == pytest.approx(10.0, abs=1e-3)


def test_beats_fall_back_when_the_model_is_unavailable(speech_settings):
    class Broken(MockBeatTracker):
        name = "beat_this"

        def load(self):
            raise ValueError("Could not load the checkpoint given the provided name")

    class Fallback(MockBeatTracker):
        name = "librosa"
        has_downbeats = False

        def track(self, samples, sample_rate):
            return {"beats": [0.5, 1.0, 1.5], "downbeats": []}

    speech = SpeechTools(speech_settings, beat_tracker=Broken(), beat_fallback=Fallback())
    with TestClient(create_app(speech_settings, speech=speech)) as c:
        body = post(c, "/v1/beats", bursts([0.5], seconds=2.0)).json()
    assert body["beats"] == [0.5, 1.0, 1.5] and body["downbeats"] == [] and body["downbeats_available"] is False
    assert body["model"]["backend"] == "librosa" and body["tempo_bpm"] == 120.0
    assert any("beat_this unavailable" in w for w in body["warnings"])


def test_summarise_tempo_and_meter():
    beats = list(np.arange(0, 12, 0.5))
    assert summarise(beats, beats[::3]) == {"tempo_bpm": 120.0, "beats_per_bar": 3}
    assert summarise([1.0], []) == {"tempo_bpm": None, "beats_per_bar": None}


# ----------------------------------------------------------------------------------- lexicon
def test_profanity_matching_is_word_bounded():
    p = ProfanityList.load()
    flagged = ["Fuck,", "FUCKING", "motherfucker", "mother-fucker", "Bullshit.", "bitch's", "f***", "sh*t", "\"asshole\"", "damn!", "Hell"]
    clean = ["assume", "class", "Scunthorpe", "hello", "shell", "cocktail", "passage", "assessed", "Dickens", "therapist", "f*", "luck"]
    assert [w for w in flagged if not p.matches(w)] == []
    assert [w for w in clean if p.matches(w)] == []


def test_profanity_file_format_and_override(tmp_path, monkeypatch):
    path = tmp_path / "list.txt"
    path.write_text("# comment\n\nheck\nprefix:frick\n")
    p = ProfanityList.load(path)
    assert p.matches("Heck!") and p.matches("fricking") and not p.matches("fuck") and len(p) == 2
    assert p.with_extra(["darn"]).matches("darn") and not p.matches("darn")
    monkeypatch.setenv("AUDIO_BACKEND", "mock")
    monkeypatch.setenv("AUDIO_WORDS_BACKEND", "mock")
    monkeypatch.setenv("AUDIO_WORDS_PROFANITY_FILE", str(path))
    s = load_settings()
    with TestClient(create_app(s)) as c:
        body = post(c, "/v1/words", bursts(ONSETS)).json()
    assert body["profanity"] == [] and body["profanity_list"] == {"source": str(path), "entries": 2}


def _w(words, gaps=None):
    out, t = [], 0.0
    for i, w in enumerate(words):
        out.append({"word": w, "start": t, "end": t + 0.3})
        t += 0.3 + (gaps[i] if gaps else 0.05)
    return out


def test_fillers_are_conservative():
    assert find_fillers(_w(["Um,", "I", "uh", "think", "erm", "so."])) == [0, 2, 4]
    # "like" only between commas/pauses, never as a verb or preposition.
    assert find_fillers(_w(["it", "was,", "like,", "huge"])) == [2]
    assert find_fillers(_w(["I", "like", "this", "song"])) == []
    assert find_fillers(_w(["looks", "like", "rain"])) == []
    assert find_fillers(_w(["it", "was", "like", "huge"], gaps=[0.05, 0.4, 0.4, 0])) == [2]
    assert find_fillers(_w(["hmm", "oh", "ah", "mm"])) == []
    assert find_profanity(_w(["what", "the", "fuck"]), ProfanityList.load()) == [2]
    assert find_pauses(_w(["a", "b", "c"], gaps=[1.0, 0.1, 0]), 0.7) == [{"start": 0.3, "end": 1.3, "after_word": 0}]
    assert normalize("“Don’t!”") == "don't"


# ----------------------------------------------------------------------------------- whisper + alignment
def test_faster_whisper_transcriber_parses_segments_and_biases_english_fillers(monkeypatch):
    calls = {}

    class W:
        def __init__(self, word, start, end, p):
            self.word, self.start, self.end, self.probability = word, start, end, p

    class Seg:
        def __init__(self, start, end, text, words):
            self.start, self.end, self.text, self.words = start, end, text, words

    class Model:
        def __init__(self, name, **kw):
            calls["init"] = (name, kw)

        def detect_language(self, audio, vad_filter=True):
            return "en", 0.97, []

        def transcribe(self, audio, **kw):
            calls["transcribe"] = kw
            return iter([Seg(0.0, 1.2, " Um, hello there.", [W(" Um,", 0.1, 0.3, 0.8), W(" hello", 0.4, 0.7, 0.95), W(" there.", 0.75, 1.1, 0.9)]),
                         Seg(1.2, 1.5, " ", [W(" ", 1.2, 1.3, 0.1)])]), types.SimpleNamespace(language="en", language_probability=0.99)

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = Model
    fake.BatchedInferencePipeline = lambda m: pytest.fail("batch_size=1 must not build the batched pipeline")
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    t = FasterWhisperTranscriber("large-v3-turbo", device="cpu", batch_size=1)
    out = t.transcribe(np.zeros(16000, np.float32), None, True)
    assert calls["init"] == ("large-v3-turbo", {"device": "cpu", "device_index": 0, "compute_type": "int8"})
    assert calls["transcribe"]["initial_prompt"] == FILLER_PROMPT and calls["transcribe"]["word_timestamps"] is True and calls["transcribe"]["vad_filter"] is True
    assert [w["word"] for w in out.words] == ["Um,", "hello", "there."] and out.words[0]["segment"] == 0
    assert out.segments == [{"id": 0, "start": 0.0, "end": 1.2, "text": "Um, hello there.", "word_start": 0, "word_end": 3}]
    assert out.language == "en" and out.language_probability == 0.97
    t.transcribe(np.zeros(16000, np.float32), "es", True)
    assert calls["transcribe"]["initial_prompt"] is None and calls["transcribe"]["language"] == "es"


def test_ctc_viterbi_recovers_a_known_path():
    # Vocabulary: 0 blank, 1 A, 2 B. Frames: _ A A _ B B B _ A _
    path = [0, 1, 1, 0, 2, 2, 2, 0, 1, 0]
    lp = np.full((len(path), 3), np.log(0.05))
    lp[np.arange(len(path)), path] = np.log(0.9)
    spans = ctc_viterbi(lp, [1, 2, 1])
    assert spans.tolist() == [[1, 2], [4, 6], [8, 8]]
    assert ctc_viterbi(lp[:2], [1, 2, 1]) is None
    # Repeated labels need a blank between them.
    assert ctc_viterbi(np.log(np.full((3, 3), 1 / 3)), [1, 1]).tolist() == [[0, 0], [2, 2]]


def test_wav2vec2_aligner_retimes_words_from_emissions():
    a = Wav2Vec2Aligner(device="cpu")
    a.model = object()
    a.vocab = {"-": 0, "|": 1, "H": 2, "I": 3, "Y": 4, "O": 5}
    frames_per_s = 50
    # Audio 2 s; "hi" spoken at 0.40-0.60 s, "yo" at 1.20-1.40 s (relative to the padded window start).
    def emissions(chunk):
        n = int(len(chunk) / 16000 * frames_per_s)
        lp = np.full((n, 6), np.log(0.01))
        lp[:, 0] = np.log(0.9)
        def put(t0, t1, ids):
            f = np.linspace(t0 * frames_per_s, t1 * frames_per_s, len(ids) + 1).astype(int)
            for k, i in enumerate(ids):
                lp[f[k]:max(f[k] + 1, f[k + 1]), :] = np.log(0.01)
                lp[f[k]:max(f[k] + 1, f[k + 1]), i] = np.log(0.95)
        put(0.40, 0.60, [2, 3]); put(0.8, 0.84, [1]); put(1.20, 1.40, [4, 5])
        return lp

    a.emissions = emissions
    tr = Transcript([{"word": "Hi,", "start": 0.25, "end": 0.5, "confidence": 0.9, "segment": 0}, {"word": "yo!", "start": 1.0, "end": 1.3, "confidence": 0.9, "segment": 0},
                     {"word": "42", "start": 1.3, "end": 1.6, "confidence": 0.9, "segment": 0}],
                    [{"id": 0, "start": 0.25, "end": 1.6, "text": "Hi, yo! 42", "word_start": 0, "word_end": 3}], "en", 1.0)
    assert a.align(np.zeros(32000, np.float32), tr) == 2
    hi, yo, num = tr.words
    # Starts are shifted 50 ms earlier (CTC spikes lag the onset); ends are not.
    assert hi["start"] == pytest.approx(0.35, abs=0.03) and hi["end"] == pytest.approx(0.60, abs=0.03)
    assert yo["start"] == pytest.approx(1.15, abs=0.03) and yo["end"] == pytest.approx(1.40, abs=0.03)
    assert num["start"] >= yo["end"] and num["end"] == 1.6  # unalignable digits keep Whisper's time, kept monotonic
    tr.language = "de"
    assert a.align(np.zeros(32000, np.float32), tr) == 0


# ----------------------------------------------------------------------------------- libdf shim
def test_libdf_shim_stft_round_trip_and_bands():
    from sam_audio_service.speech import libdf_shim as libdf

    df = libdf.DF(sr=48000, fft_size=960, hop_size=480, nb_bands=32, min_nb_erb_freqs=2)
    w = df.erb_widths()
    assert int(w.sum()) == 481 and len(w) == 32 and int(w.min()) >= 2
    x = np.random.default_rng(0).standard_normal((2, 48000)).astype(np.float32)
    spec = df.analysis(x)
    assert spec.shape == (2, 100, 481) and spec.dtype == np.complex64
    y = df.synthesis(spec)
    d = 960 - 480  # libDF's delay
    np.testing.assert_allclose(y[:, d:], x[:, : x.shape[1] - d], atol=1e-5)
    e = libdf.erb(spec, w)
    assert e.shape == (2, 100, 32)
    assert libdf.erb_norm(e, 0.99).shape == e.shape and libdf.unit_norm(spec[..., :96], 0.99).shape == (2, 100, 96)
    assert libdf.unit_norm_init(96).shape == (1, 96)


def test_libdf_shim_install_registers_modules_only_when_missing():
    code = ("import sys; from sam_audio_service.speech import libdf_shim as s; print(s.install()); import libdf; "
            "print(libdf.DF.__module__); print(hasattr(sys.modules['torchaudio.backend.common'], 'AudioMetaData'))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(__import__("pathlib").Path(__file__).resolve().parents[1]))
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["shim", "sam_audio_service.speech.libdf_shim", "True"]
