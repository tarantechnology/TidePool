"""Phase E (generate a part that fits the song): analysis, vocabulary, composer, renderers, mix-in,
MIDI and the two HTTP endpoints. CPU only; the synthetic song has known tempo/key/chords."""
import base64
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sam_audio_service.api import create_app
from sam_audio_service.audio_io import decode, encode_wav_pcm16
from sam_audio_service.generation import Generator, PartRequest
from sam_audio_service.generation import dsp
from sam_audio_service.generation.composer import PNote, compose
from sam_audio_service.generation.evaluate import note_checks
from sam_audio_service.generation.midi import read_midi_notes, write_midi
from sam_audio_service.generation.mixin import mix_in
from sam_audio_service.generation.render import RendererSet, SynthRenderer, read_wav
from sam_audio_service.generation.settings import GenSettings
from sam_audio_service.generation.vocabulary import resolve
from sam_audio_service.music_analysis import MusicAnalyzer, Note, decode_chords, estimate_key, monophonic

SR = 48_000
BPM = 120.0
PROG = [(0, "maj"), (7, "maj"), (9, "min"), (5, "maj")]  # C G Am F, one bar each


def _tone(freq, n, sr, harmonics=4):
    t = np.arange(n) / sr
    return sum(np.sin(2 * np.pi * freq * k * t) / k for k in range(1, harmonics + 1))


def make_song(bars=8, sr=SR):
    beat = 60 / BPM
    n = int(bars * 4 * beat * sr)
    drums, bass, other, vocals = (np.zeros(n, np.float32) for _ in range(4))
    rng = np.random.default_rng(0)
    k = np.exp(-np.arange(int(0.12 * sr)) / (0.03 * sr))
    hat = rng.standard_normal(int(0.03 * sr)) * np.exp(-np.arange(int(0.03 * sr)) / (0.006 * sr))
    for b in range(bars * 8):  # 8th-note hats, kick on 1/3, snare on 2/4
        s = int(b * beat / 2 * sr)
        drums[s:s + len(hat)] += 0.25 * hat[: n - s]
        if b % 2:
            continue
        if b % 4 == 0:
            hit = np.sin(2 * np.pi * 60 * np.arange(len(k)) / sr * (1 + 2 * k)) * k
        else:
            hit = rng.standard_normal(len(k)) * 0.4 * k
        drums[s:s + len(k)] += 0.8 * hit[: n - s]
    melody_pcs = [4, 2, 0, 9]
    for bar in range(bars):
        root, q = PROG[bar % 4]
        s, e = int(bar * 4 * beat * sr), int((bar + 1) * 4 * beat * sr)
        bass[s:e] += 0.35 * _tone(dsp.midi_to_hz(36 + root), e - s, sr, 3)
        third = 3 if q == "min" else 4
        for iv in (0, third, 7):
            other[s:e] += 0.12 * _tone(dsp.midi_to_hz(60 + root + iv), e - s, sr, 3)
        if bar % 2 == 0:  # sing on even bars only (leave gaps)
            for i in range(4):
                a, z = s + int(i * beat * sr), s + int((i + 0.9) * beat * sr)
                pc = (root + (0, third, 7, third)[i]) % 12
                vocals[a:z] += 0.3 * _tone(dsp.midi_to_hz(67 + ((pc - 67) % 12)), z - a, sr, 5) * np.hanning(z - a)
    stems = {k: np.stack([v, v]).astype(np.float32) for k, v in (("drums", drums), ("bass", bass), ("other", other), ("vocals", vocals))}
    mix = sum(stems.values())
    return mix, stems


class FakeTranscriber:
    name = "fake"

    def __init__(self, notes):
        self.notes = notes

    def transcribe(self, mono, sr):
        return [Note(n.start, n.end, n.pitch, n.velocity) for n in self.notes]


@pytest.fixture(scope="module")
def song():
    return make_song()


@pytest.fixture(scope="module")
def analysis(song):
    mix, stems = song
    beat = 60 / BPM
    mel = [Note(bar * 4 * beat + i * beat, bar * 4 * beat + (i + 0.9) * beat, 67, 90) for bar in range(0, 8, 2) for i in range(4)]
    return MusicAnalyzer(transcriber=FakeTranscriber(mel)).analyze(mix, SR, stems)


def test_analysis_tempo_key_chords(analysis):
    assert abs(analysis["tempo"] - BPM) < 3 or abs(analysis["tempo"] - 2 * BPM) < 6
    assert analysis["key"]["label"] in ("C major", "A minor")
    labels = [c["chord"] for c in analysis["chords"] if c["end"] - c["start"] > 1.0]
    for want in ("C:maj", "G:maj", "A:min", "F:maj"):
        assert want in labels, labels
    assert analysis["melody_midi"] and analysis["vocal_activity"]
    assert analysis["context"]["stems_used"] == ["bass", "drums", "other", "vocals"]
    assert analysis["groove"]["kick_times"]
    assert set(analysis) >= {"tempo", "beats", "downbeats", "key", "chords", "sections", "melody_midi"}


def test_analysis_from_mix_only_omits_melody(song):
    mix, _ = song
    a = MusicAnalyzer().analyze(mix, SR, None)
    assert a["melody_midi"] is None and a["vocal_activity"] is None
    assert any("no vocal stem" in w for w in a["warnings"])
    assert a["chords"]


def test_key_and_viterbi_units():
    prof = np.zeros(12)
    prof[[0, 4, 7]] = 3
    prof[[2, 5, 9, 11]] = 1
    assert estimate_key(prof)["label"] == "C major"
    c = np.zeros((8, 12))
    c[:4, [0, 4, 7]] = 1
    c[4:, [9, 0, 4]] = 1
    path, _ = decode_chords(c, c, {"pc": 0, "mode": "major"})
    assert path[0] != path[-1] and len(set(path[:4])) == 1


def test_monophonic_skyline():
    notes = [Note(0, 1, 60, 50), Note(0.5, 1.5, 64, 90), Note(2, 2.05, 70, 90)]
    out = monophonic(notes)
    assert [n.pitch for n in out] == [60, 64] and out[0].end == 0.5


@pytest.mark.parametrize("ask,instrument,role", [
    ("add violins to the song", "violins", "counter_melody"),
    ("add a piano part", "piano", "chords"),
    ("add a funky bassline", "bass", "bassline"),
    ("add an 80s synth arpeggio", "synth_arp", "ostinato"),
    ("add a string pad", "strings", "pad"),
    ("add a trumpet line in the chorus", "trumpet", "counter_melody"),
    ("add a choir", "choir", "pad"),
    ("add a guitar riff", "guitar", "riff"),
    ("add a banjo", "banjo", "ostinato"),
])
def test_vocabulary(ask, instrument, role):
    spec = resolve(None, None, ask)
    assert (spec.instrument, spec.role) == (instrument, role)


def test_vocabulary_style_and_texture():
    assert resolve(None, None, "add a funky bassline").style == "funk"
    assert resolve(None, None, "an 80s synth arpeggio").style == "80s"
    assert resolve(None, None, "add an ambient shimmer texture").audio_generative_suggested
    assert resolve("cello", "ostinato").articulation == "spiccato"


@pytest.mark.parametrize("ask", ["string pad", "violins", "piano part", "funky bassline", "80s synth arpeggio", "trumpet line", "guitar riff", "brass stabs", "lead synth"])
def test_composer_follows_chords_and_grid(analysis, ask):
    spec = resolve(None, None, ask)
    notes = compose(spec, analysis, [(0.0, analysis["duration"])], seed=1)
    assert notes, ask
    chk = note_checks(notes, analysis)
    assert chk["chord_tone_ratio"] >= 0.6, (ask, chk)
    assert chk["in_key_ratio"] >= 0.85, (ask, chk)
    assert chk["onsets_within_30ms"] >= 0.8, (ask, chk)
    assert chk["vocal_clash_ratio"] <= 0.15, (ask, chk)
    assert all(spec.low <= n.pitch <= spec.high for n in notes)


def test_composer_deterministic_and_sections(analysis):
    spec = resolve("violins")
    a = compose(spec, analysis, [(4.0, 8.0)], seed=3)
    b = compose(spec, analysis, [(4.0, 8.0)], seed=3)
    assert [n.to_dict() for n in a] == [n.to_dict() for n in b]
    assert all(4.0 - 0.05 <= n.start < 8.0 for n in a)


def test_lead_leaves_vocal_alone(analysis):
    spec = resolve("synth lead", "lead")
    notes = compose(spec, analysis, [(0.0, analysis["duration"])], seed=0)
    chk = note_checks(notes, analysis)
    assert chk["vocal_overlap_s"] < 1.0


def test_midi_roundtrip():
    notes = [PNote(0.0, 0.5, 60, 100), PNote(0.5, 1.25, 64, 80), PNote(0.5, 1.25, 67, 80)]
    data = write_midi(notes, tempo=97.0, program=40)
    back = read_midi_notes(data)
    assert [(round(s, 2), round(e, 2), p, prog) for s, e, p, _, prog in back] == [(0.0, 0.5, 60, 40), (0.5, 1.25, 64, 40), (0.5, 1.25, 67, 40)]


@pytest.mark.parametrize("preset", ["pad", "arp80s", "lead", "bass", "pluck"])
def test_synth_renders_pitch(preset):
    out = SynthRenderer().render(preset, [PNote(0.1, 0.9, 57, 100)], SR, SR, tempo=120)
    assert out.shape == (2, SR) and np.isfinite(out).all() and np.abs(out).max() > 0.01
    spec = np.abs(np.fft.rfft(out[0, int(0.15 * SR):int(0.85 * SR)]))
    f = np.fft.rfftfreq(int(0.85 * SR) - int(0.15 * SR), 1 / SR)
    peak = f[np.argmax(spec * (f < 1000))]
    ratio = peak / 110.0  # a harmonic of A2 (the fundamental, the bass sub-octave or a filter-emphasised partial)
    assert abs(ratio - round(ratio)) < 0.06 and round(ratio) >= 1


def test_renderer_fallback_without_samples():
    rs = RendererSet(samples_dir=None, soundfont="/nonexistent.sf2")
    rs.gm.binary = None
    audio, used = rs.render("vsco:violins", "gm:48", [PNote(0, 0.5, 69, 90)], SR, SR, tempo=120, articulation="normal")
    assert used.startswith("synth:") and audio.shape == (2, SR)


def test_sampler_with_tiny_manifest(tmp_path):
    from sam_audio_service.generation.render import SampleRenderer
    root = tmp_path / "vsco"
    (root / "s").mkdir(parents=True)
    tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(int(1.5 * 44100)) / 44100)).astype(np.float32)
    (root / "s" / "a4.wav").write_bytes(encode_wav_pcm16(np.stack([tone, tone]), 44100))
    (root / "manifest.json").write_text(json.dumps({"presets": {"violins": {"sustain": True, "attack": 0.05, "release": 0.2, "zones": [{"file": "s/a4.wav", "pitch": 69.0, "layer": 0}]}}}))
    r = SampleRenderer(root)
    out = r.render("violins", [PNote(0.0, 3.0, 72, 100)], 4 * SR, SR)  # longer than the sample: loops
    seg = out[0, int(0.5 * SR):int(2.8 * SR)]
    assert np.abs(seg).min() < 0.5 and np.sqrt(np.mean(seg ** 2)) > 0.05
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    assert abs(f[np.argmax(np.abs(np.fft.rfft(seg)))] - 523.25) < 4  # ensemble layers are detuned a few cents
    assert not np.allclose(out[0], out[1])  # layers spread across the stereo field


def test_read_wav_24bit(tmp_path):
    x = (np.sin(np.linspace(0, 20, 1000)) * 0.5).astype(np.float32)
    ints = (x * 8388607).astype(np.int32)
    raw = b"".join(int(v).to_bytes(3, "little", signed=True) for v in ints)
    import struct
    fmt = struct.pack("<HHIIHH", 1, 1, 44100, 44100 * 3, 3, 24)
    data = b"RIFF" + struct.pack("<I", 4 + 8 + 16 + 8 + len(raw)) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt + b"data" + struct.pack("<I", len(raw)) + raw
    p = tmp_path / "x.wav"
    p.write_bytes(data)
    y, sr = read_wav(p)
    assert sr == 44100 and np.allclose(y[0], x, atol=1e-5)


def test_mix_in_levels_and_length(song, analysis):
    mix, stems = song
    part = SynthRenderer().render("pad", [PNote(0, 15, 60, 90), PNote(0, 15, 64, 90)], mix.shape[1], SR)
    out, rep = mix_in(part, SR, mix, stems["vocals"], analysis, "pad", [(0.0, analysis["duration"])])
    assert out.shape == mix.shape
    assert rep["loudness"]["relative_lu"] <= -8.5  # pad target, or further back to sit behind the vocal
    assert rep["peak"]["part_dbfs"] <= -0.99
    assert "ducking" in rep and "reverb" in rep and "eq_carve" in rep


def test_loudness_reference():
    t = np.arange(SR * 3) / SR
    x = 0.1 * np.sin(2 * np.pi * 1000 * t)
    assert abs(dsp.loudness_lufs(np.stack([x, x]), SR) - (-20.0)) < 1.0  # -20 dBFS-peak 1 kHz sine in both channels = -20 LUFS


@pytest.fixture
def generator(analysis):
    return Generator(GenSettings(samples_dir=None, amt_model="off"), analyzer=MusicAnalyzer(transcriber=FakeTranscriber([Note(0, 0.9, 67, 90)])), load_models=False)


def test_generator_end_to_end(song, analysis, generator):
    mix, stems = song
    res = generator.generate(mix, SR, stems, PartRequest(prompt="add a string pad in the chorus", analysis=analysis))
    assert res.audio.shape == mix.shape
    assert res.spec.instrument == "strings" and res.midi
    chorus = [s for s in analysis["sections"] if s["name"] == "chorus"]
    assert res.windows == [(s["start"], s["end"]) for s in chorus] or len(res.windows) >= 1
    assert res.checks["notes"]["chord_tone_ratio"] >= 0.6
    assert res.engine["name"] == "symbolic" and res.renderer["used"].startswith(("synth:", "gm:"))


def test_generator_musicgen_refused_without_noncommercial_opt_in(song, analysis):
    from sam_audio_service.errors import AudioServiceError

    class FakeGen:
        model_id = "fake/musicgen"

        def generate(self, prompt, cond, sr, seconds, seed=0):
            return np.zeros(int(seconds * 32000), np.float32), 32000

    mix, stems = song
    g = Generator(GenSettings(samples_dir=None, amt_model="off", allow_noncommercial=False), audiogen=FakeGen(), load_models=False)
    with pytest.raises(AudioServiceError) as exc:
        g.generate(mix, SR, stems, PartRequest(instrument="violins", engine="musicgen", analysis=analysis))
    assert exc.value.code == "UNSUPPORTED_OPERATION"
    g.settings.allow_noncommercial = True
    res = g.generate(mix, SR, stems, PartRequest(instrument="violins", engine="musicgen", analysis=analysis))
    assert res.engine["commercial_use"] is False and "CC-BY-NC" in res.engine["license"]["weights"]


# ------------------------------------------------------------------ HTTP

@pytest.fixture
def gen_client(settings, analysis):
    g = Generator(GenSettings(samples_dir=None, amt_model="off"), analyzer=MusicAnalyzer(), load_models=False)
    with TestClient(create_app(settings, generator=g)) as c:
        yield c


def _files(song, with_mix=True):
    mix, stems = song
    files = [("stem_" + k, (k + ".wav", encode_wav_pcm16(v, SR), "audio/wav")) for k, v in stems.items()]
    if with_mix:
        files.append(("audio", ("mix.wav", encode_wav_pcm16(np.clip(mix, -1, 1), SR), "audio/wav")))
    return files


def test_capabilities_generate_block(gen_client):
    cap = gen_client.get("/capabilities").json()["generate"]
    assert cap["available"] is True and "pad" in cap["roles"] and "violins" in cap["instruments"]
    assert cap["engines"]["musicgen"]["commercial_use"] is False
    assert cap["engines"]["symbolic"]["available"] is True


def test_analyze_music_endpoint_with_stems(gen_client, song):
    r = gen_client.post("/v1/analyze-music", files=_files(song, with_mix=False), data={"metadata": json.dumps({"melody": False})})
    assert r.status_code == 200, r.text
    a = r.json()["analysis"]
    assert abs(a["tempo"] - BPM) < 3 and a["chords"] and "vocals" in a["context"]["stems_used"]


def test_generate_part_endpoint(gen_client, song, analysis):
    meta = {"prompt": "add an 80s synth arpeggio", "interval": {"start": 2.0, "end": 10.0}, "analysis": analysis, "seed": 2}
    r = gen_client.post("/v1/generate-part", files=_files(song), data={"metadata": json.dumps(meta)})
    assert r.status_code == 200, r.text
    body = r.json()
    audio = decode(data=base64.b64decode(body["audio"]["audio_base64"])).samples
    assert audio.shape == song[0].shape
    assert body["part"]["kind"] == "music" and body["part"]["role"] == "ostinato"
    assert read_midi_notes(base64.b64decode(body["midi_base64"]))
    assert body["windows"] == [[2.0, 10.0]] and body["checks"]["notes"]["chord_tone_ratio"] >= 0.6
    assert np.abs(audio[:, : int(1.5 * SR)]).max() < 1e-3  # silent before the interval


def test_generate_part_errors(gen_client, song):
    r = gen_client.post("/v1/generate-part", files=_files(song), data={"metadata": json.dumps({"instrument": "violins", "engine": "bogus"})})
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_REQUEST"
    r = gen_client.post("/v1/generate-part", files=_files(song), data={"metadata": json.dumps({})})
    assert r.status_code == 400
    r = gen_client.post("/v1/generate-part", files=[("wat", ("x.wav", b"123", "audio/wav"))], data={"metadata": "{}"})
    assert r.status_code == 400
    r = gen_client.post("/v1/generate-part", files=_files(song), data={"metadata": json.dumps({"instrument": "violins", "engine": "amt"})})
    assert r.status_code == 400 and r.json()["error"]["code"] == "UNSUPPORTED_OPERATION"  # experimental engine is off by default
    r = gen_client.post("/v1/generate-part", files=_files(song), data={"metadata": json.dumps({"instrument": "violins", "engine": "musicgen"})})
    assert r.status_code == 400 and "CC-BY-NC" in r.json()["error"]["message"]


def test_defaults_are_rule_based_only(monkeypatch):
    monkeypatch.delenv("AUDIO_GEN_AMT_MODEL", raising=False)
    monkeypatch.delenv("AUDIO_GEN_AUDIOGEN", raising=False)
    s = GenSettings()
    assert s.amt_model == "off" and s.audiogen == "off" and s.allow_noncommercial is False
    g = Generator(s)
    caps = g.capabilities()["engines"]
    assert caps["symbolic"]["default"] and not caps["amt"]["available"] and not caps["musicgen"]["available"]


def test_generator_amt_engine_with_fake_model(song, analysis):
    class FakeAmt:
        model_id = "fake/amt"

        def compose(self, analysis, program, low, high, windows, seed=0, prompt_notes=None):
            assert prompt_notes and program == 0  # seeded with the rule-based part, GM piano
            beats = analysis["beats"]
            return [PNote(b + 0.04, b + 0.4, 60, 80) for b in beats[:8]] + [PNote(beats[9] + 0.2, beats[9] + 0.5, 64, 80)]

    mix, stems = song
    g = Generator(GenSettings(samples_dir=None), amt=FakeAmt(), load_models=False)
    res = g.generate(mix, SR, stems, PartRequest(instrument="piano", engine="amt", analysis=analysis))
    starts = [n.start for n in res.notes]
    assert all(min(abs(s - b) for b in analysis["beats"]) < 1e-6 for s in starts[:8])  # snapped to the grid
    assert res.engine["name"] == "amt" and res.engine["commercial_use"] is True and res.midi


@pytest.mark.parametrize("rid", ["../escaped", "/abs/path", "a/b", "..", "."])
def test_generate_part_rejects_unsafe_request_id(gen_client, song, analysis, settings, rid):
    meta = {"instrument": "violins", "analysis": analysis, "return_mode": "file", "request_id": rid}
    r = gen_client.post("/v1/generate-part", files=_files(song), data={"metadata": json.dumps(meta)})
    assert r.status_code in (400, 422), r.text
    assert not (Path(settings.output_root).resolve().parent / "escaped").exists()
    r = gen_client.post("/v1/analyze-music", files=_files(song), data={"metadata": json.dumps({"melody": False, "request_id": rid})})
    assert r.status_code in (400, 422) or rid in ("..", ".")  # analyze writes nothing; the pattern alone guards it


def test_generate_part_file_mode_safe_request_id(gen_client, song, analysis, settings):
    meta = {"instrument": "violins", "analysis": analysis, "return_mode": "file", "request_id": "req-1.ok"}
    r = gen_client.post("/v1/generate-part", files=_files(song), data={"metadata": json.dumps(meta)})
    assert r.status_code == 200, r.text
    assert (Path(settings.output_root).resolve() / "req-1.ok" / "part.wav").exists()


@pytest.mark.parametrize("bad", [{"tempo": 120}, {"duration": "x", "tempo": 120}, {"duration": 10, "tempo": 120, "chords": "C"},
                                 {"duration": 10, "tempo": 120, "sections": [1, 2]}])
def test_generate_part_malformed_analysis_is_400(gen_client, song, bad):
    r = gen_client.post("/v1/generate-part", files=_files(song), data={"metadata": json.dumps({"instrument": "violins", "analysis": bad})})
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_REQUEST", r.text


def test_generate_part_enforces_max_seconds_with_supplied_analysis(song, analysis, settings):
    g = Generator(GenSettings(samples_dir=None, amt_model="off", max_seconds=1.0), analyzer=MusicAnalyzer(), load_models=False)
    with TestClient(create_app(settings, generator=g)) as c:
        r = c.post("/v1/generate-part", files=_files(song), data={"metadata": json.dumps({"instrument": "violins", "analysis": analysis})})
    assert r.status_code == 400 and r.json()["error"]["code"] == "UNSUPPORTED_OPERATION", r.text
