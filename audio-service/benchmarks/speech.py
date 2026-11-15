"""Phase B speech/rhythm benchmark (A40): /v1/words, /v1/enhance, /v1/beats.

Runs on the GPU pod, never in tests (heavy imports live inside functions). Every subcommand
writes JSON under ``--work/results`` (copied into benchmarks/results/2026-11-14-speech/raw/).

Data (``--work``, default /workspace/speech):
  data/libri_align/{id}.wav,.json  LibriSpeech test-clean utterances (CC BY 4.0) with Montreal
                                   Forced Aligner word times (gilkeyio/librispeech-alignments, CC BY 4.0),
                                   fetched through the HF datasets-server rows API (no parquet download)
  data/tts/{id}.wav,.json          scripted creator speech with profanity and fillers, synthesised with
                                   Piper (MIT, rhasspy/piper 1.2.0) + the en_US-libritts_r-medium voice
                                   (CC BY 4.0, 904 speakers). Each target word (swear word / filler)
                                   is synthesised as its own unit and spliced between its context
                                   phrases, so its reference start/end are known to the sample
                                   (energy-trimmed). Includes clean "trap" words (assume, class,
                                   cocktail, Scunthorpe, hello, like-as-verb).
  noise: ESC-50 clips from /workspace/music-bench/data/sfx_esc50 (CC BY-NC 3.0; benchmark only)
         and synthetic pink/babble-free noise.

Subcommands:
  prepare-libri [--n 100]                     (venv-speech)
  prepare-tts                                 (needs piper: /root/dfref/bin/python, Python 3.11)
  check-libdf --dump ref|shim / --compare     DeepFilterLib (py3.11 wheel) vs speech/libdf_shim.py
  words --model large-v3-turbo [--align on|off|both] [--batch 8]
  enhance                                     SI-SDR on LibriSpeech + noise, 4-min latency/VRAM
  beats                                       synthetic click tracks (F-measure) + 4-min latency/VRAM
  e2e [--url http://127.0.0.1:8061]           HTTP end to end against a running service

Re-run (pod):
  export HF_HOME=/workspace/hf TORCH_HOME=/workspace/torch-cache; PY=/workspace/venv-speech/bin/python
  $PY speech.py prepare-libri && /root/dfref/bin/python speech.py prepare-tts
  $PY speech.py words --model large-v3-turbo --align both && $PY speech.py enhance && $PY speech.py beats
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))


# ----------------------------------------------------------------------------------- utils
def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2**31
    else:
        raise ValueError(f"{path}: sample width {width}")
    return x.reshape(-1, ch).T.copy(), rate


def write_wav(path: Path, x: np.ndarray, rate: int) -> None:
    x = np.atleast_2d(x)
    pcm = (np.clip(x, -1, 1).T * 32767).round().astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(x.shape[0]); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())


def resample(x: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target:
        return x
    import soxr

    return np.stack([soxr.resample(c, sr, target) for c in np.atleast_2d(x)]).astype(np.float32)


def norm_word(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'"))


class GpuSampler:
    """Peak GPU memory of this process (nvidia-smi, includes CTranslate2 and the CUDA context)
    sampled every 100 ms, plus torch's own peak allocation."""

    def __init__(self):
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _poll(self) -> int:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            pid, mem = [p.strip() for p in line.split(",")]
            if int(pid) == os.getpid():
                return int(mem)
        return 0

    def _run(self):
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, self._poll())
            time.sleep(0.1)

    def __enter__(self):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set(); self._thread.join()
        self.peak_mib = max(self.peak_mib, self._poll())
        try:
            import torch

            self.torch_peak_mib = round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0
        except ImportError:
            self.torch_peak_mib = 0


def save(work: Path, name: str, payload: dict) -> None:
    out = work / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(payload, indent=1))
    print(f"wrote {out / (name + '.json')}")


def settings_for(**env):
    for k, v in env.items():
        os.environ[k] = str(v)
    os.environ.setdefault("AUDIO_BACKEND", "mock")
    from sam_audio_service.config import load_settings

    return load_settings()


# ----------------------------------------------------------------------------------- data
def prepare_libri(work: Path, n: int) -> None:
    out = work / "data" / "libri_align"
    out.mkdir(parents=True, exist_ok=True)
    total, per_page = 2620, 25
    offsets = np.linspace(0, total - per_page, max(1, n // per_page)).astype(int)
    count = 0
    for off in offsets:
        url = f"https://datasets-server.huggingface.co/rows?dataset=gilkeyio/librispeech-alignments&config=default&split=test_clean&offset={off}&length={per_page}"
        rows = json.load(urllib.request.urlopen(url, timeout=60))["rows"]
        for r in rows:
            row = r["row"]
            wav = out / f"{row['id']}.wav"
            if not wav.exists():
                data = urllib.request.urlopen(row["audio"][0]["src"], timeout=60).read()  # FLAC bytes
                subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)], input=data, check=True)
            (out / f"{row['id']}.json").write_text(json.dumps({"id": row["id"], "transcript": row["transcript"], "words": [w for w in row["words"] if w["word"]]}))
            count += 1
    print(f"{count} utterances in {out}")


TTS_SCRIPT = [
    # (units, kinds): a unit in [brackets] is a target synthesised alone; kind p = profanity, f = filler, n = clean trap
    "So [um] I think we should go | p",
    "What the [fuck] is going on here",
    "This is [bullshit] and you know it",
    "Oh [shit] I forgot my keys",
    "Are you [fucking] kidding me right now",
    "He is such an [asshole] sometimes",
    "[Damn] that was close",
    "Shut up you little [bitch]",
    "Well [uh] I am not really sure",
    "It was [um] honestly kind of amazing",
    "I [uh] lost the remote again",
    "Get the [hell] out of my kitchen",
    "That guy is a total [dick]",
    "We were [um] going to call you",
    "What a [bastard] move that was",
    "[Fuck] this traffic",
    "Holy [shit] look at that",
    "I [uh] might be late tomorrow",
    "Do not be a [jackass] about it",
    "[Um] can you hear me now",
    # Clean traps: nothing here may be flagged.
    "I assume the class starts at noon",
    "She ordered a cocktail in Scunthorpe",
    "Hello there I like this song",
    "The passage about the shell company was assessed",
    "I would like to thank the committee",
]


def prepare_tts(work: Path, voice: Path, speakers: list[int]) -> None:
    from piper import PiperVoice  # Python 3.11 env with piper-tts 1.2.0

    v = PiperVoice.load(str(voice), config_path=str(voice) + ".json")
    rate = v.config.sample_rate
    out = work / "data" / "tts"
    out.mkdir(parents=True, exist_ok=True)

    def synth(text: str, spk: int) -> np.ndarray:
        raw = b"".join(v.synthesize_stream_raw(text, speaker_id=spk, sentence_silence=0.0))
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
        # Trim Piper's leading/trailing silence at -50 dB re. peak (5 ms envelope). A higher
        # threshold (first try: 2 %, -34 dB) cut the onset of fricatives ("sh" in "shit").
        env = np.convolve(np.abs(x), np.ones(110) / 110, mode="same")
        on = np.flatnonzero(env > 0.003 * env.max())
        return x[max(0, on[0] - 22): on[-1] + 22] if on.size else x

    gap = np.zeros(int(0.08 * rate), np.float32)
    lead = np.zeros(int(0.5 * rate), np.float32)
    manifest = []
    for si, spk in enumerate(speakers):
        for li, line in enumerate(TTS_SCRIPT):
            text = line.split("|")[0].strip()
            parts = re.split(r"(\[[^\]]+\])", text)
            pieces, targets, t = [lead], [], len(lead)
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                target = p.startswith("[")
                audio = synth(p.strip("[]"), spk)
                if target:
                    word = p.strip("[]")
                    kind = "f" if norm_word(word) in {"um", "uh", "er"} else "p"
                    targets.append({"word": word, "kind": kind, "start": t / rate, "end": (t + len(audio)) / rate})
                pieces += [audio, gap]
                t += len(audio) + len(gap)
            pieces.append(lead)
            clip = np.concatenate(pieces)
            cid = f"spk{spk}_{li:02d}"
            write_wav(out / f"{cid}.wav", clip, rate)
            entry = {"id": cid, "speaker": spk, "text": text.replace("[", "").replace("]", ""), "targets": targets, "trap": li >= 20}
            (out / f"{cid}.json").write_text(json.dumps(entry))
            manifest.append(entry)
    # One 4-minute creator monologue for latency: all clips of all speakers back to back.
    print(f"{len(manifest)} TTS clips in {out}")


# ----------------------------------------------------------------------------------- libdf check
def check_libdf(work: Path, dump: str | None, compare: bool) -> None:
    path = work / "results" / "libdf"
    path.mkdir(parents=True, exist_ok=True)
    if dump:
        rng = np.random.default_rng(0)
        x = (0.3 * rng.standard_normal((2, 48000 * 3))).astype(np.float32)
        if dump == "ref":
            import libdf
        else:
            from sam_audio_service.speech import libdf_shim as libdf
        df = libdf.DF(sr=48000, fft_size=960, hop_size=480, nb_bands=32, min_nb_erb_freqs=2)
        spec = df.analysis(x)
        w = df.erb_widths()
        e = libdf.erb(spec, w).copy()
        en = libdf.erb_norm(libdf.erb(spec, w), 0.99)
        un = libdf.unit_norm(spec[..., :96].copy(), 0.99)
        y = df.synthesis(spec.copy())
        np.savez(path / f"{dump}.npz", spec=spec, widths=np.asarray(w), erb=e, erb_norm=en, unit_norm=un, synthesis=y, unit_norm_init=libdf.unit_norm_init(96))
    if compare:
        a, b = np.load(path / "ref.npz"), np.load(path / "shim.npz")
        result = {k: {"max_abs_diff": float(np.max(np.abs(a[k].astype(complex) - b[k].astype(complex)))), "max_abs": float(np.max(np.abs(a[k])))} for k in a.files}
        save(work, "libdf_check", result)
        print(json.dumps(result, indent=1))


# ----------------------------------------------------------------------------------- words
def align_words(ref: list[str], hyp: list[str]) -> list[tuple[int, int]]:
    """Index pairs of equal words on a minimum-edit alignment (also yields WER)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    pairs, i, j = [], n, m
    while i > 0 and j > 0:
        if ref[i - 1] == hyp[j - 1] and d[i, j] == d[i - 1, j - 1]:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif d[i, j] == d[i - 1, j - 1] + 1:
            i -= 1; j -= 1
        elif d[i, j] == d[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1], int(d[n, m])


def timing_stats(errors: list[float]) -> dict:
    e = np.abs(np.asarray(errors)) * 1000
    if not e.size:
        return {}
    return {"n": int(e.size), "mean_ms": round(float(e.mean()), 1), "median_ms": round(float(np.median(e)), 1), "p90_ms": round(float(np.percentile(e, 90)), 1),
            "within_50ms": round(float((e <= 50).mean()), 4), "within_80ms": round(float((e <= 80).mean()), 4), "within_100ms": round(float((e <= 100).mean()), 4)}


def bench_words(work: Path, model: str, align: str, batch: int, compute_type: str, beam: int) -> None:
    from sam_audio_service.audio_io import decode
    from sam_audio_service.speech import SpeechTools
    from sam_audio_service.speech.contracts import WordsMetadata

    variants = ["off", "on"] if align == "both" else [align]
    s = settings_for(AUDIO_WORDS_BACKEND="faster_whisper", AUDIO_WORDS_MODEL=model, AUDIO_WORDS_BATCH_SIZE=batch, AUDIO_WORDS_ALIGN="wav2vec2",
                     AUDIO_WORDS_COMPUTE_TYPE=compute_type, AUDIO_WORDS_BEAM_SIZE=beam)
    tools = SpeechTools(s)
    t0 = time.perf_counter()
    with tools.words.lock:
        tools.words.ensure_loaded()
    with tools.aligner.lock:
        tools.aligner.ensure_loaded()
    load_s = time.perf_counter() - t0
    results: dict = {"model": model, "batch_size": batch, "compute_type": compute_type, "beam_size": beam, "load_seconds": round(load_s, 2), "variants": {}}
    libri = sorted((work / "data" / "libri_align").glob("*.json"))
    tts = sorted((work / "data" / "tts").glob("*.json"))
    cache: dict[str, object] = {}

    def run(path: Path, align_on: bool, fillers: bool = True):
        audio = decode(path=path, target_rate=16000, max_channels=1)
        return tools.transcribe(audio, WordsMetadata(align=align_on, transcribe_fillers=fillers), "bench")

    for variant in variants:
        on = variant == "on"
        starts, ends, errs, nref, times, dur = [], [], 0, 0, 0.0, 0.0
        for j in libri:
            ref = json.loads(j.read_text())
            r = run(j.with_suffix(".wav"), on)
            times += r["timing"]["transcribe_seconds"] + r["timing"]["align_seconds"]
            dur += r["source"]["duration"]
            rw = [norm_word(w["word"]) for w in ref["words"]]
            hw = [norm_word(w["word"]) for w in r["words"]]
            pairs, dist = align_words(rw, hw)
            errs += dist; nref += len(rw)
            for i, k in pairs:
                starts.append(r["words"][k]["start"] - ref["words"][i]["start"])
                ends.append(r["words"][k]["end"] - ref["words"][i]["end"])
        boundary = timing_stats(starts + ends)
        libri_res = {"utterances": len(libri), "ref_words": nref, "wer": round(errs / max(1, nref), 4), "start": timing_stats(starts), "end": timing_stats(ends), "boundary": boundary,
                     "rtf": round(times / max(dur, 1e-9), 4), "signed_mean_start_ms": round(float(np.mean(starts)) * 1000, 1) if starts else None}
        # Scripted profanity / filler clips.
        prof_hit = prof_n = fill_hit = fill_n = fp = trap_words = 0
        target_err, target_start, target_end = [], [], []
        misses = []
        for j in tts:
            meta = json.loads(j.read_text())
            r = run(j.with_suffix(".wav"), on)
            flagged_p, flagged_f = set(r["profanity"]), set(r["fillers"])
            if meta["trap"]:
                fp += len(flagged_p) + len(flagged_f)
                trap_words += len(r["words"])
                if flagged_p or flagged_f:
                    misses.append({"id": meta["id"], "false_positive": [r["words"][i]["word"] for i in sorted(flagged_p | flagged_f)]})
                continue
            for tgt in meta["targets"]:
                flagged = flagged_p if tgt["kind"] == "p" else flagged_f
                mid = 0.5 * (tgt["start"] + tgt["end"])
                cand = [i for i in flagged if r["words"][i]["start"] - 0.15 <= mid <= r["words"][i]["end"] + 0.15]
                if tgt["kind"] == "p":
                    prof_n += 1
                else:
                    fill_n += 1
                if cand:
                    w = r["words"][cand[0]]
                    target_err += [w["start"] - tgt["start"], w["end"] - tgt["end"]]
                    target_start.append(w["start"] - tgt["start"]); target_end.append(w["end"] - tgt["end"])
                    if tgt["kind"] == "p":
                        prof_hit += 1
                    else:
                        fill_hit += 1
                else:
                    misses.append({"id": meta["id"], "missed": tgt["word"], "heard": " ".join(w["word"] for w in r["words"])})
        tts_res = {"clips": len(tts), "profanity_recall": round(prof_hit / max(1, prof_n), 4), "profanity_targets": prof_n,
                   "filler_recall": round(fill_hit / max(1, fill_n), 4), "filler_targets": fill_n, "trap_false_positives": fp, "trap_words": trap_words,
                   "target_boundary": timing_stats(target_err), "target_start": timing_stats(target_start), "target_end": timing_stats(target_end),
                   "target_signed_median_ms": {"start": round(float(np.median(target_start)) * 1000, 1) if target_start else None,
                                               "end": round(float(np.median(target_end)) * 1000, 1) if target_end else None},
                   "misses": misses[:40]}
        results["variants"][variant] = {"librispeech": libri_res, "tts": tts_res}
        print(variant, json.dumps({"libri": {k: libri_res[k] for k in ("wer", "boundary", "rtf")}, "tts": {k: tts_res[k] for k in ("profanity_recall", "filler_recall", "trap_false_positives", "target_boundary")}}, indent=1))
    # Latency + VRAM on a 4-minute file (LibriSpeech utterances concatenated with 0.4 s gaps).
    long = work / "data" / "speech_4min_16k.wav"
    if not long.exists():
        chunks, total = [], 0
        for j in libri * 3:
            x, sr = read_wav(j.with_suffix(".wav"))
            chunks += [x[0], np.zeros(int(0.4 * sr), np.float32)]
            total += x.shape[1] + int(0.4 * sr)
            if total >= 240 * sr:
                break
        write_wav(long, np.concatenate(chunks)[: 240 * 16000], 16000)
    audio = decode(path=long, target_rate=16000, max_channels=1)
    for variant in variants:
        on = variant == "on"
        runs = []
        for _ in range(3):
            with GpuSampler() as g:
                t0 = time.perf_counter()
                r = tools.transcribe(audio, WordsMetadata(align=on), "bench")
                wall = time.perf_counter() - t0
            runs.append({"wall_seconds": round(wall, 2), **r["timing"], "words": len(r["words"]), "process_vram_peak_mib": g.peak_mib, "torch_peak_mib": g.torch_peak_mib})
        results["variants"][variant]["latency_4min"] = {"duration": round(audio.duration, 1), "runs": runs}
        print(variant, "4-min", runs)
    tag = f"words_{model}_b{batch}_{compute_type}_beam{beam}".replace("/", "_")
    save(work, tag, results)


# ----------------------------------------------------------------------------------- enhance
def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref, est = ref - ref.mean(), est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = a * ref
    return float(10 * np.log10(np.sum(target**2) / (np.sum((est - target) ** 2) + 1e-12) + 1e-12))


def pink(n: int, rng) -> np.ndarray:
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(len(spec)); f[0] = 1
    return np.fft.irfft(spec / np.sqrt(f), n=n).astype(np.float32)


def bench_enhance(work: Path, n_utts: int, noise_dir: Path) -> None:
    from sam_audio_service.audio_io import decode
    from sam_audio_service.speech import SpeechTools
    from sam_audio_service.speech.contracts import EnhanceMetadata

    s = settings_for(AUDIO_ENHANCE_BACKEND="deepfilternet")
    tools = SpeechTools(s)
    t0 = time.perf_counter()
    with tools.enhancer.lock:
        tools.enhancer.ensure_loaded()
    load_s = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    libri = sorted((work / "data" / "libri_align").glob("*.wav"))[:n_utts]
    noises = sorted(noise_dir.glob("*.wav"))
    keep = [p for p in noises if re.match(r"(rain|engine|vacuum|crowd|keyboard|washing|wind|crackling|airplane|clock|chirping|footsteps|door|car_horn|dog|siren|train|helicopter)", p.name)][:30] or noises[:30]
    from sam_audio_service.audio_io import DecodedAudio

    rows = []
    for idx, path in enumerate(libri):
        clean = resample(read_wav(path)[0], 16000, 48000)[0]
        for kind in ("esc50", "pink"):
            if kind == "esc50":
                nz = decode(path=keep[idx % len(keep)], target_rate=48000, max_channels=1).samples[0]
                nz = np.tile(nz, int(np.ceil(len(clean) / max(1, len(nz)))))[: len(clean)]
            else:
                nz = pink(len(clean), rng)
            for snr in (0, 5, 10):
                g = np.sqrt(np.mean(clean**2) / (np.mean(nz**2) + 1e-12) / 10 ** (snr / 10))
                noisy = (clean + g * nz).astype(np.float32)
                peak = max(1.0, float(np.abs(noisy).max()) / 0.99)
                noisy, ref = noisy / peak, clean / peak
                row = {"utt": path.stem, "noise": kind, "snr_db": snr, "noisy_si_sdr": round(si_sdr(ref, noisy), 2)}
                for lim in (None, 12.0):
                    out, _ = tools.enhance(DecodedAudio(noisy[None], 48000, 48000, 1), EnhanceMetadata(atten_lim_db=lim))
                    row[f"enh_si_sdr_{'full' if lim is None else int(lim)}"] = round(si_sdr(ref, out[0]), 2)
                rows.append(row)
        # Clean speech must survive (no damage when there is nothing to remove).
        out, _ = tools.enhance(DecodedAudio(clean[None].astype(np.float32), 48000, 48000, 1), EnhanceMetadata())
        rows.append({"utt": path.stem, "noise": "none", "snr_db": None, "noisy_si_sdr": None, "enh_si_sdr_full": round(si_sdr(clean, out[0]), 2)})
    summary = {}
    for kind in ("esc50", "pink"):
        for snr in (0, 5, 10):
            sel = [r for r in rows if r["noise"] == kind and r["snr_db"] == snr]
            summary[f"{kind}_{snr}dB"] = {"n": len(sel), **{k: round(float(np.mean([r[k] for r in sel])), 2) for k in ("noisy_si_sdr", "enh_si_sdr_full", "enh_si_sdr_12")},
                                          "si_sdr_improvement_full": round(float(np.mean([r["enh_si_sdr_full"] - r["noisy_si_sdr"] for r in sel])), 2)}
    clean_rows = [r["enh_si_sdr_full"] for r in rows if r["noise"] == "none"]
    summary["clean_passthrough_si_sdr"] = round(float(np.mean(clean_rows)), 2)
    # 4-min stereo latency at 48 kHz (dialogue layer sized), whole-file vs default chunking.
    x = decode(path=work / "data" / "speech_4min_16k.wav", target_rate=48000, max_channels=1).samples
    stereo = np.concatenate([x, 0.9 * x]) + 0.01 * rng.standard_normal((2, x.shape[1])).astype(np.float32)
    lat = []
    for _ in range(3):
        with GpuSampler() as g:
            t0 = time.perf_counter()
            out, rep = tools.enhance(DecodedAudio(stereo.astype(np.float32), 48000, 48000, 2), EnhanceMetadata())
            wall = time.perf_counter() - t0
        lat.append({"wall_seconds": round(wall, 2), "process_vram_peak_mib": g.peak_mib, "torch_peak_mib": g.torch_peak_mib})
    save(work, "enhance_deepfilternet3", {"load_seconds": round(load_s, 2), "summary": summary, "latency_4min_stereo": lat, "chunk_seconds": s.enhance_chunk_seconds, "rows": rows,
                                         "noise_files": [p.name for p in keep], "libdf": tools.enhancer.model.libdf})
    print(json.dumps(summary, indent=1), lat)


# ----------------------------------------------------------------------------------- beats
def click_track(bpm: float, seconds: float, sr: int, meter: int, rng, swing: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kick on downbeats, snare on 2/4, hats on eighths, plus a bass note per beat."""
    n = int(seconds * sr)
    out = np.zeros(n, np.float32)
    ibi = 60.0 / bpm
    beats = np.arange(0.5, seconds - 0.5, ibi)
    t = np.arange(int(0.25 * sr)) / sr
    kick = np.sin(2 * np.pi * (50 + 80 * np.exp(-t * 30)) * t) * np.exp(-t * 12)
    snare = rng.standard_normal(len(t)) * np.exp(-t * 25) * 0.5
    hat = rng.standard_normal(int(0.05 * sr)) * np.exp(-np.arange(int(0.05 * sr)) / sr * 80) * 0.15
    for k, b in enumerate(beats):
        i = int(b * sr)
        pos = k % meter
        src = kick if pos == 0 else (snare if pos % 2 == 1 else kick * 0.6)
        out[i: i + len(src)] += src[: n - i]
        for h in (0.0, 0.5 + swing):
            j = int((b + h * ibi) * sr)
            out[j: j + len(hat)] += hat[: max(0, n - j)]
        bass = 0.2 * np.sin(2 * np.pi * (55 * 2 ** ((k // meter) % 4 / 12)) * t[: int(0.2 * sr)]) * np.exp(-t[: int(0.2 * sr)] * 5)
        out[i: i + len(bass)] += bass[: n - i]
    return out / (np.abs(out).max() + 1e-9) * 0.8, beats, beats[::meter]


def f_measure(ref: np.ndarray, est: np.ndarray, tol: float = 0.07) -> float:
    if not len(ref) or not len(est):
        return 0.0
    used, hits = set(), 0
    for r in ref:
        d = np.abs(np.asarray(est) - r)
        j = int(np.argmin(d))
        if d[j] <= tol and j not in used:
            used.add(j); hits += 1
    p, rc = hits / len(est), hits / len(ref)
    return round(2 * p * rc / (p + rc), 4) if p + rc else 0.0


def bench_beats(work: Path, song: Path) -> None:
    from sam_audio_service.audio_io import DecodedAudio, decode
    from sam_audio_service.speech import SpeechTools
    from sam_audio_service.speech.contracts import BeatsMetadata

    rng = np.random.default_rng(0)
    out: dict = {}
    for backend in ("beat_this", "librosa"):
        s = settings_for(AUDIO_BEATS_BACKEND=backend, AUDIO_BEATS_FALLBACK="off")
        tools = SpeechTools(s)
        t0 = time.perf_counter()
        with tools.beats.lock:
            tools.beats.ensure_loaded()
        load_s = time.perf_counter() - t0
        rows = []
        for bpm, meter in ((90, 4), (100, 4), (120, 4), (128, 4), (140, 4), (174, 4), (96, 3), (150, 3)):
            x, beats, downbeats = click_track(bpm, 30, 22050, meter, rng)
            r = tools.track_beats(DecodedAudio(x[None], 22050, 22050, 1), BeatsMetadata(), "b")
            rows.append({"bpm": bpm, "meter": meter, "beat_f": f_measure(beats, np.asarray(r["beats"])), "downbeat_f": f_measure(downbeats, np.asarray(r["downbeats"])) if r["downbeats_available"] else None,
                         "tempo_est": r["tempo_bpm"], "beats_per_bar_est": r["beats_per_bar"]})
        song_audio = decode(path=song, target_rate=22050, max_channels=1)
        lat = []
        for _ in range(3):
            with GpuSampler() as g:
                t0 = time.perf_counter()
                r = tools.track_beats(song_audio, BeatsMetadata(), "b")
                wall = time.perf_counter() - t0
            lat.append({"wall_seconds": round(wall, 2), "process_vram_peak_mib": g.peak_mib, "torch_peak_mib": g.torch_peak_mib})
        out[backend] = {"load_seconds": round(load_s, 2), "synthetic": rows, "mean_beat_f": round(float(np.mean([r["beat_f"] for r in rows])), 4),
                        "mean_downbeat_f": round(float(np.mean([r["downbeat_f"] for r in rows if r["downbeat_f"] is not None])), 4) if backend == "beat_this" else None,
                        "song": {"file": song.name, "duration": round(song_audio.duration, 1), "tempo_bpm": r["tempo_bpm"], "beats_per_bar": r["beats_per_bar"], "beats": len(r["beats"]), "downbeats": len(r["downbeats"])},
                        "latency_song": lat, "model": r["model"]}
        print(backend, json.dumps({k: v for k, v in out[backend].items() if k != "synthetic"}, indent=1))
        tools.unload()
    # Agreement between the two trackers on the real song (beat F at 70 ms, beat_this as reference).
    save(work, "beats", out)


def e2e(work: Path, url: str, song: Path) -> None:
    """End to end over HTTP against a running service (the pod's test instance on :8061)."""
    import base64

    import httpx

    out: dict = {}
    with httpx.Client(base_url=url, timeout=600) as c:
        caps = c.get("/capabilities").json()
        out["capabilities"] = {k: caps[k] for k in ("words", "enhance", "beats")}
        out["health_speech"] = c.get("/health").json()["speech"]
        clip = work / "data" / "tts" / "spk0_01.wav"
        for _ in range(2):  # second call is warm
            t0 = time.perf_counter()
            r = c.post("/v1/words", files={"audio": (clip.name, clip.read_bytes(), "audio/wav")}, data={"metadata": json.dumps({"language": "en", "extra_profanity": ["going"]})})
            wall = time.perf_counter() - t0
        body = r.json()
        out["words"] = {"status": r.status_code, "wall_seconds": round(wall, 2), "text": " ".join(w["word"] for w in body["words"]),
                        "profanity": [body["words"][i]["word"] for i in body["profanity"]], "fillers": [body["words"][i]["word"] for i in body["fillers"]],
                        "aligned": body["aligned"], "model": body["model"], "warnings": body["warnings"]}
        noisy = work / "data" / "speech_4min_16k.wav"
        x, sr = read_wav(noisy)
        x = x[:, : 20 * sr] + 0.03 * np.random.default_rng(0).standard_normal((1, 20 * sr)).astype(np.float32)
        tmp = work / "e2e_noisy_16k.wav"
        write_wav(tmp, x, sr)
        t0 = time.perf_counter()
        r = c.post("/v1/enhance", files={"audio": (tmp.name, tmp.read_bytes(), "audio/wav")}, data={"metadata": json.dumps({"atten_lim_db": 20})})
        body = r.json()
        wav = base64.b64decode(body["audio_base64"]) if r.status_code == 200 else b""
        out["enhance"] = {"status": r.status_code, "wall_seconds": round(time.perf_counter() - t0, 2), **{k: body.get(k) for k in ("sample_rate", "channels", "sample_count", "atten_lim_db", "levels", "model", "warnings")},
                          "input_sample_count": int(x.shape[1]), "wav_bytes": len(wav)}
        t0 = time.perf_counter()
        r = c.post("/v1/beats", files={"audio": (song.name, song.read_bytes(), "audio/wav")})
        body = r.json()
        out["beats"] = {"status": r.status_code, "wall_seconds": round(time.perf_counter() - t0, 2), **{k: body.get(k) for k in ("tempo_bpm", "beats_per_bar", "downbeats_available", "model", "warnings")},
                        "beats": len(body.get("beats", [])), "downbeats": len(body.get("downbeats", [])), "first_downbeats": body.get("downbeats", [])[:4]}
        r = c.post("/v1/words", files={"audio": ("empty.wav", b"", "audio/wav")})
        out["error_example"] = {"status": r.status_code, "code": r.json()["error"]["code"]}
    save(work, "e2e_8061", out)
    print(json.dumps(out, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=Path("/workspace/speech"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare-libri"); p.add_argument("--n", type=int, default=100)
    p = sub.add_parser("prepare-tts"); p.add_argument("--voice", type=Path, default=Path("/root/piper/en_US-libritts_r-medium.onnx")); p.add_argument("--speakers", default="0,15,108,230,401,640")
    p = sub.add_parser("check-libdf"); p.add_argument("--dump", choices=["ref", "shim"]); p.add_argument("--compare", action="store_true")
    p = sub.add_parser("words"); p.add_argument("--model", default="large-v3-turbo"); p.add_argument("--align", default="both", choices=["on", "off", "both"]); p.add_argument("--batch", type=int, default=8)
    p.add_argument("--compute-type", default="float16"); p.add_argument("--beam", type=int, default=5)
    p = sub.add_parser("enhance"); p.add_argument("--n", type=int, default=20); p.add_argument("--noise-dir", type=Path, default=Path("/workspace/music-bench/data/sfx_esc50"))
    p = sub.add_parser("beats"); p.add_argument("--song", type=Path, default=Path("/workspace/music-bench/data/fullsong.wav"))
    p = sub.add_parser("e2e"); p.add_argument("--url", default="http://127.0.0.1:8061"); p.add_argument("--song", type=Path, default=Path("/workspace/music-bench/data/fullsong.wav"))
    a = ap.parse_args()
    if a.cmd == "prepare-libri":
        prepare_libri(a.work, a.n)
    elif a.cmd == "prepare-tts":
        prepare_tts(a.work, a.voice, [int(s) for s in a.speakers.split(",")])
    elif a.cmd == "check-libdf":
        check_libdf(a.work, a.dump, a.compare)
    elif a.cmd == "words":
        bench_words(a.work, a.model, a.align, a.batch, a.compute_type, a.beam)
    elif a.cmd == "enhance":
        bench_enhance(a.work, a.n, a.noise_dir)
    elif a.cmd == "beats":
        bench_beats(a.work, a.song)
    elif a.cmd == "e2e":
        e2e(a.work, a.url, a.song)


if __name__ == "__main__":
    main()
