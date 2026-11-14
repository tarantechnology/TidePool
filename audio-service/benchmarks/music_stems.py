"""Phase 0 music-stems benchmark: MSS separators, AudioSep text baseline, router tagger.

Runs on the GPU pod, never in tests (heavy imports live inside functions). Every subcommand
writes JSON next to its outputs so the report can be regenerated from raw results.

Data layout (``--work``, default /workspace/music-bench):
  musdb/dec/NN/{mix.wav, stems.npz, name.txt}   MUSDB18 7 s test previews at 48 kHz stereo
  data/{speech_say,speech_libri,sfx_esc50,speech_over_music}/*.wav, data/fullsong.wav
  results/*.json                                  raw results (small, copied into the repo)
Estimates go to ``--est`` (default /root/mb-est, the pod's local disk, not the network volume).

Pipeline (what the service will do):
  48 kHz float32 [2, n] -> julius.resample_frac to 44.1 kHz -> model -> back to 48 kHz -> crop/pad
  The mixture fed to every model is ``sum(stems)`` so references and mixture are exactly
  consistent (MUSDB's own mixture differs from the stem sum by a limiter on a few tracks).

Metrics (identical for every model, all at 48 kHz):
  usdr  global SDR per track and stem, 10*log10(sum ref^2 / sum (ref - est)^2) over both channels
        (the MDX'21 "SDR"; no distortion filter, so stricter than museval/BSSEval v4)
  csdr  same formula on 1 s non-overlapping chunks, median over chunks whose reference is not
        silent (mean square > 1e-8), i.e. museval's chunk/median aggregation without its 512-tap
        filter. Tracks whose stem is silent everywhere are excluded for that stem.
  *_mono  the same on the channel mean (est and ref), used for the AudioSep comparison.
  Variants: raw model output, and ``proj`` = mixture-consistency projection
  (est_k += (mix - sum est) * E_k / sum E, E_k = energy of est_k).

Subcommands (see --help): prepare-musdb, prepare-classifier-data, separate, audiosep, score,
classify, fullsong, coload.

Re-run (pod):  bash audio-service/scripts/setup_music_stems_venv.sh
  export TORCH_HOME=/workspace/torch-cache HF_HOME=/workspace/hf; PY=/workspace/venv-stems/bin/python
  $PY music_stems.py prepare-musdb && $PY music_stems.py prepare-classifier-data
  $PY music_stems.py separate --model demucs:htdemucs_ft           (one process per candidate)
  $PY music_stems.py separate --model chain:sep:model_bs_roformer_ep_317_sdr_12.9755.ckpt+demucs:htdemucs_ft
  $PY music_stems.py separate --config candidate.json              (declarative candidate)
  $PY music_stems.py audiosep && $PY music_stems.py score <tags under --est> --name all
  $PY music_stems.py classify --tagger panns ; $PY music_stems.py fullsong --project --model ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SR = 48_000
MODEL_SR = 44_100
STEMS = ("vocals", "drums", "bass", "other")


# ----------------------------------------------------------------------------- io helpers

def read_audio(path: str, sr: int = SR, mono: bool = False) -> np.ndarray:
    """Decode any file with ffmpeg to float32 [channels, n] at ``sr``."""
    ch = 1 if mono else 2
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, ch).T.copy()


def write_wav(path: str, x: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(path, np.asarray(x, dtype=np.float32).T, sr, subtype="FLOAT")


def clip_dirs(work: str, limit: int | None = None) -> list[str]:
    dirs = sorted(glob.glob(f"{work}/musdb/dec/[0-9][0-9]"))
    return dirs[:limit] if limit else dirs


def load_clip(d: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    z = np.load(f"{d}/stems.npz")
    ref = {k: z[k].astype(np.float32) for k in STEMS}
    return sum(ref.values()).astype(np.float32), ref


def fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[..., :n] if x.shape[-1] >= n else np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, n - x.shape[-1])])


def resample(x, sr_in: int, sr_out: int):
    """julius polyphase resampling on a torch tensor (what Demucs itself uses)."""
    import julius
    return x if sr_in == sr_out else julius.resample_frac(x, sr_in, sr_out)


def gpu_mem() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {}
    return {"peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20)}


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ----------------------------------------------------------------------------- data prep

def cmd_prepare_musdb(a):
    """Download the free MUSDB18 7 s previews (test subset) and decode to 48 kHz stereo."""
    import musdb
    import torch
    root = f"{a.work}/musdb/raw"
    db = musdb.DB(root=root, download=True, subsets="test")
    print(f"{len(db)} test tracks")
    for i, t in enumerate(db.tracks):
        d = f"{a.work}/musdb/dec/{i:02d}"
        os.makedirs(d, exist_ok=True)
        if os.path.exists(f"{d}/stems.npz"):
            continue
        st = {k: torch.from_numpy(t.targets[k].audio.T.astype(np.float32)) for k in STEMS}  # [2, n] @ 44.1k
        st = {k: resample(v, t.rate, SR).numpy() for k, v in st.items()}
        mix = resample(torch.from_numpy(t.audio.T.astype(np.float32)), t.rate, SR).numpy()
        np.savez(f"{d}/stems.npz", **st)
        write_wav(f"{d}/mix.wav", mix, SR)
        Path(f"{d}/name.txt").write_text(t.name)
        print(i, t.name, mix.shape)


def cmd_prepare_classifier_data(a):
    """Build the labelled router set: speech, SFX, speech-over-music, plus a ~4 min song.

    music       the 50 MUSDB mixes (read in place) + data/fullsong.wav
    speech      data/speech_say (macOS `say`, uploaded) + data/speech_libri (LibriSpeech dummy)
    sfx         data/sfx_esc50 (ESC-50: 2 clips from each of the 50 classes; none is music)
    hard        data/speech_over_music: speech over a MUSDB instrumental at -10/-20 dB
    """
    import urllib.request
    data = f"{a.work}/data"
    rng = np.random.default_rng(0)
    # LibriSpeech (HF dummy split; 73 utterances, ~10 MB)
    out = f"{data}/speech_libri"
    os.makedirs(out, exist_ok=True)
    if not glob.glob(f"{out}/*.wav"):
        import io
        import soundfile as sf
        from huggingface_hub import hf_hub_download, list_repo_files
        repo = "hf-internal-testing/librispeech_asr_dummy"
        files = [f for f in list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
        import pandas as pd
        for f in files:
            df = pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset"))
            for j, row in df.iterrows():
                wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
                sf.write(f"{out}/{row['id']}.wav", wav, sr)
    # ESC-50: 2 clips per class
    out = f"{data}/sfx_esc50"
    os.makedirs(out, exist_ok=True)
    if not glob.glob(f"{out}/*.wav"):
        base = "https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
        meta = urllib.request.urlopen(f"{base}/meta/esc50.csv").read().decode().splitlines()[1:]
        by = {}
        for line in meta:
            fn, fold, target, cat = line.split(",")[:4]
            by.setdefault(cat, []).append(fn)
        for cat, fns in sorted(by.items()):
            for fn in fns[:2]:
                urllib.request.urlretrieve(f"https://github.com/karolpiczak/ESC-50/raw/master/audio/{fn}", f"{out}/{cat}__{fn}")
    # speech over music (video-soundtrack-like hard case)
    out = f"{data}/speech_over_music"
    os.makedirs(out, exist_ok=True)
    speech = sorted(glob.glob(f"{data}/speech_say/*.wav"))
    dirs = clip_dirs(a.work)
    for j, sp in enumerate(speech[:10]):
        s = read_audio(sp, mono=True)[0]
        d = dirs[(7 * j) % len(dirs)]
        _, ref = load_clip(d)
        inst = (ref["drums"] + ref["bass"] + ref["other"]).mean(0)
        inst = np.tile(inst, int(np.ceil(len(s) / len(inst))))[:len(s)]
        for rel in (-10, -20):
            g = np.sqrt(np.mean(s ** 2) / max(np.mean(inst ** 2), 1e-12)) * 10 ** (rel / 20)
            x = s + g * inst
            write_wav(f"{out}/som{rel}_{j:02d}.wav", (x / max(1.0, np.abs(x).max() / 0.9))[None], SR)
    # ~4 min "song": the free MUSDB set has only 7 s previews, so concatenate 34 consecutive
    # previews (with 20 ms crossfades). Only used for timing / VRAM and as one more music file.
    fs = f"{data}/fullsong.wav"
    if not os.path.exists(fs):
        parts = []
        for d in dirs[:34]:
            m, _ = load_clip(d)
            parts.append(m)
        fade = int(0.02 * SR)
        song = parts[0]
        for p in parts[1:]:
            w = np.linspace(0, 1, fade, dtype=np.float32)
            song = np.concatenate([song[:, :-fade], song[:, -fade:] * (1 - w) + p[:, :fade] * w, p[:, fade:]], axis=1)
        write_wav(fs, song, SR)
        print("fullsong", song.shape[1] / SR, "s")
    for k in ("speech_say", "speech_libri", "sfx_esc50", "speech_over_music"):
        print(k, len(glob.glob(f"{data}/{k}/*.wav")))


# ----------------------------------------------------------------------------- separators

class DemucsRunner:
    def __init__(self, name: str, shifts: int, overlap: float, device: str, segment: float | None = None):
        self.name, self.shifts, self.overlap, self.device, self.segment = name, shifts, overlap, device, segment
        self.normalize = True   # mean/std input normalisation like the demucs CLI (StemForge: off)

    def load(self):
        from demucs.pretrained import get_model
        self.model = get_model(self.name)
        self.model.to(self.device).eval()
        self.sources = list(self.model.sources)
        assert self.model.samplerate == MODEL_SR

    def run44(self, x44):
        """x44: torch [2, n] at 44.1 kHz on cpu -> dict of torch [2, n] (normalised like the CLI)."""
        import torch
        from demucs.apply import apply_model
        ref = x44.mean(0)
        mean, std = (ref.mean(), ref.std() + 1e-8) if self.normalize else (torch.tensor(0.0), torch.tensor(1.0))
        with torch.inference_mode():
            out = apply_model(self.model, ((x44 - mean) / std)[None].to(self.device), shifts=self.shifts,
                              split=True, overlap=self.overlap, progress=False, device=self.device,
                              segment=self.segment)[0]
        out = (out * std + mean / len(self.sources)).cpu()   # keep sum(out) consistent with the mix
        return dict(zip(self.sources, out))


class SeparatorRunner:
    """python-audio-separator (file in, files out). Uses /dev/shm to keep I/O off the timing."""

    HEADROOM = 0.5

    def __init__(self, model_file: str, model_dir: str, mdxc_overlap: int | None = None, autocast: bool = False):
        self.model_file, self.model_dir, self.mdxc_overlap, self.autocast = model_file, model_dir, mdxc_overlap, autocast
        self.tmp = f"/dev/shm/mb-sep-{os.getpid()}"

    def load(self):
        import logging
        from audio_separator.separator import Separator
        os.makedirs(self.tmp, exist_ok=True)
        # normalization_threshold: the library peak-normalises the input mix AND every output stem
        # to this peak (default 0.9, max allowed 1.0) without undoing it, which breaks stem gains and
        # mixture consistency. We use 1.0 and keep the input peak <= HEADROOM (undone afterwards),
        # so no stem reaches 1.0 in practice. use_soundfile avoids the pydub/ffmpeg writer.
        self.sep = Separator(log_level=logging.WARNING, model_file_dir=self.model_dir, output_dir=self.tmp,
                             output_format="WAV", sample_rate=MODEL_SR, normalization_threshold=1.0,
                             use_soundfile=True, use_autocast=self.autocast,
                             mdxc_params={"segment_size": 256, "override_model_segment_size": False,
                                          "batch_size": None, "overlap": self.mdxc_overlap, "pitch_shift": 0})
        self.sep.load_model(model_filename=self.model_file)

    def run44(self, x44):
        import soundfile as sf
        import torch
        inp = f"{self.tmp}/in.wav"
        peak = float(x44.abs().max())
        g = self.HEADROOM / peak if peak > self.HEADROOM else 1.0
        sf.write(inp, (x44 * g).numpy().T, MODEL_SR, subtype="FLOAT")
        files = self.sep.separate(inp)
        out = {}
        for f in files:
            p = f if os.path.isabs(f) else os.path.join(self.tmp, f)
            m = re.search(r"_\(([^)]+)\)", os.path.basename(p))
            y, sr = sf.read(p, dtype="float32", always_2d=True)
            assert sr == MODEL_SR, sr
            out[m.group(1).lower()] = torch.from_numpy(y.T.copy()) / g
            os.remove(p)
        if set(out) == {"vocals", "other"}:    # some vocal models call the accompaniment "other"
            out["instrumental"] = out.pop("other")
        return out


class ZFTRoformerRunner:
    """ZFTurbo Music-Source-Separation-Training BS-RoFormer checkpoint built with lucidrains'
    ``bs-roformer`` package (StemForge's 4-stem path). Weights: /workspace/torch-cache/zft/<name>.{ckpt,yaml}.

    bs-roformer must be 0.3.x: >=1.0 changed the transformer blocks (hyper-connections, fused qkv)
    and silently leaves 444 tensors uninitialised with strict=False. The checkpoint's mask
    estimators use mlp_expansion_factor=2, which BSRoformer does not expose; we rebuild them.
    Inference = ZFTurbo demix: chunk = audio.chunk_size, step = chunk // num_overlap, triangular
    window, reflect pad by (chunk - step) when the track is long enough, divide by summed weights.
    """

    def __init__(self, name: str, device: str, fp16: bool = False, num_overlap: int | None = None,
                 root: str = "/workspace/torch-cache/zft"):
        self.name, self.device, self.fp16, self.num_overlap, self.root = name, device, fp16, num_overlap, root

    def load(self):
        import inspect
        import torch
        import yaml
        from torch import nn
        import bs_roformer.bs_roformer as B
        cfg = yaml.load(open(f"{self.root}/{self.name}.yaml"), Loader=yaml.FullLoader)
        mc = cfg["model"]
        sig = inspect.signature(B.BSRoformer.__init__).parameters
        kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in mc.items() if k in sig}
        self.popped = [k for k in mc if k not in sig]
        m = B.BSRoformer(**kw)
        if "mlp_expansion_factor" not in sig and mc.get("mlp_expansion_factor", 4) != 4:
            dims = m.mask_estimators[0].dim_inputs
            m.mask_estimators = nn.ModuleList([B.MaskEstimator(dim=mc["dim"], dim_inputs=dims, depth=mc["mask_estimator_depth"],
                                                               mlp_expansion_factor=mc["mlp_expansion_factor"])
                                               for _ in range(mc["num_stems"])])
        sd = torch.load(f"{self.root}/{self.name}.ckpt", map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        r = m.load_state_dict(sd, strict=False)
        self.load_report = {"popped_config_keys": self.popped, "missing_keys": len(r.missing_keys),
                            "unexpected_keys": len(r.unexpected_keys), "model_tensors": len(m.state_dict())}
        print("zft load", self.load_report, flush=True)
        if r.missing_keys or r.unexpected_keys:
            raise SystemExit(f"checkpoint does not match the model: {self.load_report}")
        self.model = m.to(self.device).eval()
        self.sources = list(cfg["training"]["instruments"])
        self.chunk = int(cfg["audio"]["chunk_size"])
        self.overlap_n = int(self.num_overlap or cfg["inference"].get("num_overlap", 2))
        assert int(cfg["audio"]["sample_rate"]) == MODEL_SR

    def run44(self, x44):
        import torch
        import torch.nn.functional as F
        C, step = self.chunk, self.chunk // self.overlap_n
        border = C - step
        x = x44.to(self.device)
        L = x.shape[-1]
        padded = L > 2 * border and border > 0
        if padded:
            x = F.pad(x[None], (border, border), mode="reflect")[0]
        L2 = x.shape[-1]
        half = (C + 1) // 2
        ramp = torch.arange(1, half + 1, device=self.device, dtype=torch.float32) / half
        win = torch.cat([ramp, ramp.flip(0)])[:C] if C % 2 == 0 else torch.cat([ramp, ramp.flip(0)[1:]])
        S = len(self.sources)
        res = torch.zeros(S, 2, L2, device=self.device)
        cnt = torch.zeros(L2, device=self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            i = 0
            while i < L2:
                part = x[:, i:i + C]
                n = part.shape[-1]
                if n < C:
                    mode = "reflect" if n > C // 2 + 1 else "constant"
                    part = F.pad(part[None], (0, C - n), mode=mode)[0]
                y = self.model(part[None])[0].float()          # [S, 2, C]
                w = win[:n]
                res[..., i:i + n] += y[..., :n] * w
                cnt[i:i + n] += w
                i += step
        res = res / cnt.clamp_min(1e-8)
        if padded:
            res = res[..., border:border + L]
        return dict(zip(self.sources, res.cpu()))


def to_four(est: dict, n: int, mix: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Map a model's stems onto vocals/drums/bass/other (guitar, piano, etc. -> other)."""
    out = {k: np.zeros((2, n), np.float32) for k in STEMS}
    for k, v in est.items():
        k = {"instrumental": None, "no_vocals": None}.get(k, k)
        if k is None:
            continue
        out[k if k in STEMS else "other"] += fit(np.asarray(v, np.float32), n)
    return out


def make_runner(spec: str, a):
    kind, _, arg = spec.partition(":")
    if kind == "demucs":
        r = DemucsRunner(arg, a.shifts, a.overlap, a.device, a.segment)
        r.normalize = getattr(a, "demucs_normalize", True)
        return r
    if kind == "sep":
        return SeparatorRunner(arg, a.model_dir, getattr(a, "mdxc_overlap", None), getattr(a, "sep_autocast", False))
    if kind == "zft":
        return ZFTRoformerRunner(arg, a.device, fp16=getattr(a, "fp16", False), num_overlap=getattr(a, "num_overlap", None))
    raise SystemExit(f"unknown model spec {spec}")


def split48(runners, mix48: np.ndarray, chain: bool, clip: bool = False) -> tuple[dict, dict]:
    """Run one model (or the chain) on a 48 kHz mixture, returning native-name stems at 48 kHz."""
    import torch
    n = mix48.shape[1]
    t = {}
    t0 = time.perf_counter()
    x44 = resample(torch.from_numpy(mix48), SR, MODEL_SR)
    if not chain:
        est44 = runners[0].run44(x44)
    else:
        voc = runners[0].run44(x44)
        sync()
        t["stage1_s"] = time.perf_counter() - t0
        v44 = fit(voc["vocals"], x44.shape[-1])
        inst44 = x44 - v44                      # instrumental = mix - vocals (exact complement)
        rest = runners[1].run44(inst44)
        est44 = {"vocals": v44}
        for k, val in rest.items():            # Demucs' "vocals" on the instrumental is leakage -> other
            est44["other_vocal_leak" if k == "vocals" else k] = val
    est = {k: fit(resample(v, MODEL_SR, SR).numpy(), n) for k, v in est44.items()}
    if clip:                                    # StemForge: each stem clipped to +-1 after resampling
        est = {k: np.clip(v, -1.0, 1.0) for k, v in est.items()}
    sync()
    t["total_s"] = time.perf_counter() - t0
    return est, t


def cmd_separate(a):
    import torch
    os.environ.setdefault("TORCH_HOME", "/workspace/torch-cache")
    torch.manual_seed(0)
    chain = a.model.startswith("chain:")
    specs = a.model[len("chain:"):].split("+") if chain else [a.model]
    tag = a.tag or re.sub(r"[^A-Za-z0-9_.+-]", "_", a.model) + (f"_s{a.shifts}" if a.shifts else "")
    out = f"{a.est}/{tag}"
    os.makedirs(out, exist_ok=True)
    runners = [make_runner(s, a) for s in specs]
    t0 = time.perf_counter()
    for r in runners:
        r.load()
    sync()
    load_s = time.perf_counter() - t0
    rows = []
    for d in clip_dirs(a.work, a.limit):
        mix, _ = load_clip(d)
        est, t = split48(runners, mix, chain, clip=getattr(a, "clip", False))
        np.savez(f"{out}/{os.path.basename(d)}.npz", **est)
        rows.append({"clip": os.path.basename(d), "seconds": mix.shape[1] / SR, **t})
        print(tag, rows[-1], flush=True)
    secs = [r["total_s"] for r in rows]
    meta = {"tag": tag, "model": a.model, "shifts": a.shifts, "overlap": a.overlap, "segment": a.segment,
            "load_s": load_s, "first_clip_s": secs[0], "mean_clip_s_excl_first": float(np.mean(secs[1:] or secs)),
            "audio_s_per_clip": rows[0]["seconds"], "rtf": float(np.sum(secs[1:]) / sum(r["seconds"] for r in rows[1:])) if len(rows) > 1 else None,
            **gpu_mem(), "stems_native": sorted(est.keys()),
            "config": json.load(open(a.config)) if getattr(a, "config", None) else None,
            "load_report": [getattr(r, "load_report", None) for r in runners], "clips": rows}
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    os.makedirs(f"{a.work}/results", exist_ok=True)
    json.dump({k: v for k, v in meta.items() if k != "clips"}, open(f"{a.work}/results/sep_{tag}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != "clips"}, indent=1))


def cmd_audiosep(a):
    """AudioSep text-prompt baseline, loaded in-process exactly like backends/audiosep.py:
    mono 48 kHz -> torchaudio resample to 32 kHz -> ss_model -> back to 48 kHz. One window
    (clips are 7 s; the service uses 10 s windows). Raw output, no mask refinement."""
    import torch
    import torchaudio
    root = Path(a.audiosep_root)
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([(np.core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
                                              type(np.dtype(np.float64)), type(np.dtype(np.float32)), type(np.dtype(np.int64))])
    sys.path.insert(0, str(root))
    cwd = os.getcwd()
    os.chdir(root)  # upstream code resolves some paths relative to the checkout
    from models.clap_encoder import CLAP_Encoder
    from utils import load_ss_model, parse_yaml
    t0 = time.perf_counter()
    enc = CLAP_Encoder(pretrained_path=str(root / "checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt")).eval()
    model = load_ss_model(parse_yaml(str(root / "config/audiosep_base.yaml")),
                          str(root / "checkpoint/audiosep_base_4M_steps.ckpt"), enc).eval().to(a.device)
    sync()
    load_s = time.perf_counter() - t0
    os.chdir(cwd)
    prompts = [p.strip() for p in a.prompts.split(",")]
    out = f"{a.est}/audiosep"
    os.makedirs(out, exist_ok=True)
    times = []
    for d in clip_dirs(a.work, a.limit):
        mix, _ = load_clip(d)
        m = torch.from_numpy(mix.mean(0))
        est = {}
        for p in prompts:
            t1 = time.perf_counter()
            x = torchaudio.functional.resample(m, SR, 32_000).to(a.device)
            with torch.inference_mode():
                cond = model.query_encoder.get_query_embed(modality="text", text=[p], device=a.device)
                y = model.ss_model({"mixture": x[None, None, :], "condition": cond})["waveform"].reshape(-1)
                y = torchaudio.functional.resample(y, 32_000, SR).float().cpu().numpy()
            sync()
            times.append(time.perf_counter() - t1)
            est[p] = fit(y[None], mix.shape[1])[0]
        np.savez(f"{out}/{os.path.basename(d)}.npz", **est)
        print(d, flush=True)
    meta = {"tag": "audiosep", "prompts": prompts, "load_s": load_s, "mean_prompt_s": float(np.mean(times[1:])),
            **gpu_mem()}
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
    os.makedirs(f"{a.work}/results", exist_ok=True)
    json.dump(meta, open(f"{a.work}/results/sep_audiosep.json", "w"), indent=1)
    print(meta)


# ----------------------------------------------------------------------------- scoring

def _sdr(ref, est):
    return 10 * np.log10((np.sum(ref ** 2) + 1e-12) / (np.sum((ref - est) ** 2) + 1e-12))


def metrics(ref: np.ndarray, est: np.ndarray, sr: int = SR) -> dict:
    """ref/est [..., n]; returns usdr/csdr (nan when the reference is silent throughout)."""
    n = ref.shape[-1]
    vals = []
    for s in range(0, n - sr // 2, sr):          # 1 s chunks; drop a trailing chunk < 0.5 s
        r, e = ref[..., s:s + sr], est[..., s:s + sr]
        if np.mean(r ** 2) > 1e-8:
            vals.append(_sdr(r, e))
    if not vals:
        return {"usdr": float("nan"), "csdr": float("nan")}
    return {"usdr": float(_sdr(ref, est)), "csdr": float(np.median(vals))}


def project(est: dict, mix: np.ndarray) -> dict:
    """Mixture-consistency projection: spread mix - sum(est) over stems by energy."""
    keys = list(est)
    e = np.array([np.sum(est[k] ** 2) for k in keys]) + 1e-12
    w = e / e.sum()
    err = mix - sum(est.values())
    return {k: est[k] + w[i] * err for i, k in enumerate(keys)}


def cmd_score(a):
    import csv
    os.makedirs(f"{a.work}/results", exist_ok=True)
    all_rows = []
    summary = {}
    for tag in a.tags:
        rows = []
        for d in clip_dirs(a.work, a.limit):
            f = f"{a.est}/{tag}/{os.path.basename(d)}.npz"
            if not os.path.exists(f):
                continue
            mix, ref = load_clip(d)
            z = np.load(f)
            if tag == "audiosep":
                refine = _service_refine()
                for p in z.files:
                    k = AUDIOSEP_PROMPT_STEM.get(p)
                    if k is None:
                        continue
                    ests = {f"prompt={p}": z[p]}
                    if refine is not None:   # the service's optional STFT ratio mask (off for audiosep by default)
                        ests[f"prompt={p}+mask"] = refine(mix.mean(0)[None], z[p][None])[0]
                    for vname, e in ests.items():
                        m = metrics(ref[k].mean(0), e)
                        rows.append({"tag": tag, "variant": vname, "clip": os.path.basename(d), "stem": k,
                                     "usdr_mono": m["usdr"], "csdr_mono": m["csdr"]})
                continue
            native = {k: z[k] for k in z.files}
            variants = {"raw": native, "proj": project(native, mix)}
            for vname, est in variants.items():
                four = to_four(est, mix.shape[1])
                cons = float(np.abs(sum(est.values()) - mix).max())
                vocals_only = set(est) <= {"vocals", "instrumental"}
                for k in STEMS:
                    if vocals_only and k != "vocals":
                        continue
                    m = metrics(ref[k], four[k])
                    mm = metrics(ref[k].mean(0), four[k].mean(0))
                    rows.append({"tag": tag, "variant": vname, "clip": os.path.basename(d), "stem": k,
                                 "usdr": m["usdr"], "csdr": m["csdr"], "usdr_mono": mm["usdr"], "csdr_mono": mm["csdr"],
                                 "max_abs_mix_err": cons})
        all_rows += rows
        for r in rows:
            key = (tag, r["variant"], r["stem"])
            summary.setdefault(key, []).append(r)
    agg = []
    for (tag, var, stem), rs in sorted(summary.items()):
        row = {"tag": tag, "variant": var, "stem": stem}
        for m in ("usdr", "csdr", "usdr_mono", "csdr_mono"):
            v = np.array([r[m] for r in rs if m in r and np.isfinite(r[m])])
            if len(v):
                row[f"{m}_mean"], row[f"{m}_median"], row["n"] = round(float(v.mean()), 3), round(float(np.median(v)), 3), len(v)
        if "max_abs_mix_err" in rs[0]:
            row["max_abs_mix_err"] = float(max(r["max_abs_mix_err"] for r in rs))
        agg.append(row)
    name = a.name or "scores"
    json.dump(agg, open(f"{a.work}/results/{name}_summary.json", "w"), indent=1)
    with open(f"{a.work}/results/{name}_per_clip.csv", "w", newline="") as fh:
        keys = sorted({k for r in all_rows for k in r}, key=lambda k: ["tag", "variant", "clip", "stem"].index(k) if k in ["tag", "variant", "clip", "stem"] else 9)
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    for r in agg:
        print(r)


def _service_refine():
    """sam_audio_service.refine.mask_refine if the service package is importable (repo checkout)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        from sam_audio_service.refine import mask_refine
        return mask_refine
    except Exception:
        return None


AUDIOSEP_PROMPT_STEM = {"vocals": "vocals", "singing": "vocals", "singing voice": "vocals",
                        "drums": "drums", "drum kit": "drums", "bass guitar": "bass", "bass": "bass"}


# ----------------------------------------------------------------------------- classifier

def labelled_files(work: str) -> list[tuple[str, str]]:
    files = [(f"{d}/mix.wav", "music") for d in clip_dirs(work)] + [(f"{work}/data/fullsong.wav", "music")]
    for sub, lab in (("speech_say", "speech"), ("speech_libri", "speech"), ("sfx_esc50", "sfx"),
                     ("speech_over_music", "speech_over_music")):
        files += [(f, lab) for f in sorted(glob.glob(f"{work}/data/{sub}/*.wav"))]
    return files


def windows(n: int, sr: int, win_s: float, k: int) -> list[int]:
    w = int(win_s * sr)
    if n <= w:
        return [0]
    k = min(k, int(np.ceil(n / w)))
    return [int(s) for s in np.linspace(0, n - w, k)]


class ASTTagger:
    sr, win_s = 16_000, 10.24
    model_id = "MIT/ast-finetuned-audioset-10-10-0.4593"

    def load(self, device):
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        self.fe = ASTFeatureExtractor.from_pretrained(self.model_id)
        self.model = ASTForAudioClassification.from_pretrained(self.model_id).to(device).eval()
        self.device = device
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def probs(self, segs: list[np.ndarray]) -> np.ndarray:
        import torch
        feats = self.fe(segs, sampling_rate=self.sr, return_tensors="pt")["input_values"].to(self.device)
        with torch.inference_mode():
            return torch.sigmoid(self.model(feats).logits).float().cpu().numpy()


class PANNsTagger:
    sr, win_s = 32_000, 10.0

    def load(self, device):
        from panns_inference import AudioTagging, labels
        self.at = AudioTagging(checkpoint_path=None, device=device)
        self.labels = list(labels)

    def probs(self, segs):
        w = int(self.win_s * self.sr)
        x = np.stack([fit(s[None], w)[0] for s in segs])
        clip, _ = self.at.inference(x)
        return np.asarray(clip)


def cmd_classify(a):
    import torch
    tagger = {"ast": ASTTagger, "panns": PANNsTagger}[a.tagger]()
    sync()
    t0 = time.perf_counter()
    tagger.load(a.device)
    sync()
    load_s = time.perf_counter() - t0
    li = {l: i for i, l in enumerate(tagger.labels)}
    music_i, speech_i = li["Music"], li["Speech"]
    rows = []
    for j, (f, lab) in enumerate(labelled_files(a.work)):
        t1 = time.perf_counter()
        x = read_audio(f, sr=tagger.sr, mono=True)[0]
        w = int(tagger.win_s * tagger.sr)
        segs = [x[s:s + w] for s in windows(len(x), tagger.sr, tagger.win_s, a.windows)]
        p = tagger.probs(segs)
        sync()
        dt = time.perf_counter() - t1
        rows.append({"file": f.split(a.work + "/")[-1], "label": lab, "dur_s": len(x) / tagger.sr, "n_windows": len(segs),
                     "music": float(p[:, music_i].mean()), "speech": float(p[:, speech_i].mean()),
                     "top": tagger.labels[int(p.mean(0).argmax())], "latency_s": dt})
        if j < 3 or j % 25 == 0:
            print(rows[-1], flush=True)
    # threshold sweep on music vs (speech + sfx); speech_over_music reported separately
    def acc(rule, thr, labs):
        sel = [r for r in rows if r["label"] in labs]
        pred = [rule(r) >= thr for r in sel]
        truth = [r["label"] == "music" for r in sel]
        return float(np.mean(np.array(pred) == np.array(truth))), sel, pred
    rules = {"music": lambda r: r["music"], "music_minus_speech": lambda r: r["music"] - r["speech"]}
    res = {"tagger": a.tagger, "load_s": load_s, "windows": a.windows, **gpu_mem(), "rules": {}}
    lat = [r["latency_s"] for r in rows[1:]]
    res["latency_warm_mean_s"], res["latency_warm_p95_s"] = float(np.mean(lat)), float(np.percentile(lat, 95))
    res["latency_first_s"] = rows[0]["latency_s"]
    res["latency_fullsong_s"] = next(r["latency_s"] for r in rows if r["file"].endswith("fullsong.wav"))
    for name, rule in rules.items():
        grid = np.round(np.arange(-0.9, 0.96, 0.01), 2)
        accs = [(acc(rule, t, ("music", "speech", "sfx"))[0], t) for t in grid]
        best = max(a_ for a_, _ in accs)
        ok = [t for a_, t in accs if a_ == best]
        thr = float(ok[len(ok) // 2])          # middle of the best plateau
        _, sel, pred = acc(rule, thr, ("music", "speech", "sfx"))
        conf = {}
        for r, p_ in zip(sel, pred):
            conf.setdefault(r["label"], {"pred_music": 0, "pred_not_music": 0})["pred_music" if p_ else "pred_not_music"] += 1
        som = [r for r in rows if r["label"] == "speech_over_music"]
        res["rules"][name] = {
            "threshold": thr, "best_plateau": [float(min(ok)), float(max(ok))], "accuracy": best, "confusion": conf,
            "errors": [{"file": r["file"], "label": r["label"], "score": round(rule(r), 3), "top": r["top"]}
                       for r, p_ in zip(sel, pred) if p_ != (r["label"] == "music")],
            "speech_over_music_pred_music": {k: sum(rule(r) >= thr for r in som if os.path.basename(r["file"]).startswith(k))
                                             for k in ("som-10", "som-20")},
            "accuracy_at_0.5": acc(rule, 0.5, ("music", "speech", "sfx"))[0] if name == "music" else None,
        }
    os.makedirs(f"{a.work}/results", exist_ok=True)
    json.dump(res, open(f"{a.work}/results/classify_{a.tagger}.json", "w"), indent=1)
    with open(f"{a.work}/results/classify_{a.tagger}_per_file.json", "w") as fh:
        json.dump(rows, fh, indent=0)
    print(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------- full song

def cmd_fullsong(a):
    import torch
    os.environ.setdefault("TORCH_HOME", "/workspace/torch-cache")
    chain = a.model.startswith("chain:")
    specs = a.model[len("chain:"):].split("+") if chain else [a.model]
    runners = [make_runner(s, a) for s in specs]
    t0 = time.perf_counter()
    for r in runners:
        r.load()
    sync()
    load_s = time.perf_counter() - t0
    mix = read_audio(a.file)
    dur = mix.shape[1] / SR
    runs = []
    for i in range(a.repeats):
        torch.cuda.reset_peak_memory_stats()
        est, t = split48(runners, mix, chain, clip=getattr(a, "clip", False))
        if a.project:
            t1 = time.perf_counter()
            est = project(est, mix)
            t["project_s"] = time.perf_counter() - t1
        t["max_abs_mix_err"] = float(np.abs(sum(est.values()) - mix).max())
        runs.append({**t, **gpu_mem()})
        print(runs[-1], flush=True)
    res = {"model": a.model, "shifts": a.shifts, "overlap": a.overlap, "file": a.file, "duration_s": dur,
           "load_s": load_s, "runs": runs, "rtf_warm": runs[-1]["total_s"] / dur}
    os.makedirs(f"{a.work}/results", exist_ok=True)
    tag = getattr(a, "tag", None) or re.sub(r"[^A-Za-z0-9_.+-]", "_", a.model) + f"_s{a.shifts}"
    res["tag"] = tag
    json.dump(res, open(f"{a.work}/results/fullsong_{tag}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def cmd_coload(a):
    """Same-process co-install check: AudioSep + the splitter (+ chain) + the tagger in ONE process,
    each run once on a clip; reports cumulative VRAM (what the service would hold)."""
    import torch
    os.environ.setdefault("TORCH_HOME", "/workspace/torch-cache")
    out = {"steps": []}
    def step(name):
        sync()
        out["steps"].append({"step": name, **gpu_mem(), "allocated_now_mb": round(torch.cuda.memory_allocated() / 2**20)})
        print(out["steps"][-1], flush=True)
    root = Path(a.audiosep_root)
    torch.serialization.add_safe_globals([(np.core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
                                          type(np.dtype(np.float64)), type(np.dtype(np.float32)), type(np.dtype(np.int64))])
    sys.path.insert(0, str(root))
    from models.clap_encoder import CLAP_Encoder
    from utils import load_ss_model, parse_yaml
    enc = CLAP_Encoder(pretrained_path=str(root / "checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt")).eval()
    asep = load_ss_model(parse_yaml(str(root / "config/audiosep_base.yaml")),
                         str(root / "checkpoint/audiosep_base_4M_steps.ckpt"), enc).eval().to(a.device)
    step("audiosep loaded")
    chain = a.model.startswith("chain:")
    runners = [make_runner(sp, a) for sp in (a.model[6:].split("+") if chain else [a.model])]
    for r in runners:
        r.load()
    step(f"splitter loaded ({a.model})")
    tagger = PANNsTagger()
    tagger.load(a.device)
    step("panns loaded")
    mix, _ = load_clip(clip_dirs(a.work)[0])
    est, t = split48(runners, mix, chain)
    step(f"split ran {t['total_s']:.2f}s")
    import torchaudio
    x = torchaudio.functional.resample(torch.from_numpy(mix.mean(0)), SR, 32_000).to(a.device)
    with torch.inference_mode():
        cond = asep.query_encoder.get_query_embed(modality="text", text=["dog barking"], device=a.device)
        asep.ss_model({"mixture": x[None, None, :], "condition": cond})
    step("audiosep ran")
    tagger.probs([read_audio(f"{clip_dirs(a.work)[0]}/mix.wav", sr=tagger.sr, mono=True)[0]])
    step("panns ran")
    import importlib.metadata as md
    out["versions"] = {p: md.version(p) for p in ("torch", "torchaudio", "demucs", "audio-separator", "bs-roformer",
                                                  "panns-inference", "transformers", "lightning", "numpy")}
    json.dump(out, open(f"{a.work}/results/coload.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------- cli

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", default="/workspace/music-bench")
    p.add_argument("--est", default="/root/mb-est")
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=None, help="only the first N clips")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare-musdb")
    sub.add_parser("prepare-classifier-data")

    def sep_args(q):
        q.add_argument("--model", default=None,
                       help="demucs:<name> | sep:<audio-separator model file> | chain:sep:<file>+demucs:<name>")
        q.add_argument("--config", default=None,
                       help="JSON file declaring a candidate: {tag, model, shifts, overlap, segment, "
                            "demucs_normalize, clip, mdxc_overlap, sep_autocast, fp16, num_overlap}; keys override the flags")
        q.add_argument("--shifts", type=int, default=0)
        q.add_argument("--overlap", type=float, default=0.25)
        q.add_argument("--segment", type=float, default=None)
        q.add_argument("--model-dir", default="/workspace/torch-cache/audio-separator")
    q = sub.add_parser("separate")
    sep_args(q)
    q.add_argument("--tag", default=None)
    q = sub.add_parser("fullsong")
    sep_args(q)
    q.add_argument("--file", default="/workspace/music-bench/data/fullsong.wav")
    q.add_argument("--repeats", type=int, default=2)
    q.add_argument("--project", action="store_true")
    q = sub.add_parser("coload")
    sep_args(q)
    q.add_argument("--audiosep-root", default="/workspace/AudioSep")
    q = sub.add_parser("audiosep")
    q.add_argument("--audiosep-root", default="/workspace/AudioSep")
    q.add_argument("--prompts", default="vocals,singing,drums,bass guitar,bass")
    q = sub.add_parser("score")
    q.add_argument("tags", nargs="+")
    q.add_argument("--name", default=None)
    q = sub.add_parser("classify")
    q.add_argument("--tagger", choices=["ast", "panns"], required=True)
    q.add_argument("--windows", type=int, default=4)
    a = p.parse_args(argv)
    if getattr(a, "config", None):
        for k, v in json.load(open(a.config)).items():
            setattr(a, k.replace("-", "_"), v)
    if a.cmd in ("separate", "fullsong", "coload") and not a.model:
        p.error("--model or --config with a model is required")
    {"prepare-musdb": cmd_prepare_musdb, "prepare-classifier-data": cmd_prepare_classifier_data,
     "separate": cmd_separate, "audiosep": cmd_audiosep, "score": cmd_score, "classify": cmd_classify,
     "fullsong": cmd_fullsong, "coload": cmd_coload}[a.cmd](a)


if __name__ == "__main__":
    main()
