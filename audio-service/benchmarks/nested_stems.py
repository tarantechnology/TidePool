"""Phase 4 nested-stems benchmark: split the residual "other" stem into the instruments inside it.

Runs on the GPU pod, never in tests (heavy imports live inside functions). Every subcommand
writes JSON so the report can be regenerated from raw results (no audio is kept in the repo).

Ground truth: BabySlakh (the first 20 Slakh2100 tracks, CC-BY 4.0, zenodo 4603870), 16 kHz mono
per-instrument stems with Slakh ``inst_class`` labels. Clips are 20 s excerpts upsampled to
48 kHz (julius) and duplicated to stereo, so every model sees what the service would send it.
The 16 kHz source means nothing above 8 kHz is scored; see the report's caveats.

Families (``SLAKH_FAMILY``): guitar, piano (Slakh "Piano" = acoustic + electric piano,
harpsichord, clavinet), organ, synth (lead + pad), strings (solo + ensemble), brass,
woodwinds (reed + pipe), rest (chromatic percussion, ethnic, percussive, sound effects).

Conditions (the "other" stem the nested model receives):
  A  sum of every family except guitar      (parent split = 6-stem / "not vocals/drums/bass/guitar")
  B  sum of every family including guitar   (parent split = 4-stem htdemucs, the live config)
  E  htdemucs's "other" stem of the full Slakh mix (drums + bass + B): realistic bleed, scored
     against the ground-truth families of B
  X  like A without piano either (organ, synth, strings, brass, woodwinds, rest): every
     clip has guitar and piano, so A and X are where those two are absent

Metrics (mono, at 48 kHz, identical for every system):
  usdr / csdr   as REPORT.md (global SDR; median of 1 s chunk SDRs over non-silent chunks)
  mix_usdr      the same with the input mixture as the estimate; sdri = usdr - mix_usdr
  est_db        estimate energy relative to the input (dB): the presence signal. A family is
                *present* when its reference is >= PRESENT_DB of the input, *absent* when the
                track has none of it in the window (< ABSENT_DB); in between it is not scored.

Subcommands: prepare, prepare-e2e, queries, separate --system {demucs6s,banquet,audiosep,uvr_wind},
detect, score, fullsong.

Pod:
  export TORCH_HOME=/workspace/torch-cache HF_HOME=/workspace/hf; PY=/workspace/venv-nested/bin/python
  $PY nested_stems.py prepare --slakh /workspace/nested/data/babyslakh_16k
  $PY nested_stems.py prepare-e2e && $PY nested_stems.py queries
  $PY nested_stems.py separate --system demucs6s       (one process per system)
  $PY nested_stems.py detect && $PY nested_stems.py score
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from music_stems import MODEL_SR, SR, SeparatorRunner, fit, gpu_mem, metrics, read_audio, resample, sync  # noqa: E402

SLAKH_SR = 16_000
CLIP_S = 20.0
CLIPS_PER_TRACK = 2
PRESENT_DB = -25.0
ABSENT_DB = -60.0
FAMILIES = ("guitar", "piano", "organ", "synth", "strings", "brass", "woodwinds", "rest")
SLAKH_FAMILY = {
    "Guitar": "guitar", "Piano": "piano", "Organ": "organ", "Synth Lead": "synth", "Synth Pad": "synth",
    "Strings": "strings", "Strings (continued)": "strings", "Brass": "brass", "Reed": "woodwinds", "Pipe": "woodwinds",
    "Chromatic Percussion": "rest", "Ethnic": "rest", "Percussive": "rest", "Sound Effects": "rest", "Sound effects": "rest",
    "Synth Effects": "rest", "Drums": None, "Bass": None,
}
CONDITIONS = ("A", "B", "E", "X")
# Slakh's "Piano" class is acoustic + electric piano / harpsichord / clavinet; the detector splits
# those into piano and keys (both served by the same model output), so either counts.
DETECTED_AS = {"piano": ("piano", "keys")}


def detected(d: dict, family: str, threshold) -> bool:
    """Whether detection row ``d`` has ``family``: marked present (``threshold="present"``) or a
    family score >= ``threshold``."""
    names = DETECTED_AS.get(family, (family,))
    if threshold == "present":
        return any(n in d.get("present", ()) for n in names)
    return max(d.get("families", {}).get(n, 0.0) for n in names) >= threshold
# Families left out of each condition's input (their reference is silence: the presence test).
EXCLUDED = {"A": {"guitar"}, "B": set(), "E": set(), "X": {"guitar", "piano"}}
EST_ROOT = "/root/nested-est"
WORK = "/workspace/nested"


def _db(x: float, ref: float) -> float:
    return float(10 * np.log10(x / ref)) if x > 0 and ref > 0 else -120.0


def _e(x: np.ndarray) -> float:
    return float(np.sum(np.square(x, dtype=np.float64)))


def clip_paths(work: str) -> list[str]:
    return sorted(glob.glob(f"{work}/clips/*.npz"))


def load_clip(path: str, cond: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Mono 48 kHz input for ``cond`` and the reference per family (zeros when left out)."""
    z = np.load(path)
    refs = {f: (np.zeros_like(z[f]) if f in EXCLUDED[cond] else z[f]) for f in FAMILIES}
    if cond == "E":
        return np.load(path[:-4] + ".e2e.npy"), refs
    return sum(refs.values()).astype(np.float32), refs


def stereo(x: np.ndarray) -> np.ndarray:
    return np.repeat(x[None], 2, axis=0) if x.ndim == 1 else x


# ----------------------------------------------------------------------------- data prep

def _slakh_track(d: str) -> tuple[dict[str, np.ndarray], int]:
    import soundfile as sf
    import yaml
    meta = yaml.safe_load(open(f"{d}/metadata.yaml"))
    parts: dict[str, list[np.ndarray]] = {}
    n = 0
    for sid, info in meta["stems"].items():
        p = f"{d}/stems/{sid}.wav"
        if not os.path.exists(p):
            continue
        x, sr = sf.read(p, dtype="float32")
        assert sr == SLAKH_SR, sr
        x = x.mean(axis=1) if x.ndim == 2 else x
        n = max(n, len(x))
        family = SLAKH_FAMILY.get(info["inst_class"], "rest")
        parts.setdefault(family or "drums_bass", []).append(x)
    out = {k: sum(fit(x[None], n)[0] for x in xs).astype(np.float32) for k, xs in parts.items()}
    for f in FAMILIES:
        out.setdefault(f, np.zeros(n, np.float32))
    out.setdefault("drums_bass", np.zeros(n, np.float32))
    return out, n


def cmd_prepare(a):
    """20 s excerpts (CLIPS_PER_TRACK per track, the windows with the most active families,
    then the most "other" energy), resampled 16 -> 48 kHz."""
    import torch
    os.makedirs(f"{a.work}/clips", exist_ok=True)
    index = []
    for d in sorted(glob.glob(f"{a.slakh}/Track*")):
        track = os.path.basename(d)
        fam, n = _slakh_track(d)
        size, hop = int(CLIP_S * SLAKH_SR), int(5 * SLAKH_SR)
        cands = []
        for s in range(0, max(1, n - size), hop):
            other = sum(fam[f][s:s + size] for f in FAMILIES)
            e_other = _e(other)
            if e_other <= 0:
                continue
            active = sum(_db(_e(fam[f][s:s + size]), e_other) >= PRESENT_DB for f in FAMILIES)
            cands.append((active, e_other, s))
        cands.sort(reverse=True)
        chosen: list[int] = []
        for _, _, s in cands:
            if all(abs(s - c) >= size for c in chosen):
                chosen.append(s)
            if len(chosen) == CLIPS_PER_TRACK:
                break
        for k, s in enumerate(sorted(chosen)):
            seg = {key: fit(v[s:s + size][None], size)[0] for key, v in fam.items()}
            up = {key: resample(torch.from_numpy(v.copy()), SLAKH_SR, SR).numpy().astype(np.float32) for key, v in seg.items()}
            name = f"{track}_{k}"
            np.savez(f"{a.work}/clips/{name}.npz", **up)
            other = sum(up[f] for f in FAMILIES)
            row = {"clip": name, "track": track, "start_s": s / SLAKH_SR,
                   "family_db": {f: round(_db(_e(up[f]), _e(other)), 2) for f in FAMILIES}}
            index.append(row)
            print(row, flush=True)
    json.dump(index, open(f"{a.work}/clips/index.json", "w"), indent=1)
    os.makedirs(f"{a.work}/results", exist_ok=True)
    json.dump(index, open(f"{a.work}/results/clips_index.json", "w"), indent=1)


def cmd_prepare_e2e(a):
    """Condition E: htdemucs on the full mix (drums + bass + every family), keep its "other"."""
    import torch
    from music_stems import DemucsRunner
    r = DemucsRunner("htdemucs", 0, 0.25, a.device)
    r.load()
    for p in clip_paths(a.work):
        z = np.load(p)
        full = stereo(sum(z[f] for f in FAMILIES) + z["drums_bass"])
        x44 = resample(torch.from_numpy(full), SR, MODEL_SR)
        out = r.run44(x44)
        other = fit(resample(out["other"], MODEL_SR, SR).numpy(), full.shape[1]).mean(0).astype(np.float32)
        np.save(p[:-4] + ".e2e.npy", other)
        print(p, round(_db(_e(other), _e(sum(z[f] for f in FAMILIES))), 2), flush=True)


def cmd_queries(a):
    """Banquet query bank: per family, the 10 s window with the most energy in the two
    tracks where that family is loudest. A clip never uses a query from its own track."""
    import torch
    index = json.load(open(f"{a.work}/clips/index.json"))
    bank: dict[str, list[dict]] = {}
    tracks = sorted({r["track"] for r in index})
    size = int(10 * SLAKH_SR)
    os.makedirs(f"{a.work}/queries", exist_ok=True)
    for f in FAMILIES:
        if f == "rest":
            continue
        best = []
        for t in tracks:
            fam, n = _slakh_track(f"{a.slakh}/{t}")
            x = fam[f]
            if _e(x) <= 0:
                continue
            starts = list(range(0, max(1, len(x) - size), SLAKH_SR))
            energies = [_e(x[s:s + size]) for s in starts]
            i = int(np.argmax(energies))
            # prefer a solo-ish window: the family's share of the track's non-drum mix
            best.append((energies[i], t, starts[i], x[starts[i]:starts[i] + size].copy()))
        best.sort(key=lambda b: -b[0])
        bank[f] = []
        for e, t, s, x in best[:2]:
            q = resample(torch.from_numpy(fit(x[None], size)[0]), SLAKH_SR, MODEL_SR).numpy().astype(np.float32)
            path = f"{a.work}/queries/{f}_{t}.npy"
            np.save(path, q)
            bank[f].append({"track": t, "start_s": s / SLAKH_SR, "path": path})
        print(f, bank[f], flush=True)
    json.dump(bank, open(f"{a.work}/queries/bank.json", "w"), indent=1)


# ----------------------------------------------------------------------------- systems

class Demucs6s:
    """htdemucs_6s on the "other" stem: guitar and piano; drums/bass/vocals/other -> rest."""
    families = ("guitar", "piano")

    def __init__(self, a):
        from music_stems import DemucsRunner
        self.r = DemucsRunner("htdemucs_6s", 0, a.overlap, a.device)

    def load(self):
        self.r.load()

    def run(self, x48: np.ndarray, families) -> dict[str, np.ndarray]:
        import torch
        x44 = resample(torch.from_numpy(stereo(x48)), SR, MODEL_SR)
        out = self.r.run44(x44)
        n = x48.shape[-1]
        return {f: fit(resample(out[f], MODEL_SR, SR).numpy(), n) for f in self.families if f in families}


class Banquet:
    """query-bandit (Watcharasupat & Lerch, ISMIR 2024): one decoder, conditioned by the PaSST
    embedding of a 10 s query clip of the target instrument (here: from another BabySlakh track)."""
    families = ("guitar", "piano", "organ", "synth", "strings", "brass", "woodwinds")
    MIN_S = 36.0

    def __init__(self, a):
        self.a = a

    def load(self):
        import torch
        root = self.a.banquet_root
        os.environ.setdefault("CONFIG_ROOT", f"{root}/config")
        sys.path.insert(0, root)
        cwd = os.getcwd()
        os.chdir(root)
        import train as bt  # query-bandit's train.py
        from core.models.ebase import EndToEndLightningSystem
        config = bt._load_config(f"{root}/expt/bandit-everything-test.yml")
        config.data.inference_kwargs.batch_size = self.a.banquet_batch
        config.data.inference_kwargs.hop_size_seconds = self.a.banquet_hop
        self.system = EndToEndLightningSystem.load_from_checkpoint(
            self.a.banquet_ckpt, strict=True, model=bt._build_model(config), loss_handler=bt._build_loss(config),
            metrics=bt._dummy_metrics(config), augmentation_handler=bt._dummy_augmentation(),
            inference_handler=config.data.inference_kwargs, optimization_bundle=bt._build_optimization_bundle(config),
            fast_run=config.fast_run, batch_size=config.data.batch_size,
            effective_batch_size=config.data.get("effective_batch_size", None), commitment_weight=config.get("commitment_weight", 1.0),
            map_location="cpu",
        ).to(self.a.device).eval()
        os.chdir(cwd)
        self.bank = json.load(open(f"{self.a.work}/queries/bank.json"))
        self.torch = torch

    def query(self, family: str, track: str | None):
        for q in self.bank[family]:
            if q["track"] != track:
                return np.load(q["path"]), q["track"]
        raise RuntimeError(f"no query for {family}")

    def run(self, x48: np.ndarray, families, track: str | None = None) -> dict[str, np.ndarray]:
        torch = self.torch
        x44 = resample(torch.from_numpy(stereo(x48)), SR, MODEL_SR)
        n, n44 = x48.shape[-1], x44.shape[-1]
        # chunked_inference reflect-pads up to ~6 x (chunk - hop) on the right, which fails on
        # short inputs; zero-pad the end to MIN_S and crop afterwards.
        pad = max(0, int(self.MIN_S * MODEL_SR) - n44)
        x44p = torch.nn.functional.pad(x44, (0, pad))
        out = {}
        for f in families:
            if f not in self.families:
                continue
            q, _ = self.query(f, track)
            qs = torch.from_numpy(stereo(q))[None].to(self.a.device)
            batch = {"mixture": {"audio": x44p[None].to(self.a.device)}, "query": {"audio": qs},
                     "metadata": {"stem": ["target"]}, "estimates": {}}
            with torch.inference_mode():
                est = self.system.chunked_inference(batch)["estimates"]["target"]["audio"][0].float().cpu()
            est = est[..., :n44]
            out[f] = fit(resample(est, MODEL_SR, SR).numpy(), n)
        return out


AUDIOSEP_PROMPTS = {"guitar": "guitar", "piano": "piano", "organ": "organ", "synth": "synthesizer", "strings": "string section",
                    "brass": "brass instruments", "woodwinds": "woodwind instruments"}


class AudioSep:
    """AudioSep text prompts, mono 32 kHz in-process (as backends/audiosep.py); the floor."""
    families = tuple(AUDIOSEP_PROMPTS)

    def __init__(self, a):
        self.a = a

    def load(self):
        import torch
        root = Path(self.a.audiosep_root)
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([(np.core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
                                                  type(np.dtype(np.float64)), type(np.dtype(np.float32)), type(np.dtype(np.int64))])
        sys.path.insert(0, str(root))
        cwd = os.getcwd()
        os.chdir(root)
        from models.clap_encoder import CLAP_Encoder
        from utils import load_ss_model, parse_yaml
        enc = CLAP_Encoder(pretrained_path=str(root / "checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt")).eval()
        self.model = load_ss_model(parse_yaml(str(root / "config/audiosep_base.yaml")),
                                   str(root / "checkpoint/audiosep_base_4M_steps.ckpt"), enc).eval().to(self.a.device)
        os.chdir(cwd)

    def run(self, x48: np.ndarray, families) -> dict[str, np.ndarray]:
        import torch
        import torchaudio
        m = torch.from_numpy(x48 if x48.ndim == 1 else x48.mean(0))
        x = torchaudio.functional.resample(m, SR, 32_000).to(self.a.device)
        out = {}
        for f in families:
            if f not in AUDIOSEP_PROMPTS:
                continue
            with torch.inference_mode():
                cond = self.model.query_encoder.get_query_embed(modality="text", text=[AUDIOSEP_PROMPTS[f]], device=self.a.device)
                y = self.model.ss_model({"mixture": x[None, None, :], "condition": cond})["waveform"].reshape(-1)
                y = torchaudio.functional.resample(y, 32_000, SR).float().cpu().numpy()
            out[f] = fit(y[None], m.shape[-1])[0]
        return out


class UvrWind:
    """UVR "17_HP-Wind_Inst-UVR" (VR arch) through audio-separator: woodwinds vs the rest."""
    families = ("woodwinds",)

    def __init__(self, a):
        self.r = SeparatorRunner("17_HP-Wind_Inst-UVR.pth", a.model_dir)

    def load(self):
        self.r.load()

    def run(self, x48: np.ndarray, families) -> dict[str, np.ndarray]:
        import torch
        if "woodwinds" not in families:
            return {}
        x44 = resample(torch.from_numpy(stereo(x48)), SR, MODEL_SR)
        out = self.r.run44(x44)
        key = next(k for k in out if k.replace(" ", "") in {"woodwinds", "woodwind"})
        return {"woodwinds": fit(resample(out[key], MODEL_SR, SR).numpy(), x48.shape[-1])}


SYSTEMS = {"demucs6s": Demucs6s, "banquet": Banquet, "audiosep": AudioSep, "uvr_wind": UvrWind}


def cmd_separate(a):
    import torch
    torch.manual_seed(0)
    system = SYSTEMS[a.system](a)
    t0 = time.perf_counter()
    system.load()
    sync()
    load_s = time.perf_counter() - t0
    index = {r["clip"]: r for r in json.load(open(f"{a.work}/clips/index.json"))}
    conds = a.conds.split(",")
    times = []
    for p in clip_paths(a.work)[: a.limit]:
        name = Path(p).stem
        for cond in conds:
            mix, refs = load_clip(p, cond)
            fams = list(system.families)
            t1 = time.perf_counter()
            kw = {"track": index[name]["track"]} if a.system == "banquet" else {}
            est = system.run(mix, fams, **kw)
            sync()
            times.append({"clip": name, "cond": cond, "s": time.perf_counter() - t1, "families": len(est)})
            out = f"{a.est}/{a.tag or a.system}/{cond}"
            os.makedirs(out, exist_ok=True)
            np.savez(f"{out}/{name}.npz", **{f: (v.mean(0) if v.ndim == 2 else v).astype(np.float32) for f, v in est.items()})
            print(a.system, times[-1], flush=True)
    meta = {"system": a.system, "tag": a.tag or a.system, "load_s": load_s, "families": list(system.families),
            "mean_clip_s": float(np.mean([t["s"] for t in times[1:]] or [times[0]["s"]])), "clip_audio_s": CLIP_S, **gpu_mem(),
            "args": {k: v for k, v in vars(a).items() if isinstance(v, (str, int, float, type(None)))}}
    os.makedirs(f"{a.work}/results", exist_ok=True)
    json.dump(meta, open(f"{a.work}/results/sep_{a.tag or a.system}.json", "w"), indent=1)
    print(json.dumps(meta, indent=1))


# ----------------------------------------------------------------------------- detection

def _tagger(device: str):
    from sam_audio_service.classifier import PannsClassifier
    t = PannsClassifier(device=device)
    t.load()
    return t


def cmd_detect(a):
    """The service's detection path: ``labeling.detect_instruments`` on the input with the
    router's PANNs tagger (calibrated per-family presence thresholds). Family scores are kept
    so fixed thresholds can be swept in ``score`` too."""
    from sam_audio_service.labeling import detect_instruments
    tagger = _tagger(a.device)
    rows = []
    for p in clip_paths(a.work)[: a.limit]:
        for cond in a.conds.split(","):
            mix, refs = load_clip(p, cond)
            m = stereo(mix)
            t0 = time.perf_counter()
            found = detect_instruments(tagger, m, SR, parent="other", max_windows=12)
            rows.append({"clip": Path(p).stem, "cond": cond, "s": time.perf_counter() - t0, "instruments": [i.as_dict() for i in found],
                         "families": {i.family: round(i.score, 4) for i in found}, "present": [i.family for i in found if i.present]})
            print(rows[-1]["clip"], cond, rows[-1]["present"], rows[-1]["families"], flush=True)
    os.makedirs(f"{a.work}/results", exist_ok=True)
    json.dump(rows, open(f"{a.work}/results/detect_panns.json", "w"), indent=1)


# ----------------------------------------------------------------------------- scoring

def cmd_score(a):
    rows = []
    for p in clip_paths(a.work)[: a.limit]:
        name = Path(p).stem
        for cond in CONDITIONS:
            mix, refs = load_clip(p, cond)
            e_mix = _e(mix)
            for tag in a.tags:
                f_est = f"{a.est}/{tag}/{cond}/{name}.npz"
                if not os.path.exists(f_est):
                    continue
                z = np.load(f_est)
                for f in z.files:
                    ref, est = refs[f], z[f]
                    ref_db = _db(_e(ref), e_mix)
                    row = {"tag": tag, "cond": cond, "clip": name, "family": f, "ref_db": round(ref_db, 2),
                           "est_db": round(_db(_e(est), e_mix), 2)}
                    if ref_db >= PRESENT_DB:
                        m, mm = metrics(ref, est), metrics(ref, mix)
                        row.update(state="present", usdr=m["usdr"], csdr=m["csdr"], mix_usdr=mm["usdr"], sdri=m["usdr"] - mm["usdr"])
                    elif ref_db < ABSENT_DB:
                        row["state"] = "absent"
                    else:
                        row["state"] = "trace"
                    rows.append(row)
    agg = []
    keys = sorted({(r["tag"], r["cond"], r["family"]) for r in rows})
    for tag, cond, fam in keys:
        rs = [r for r in rows if (r["tag"], r["cond"], r["family"]) == (tag, cond, fam)]
        pres = [r for r in rs if r["state"] == "present"]
        absn = [r for r in rs if r["state"] == "absent"]
        row = {"tag": tag, "cond": cond, "family": fam, "n_present": len(pres), "n_absent": len(absn)}
        if pres:
            for m in ("usdr", "csdr", "sdri", "mix_usdr"):
                v = np.array([r[m] for r in pres if np.isfinite(r[m])])
                row[f"{m}_mean"] = round(float(v.mean()), 2)
                row[f"{m}_median"] = round(float(np.median(v)), 2)
            row["present_est_db_p10"] = round(float(np.percentile([r["est_db"] for r in pres], 10)), 2)
        if absn:
            v = np.array([r["est_db"] for r in absn])
            row["absent_est_db_median"] = round(float(np.median(v)), 2)
            row["absent_est_db_p90"] = round(float(np.percentile(v, 90)), 2)
            row["absent_est_db_max"] = round(float(v.max()), 2)
        for thr in (-30, -25, -20, -15, -12, -10):
            if absn:
                row[f"fp@{thr}"] = round(float(np.mean([r["est_db"] >= thr for r in absn])), 3)
            if pres:
                row[f"miss@{thr}"] = round(float(np.mean([r["est_db"] < thr for r in pres])), 3)
        agg.append(row)
    det = _score_detection(a) if os.path.exists(f"{a.work}/results/detect_panns.json") else None
    policy = _score_policy(a, rows)
    os.makedirs(f"{a.work}/results", exist_ok=True)
    json.dump({"present_db": PRESENT_DB, "absent_db": ABSENT_DB, "summary": agg, "detection": det, "policy": policy},
              open(f"{a.work}/results/{a.name}_summary.json", "w"), indent=1)
    json.dump(rows, open(f"{a.work}/results/{a.name}_per_clip.json", "w"))
    for r in agg:
        print({k: v for k, v in r.items() if not k.startswith(("fp@", "miss@"))})
    if det:
        print(json.dumps(det, indent=1))


def _score_policy(a, rows: list[dict]) -> list[dict]:
    """What the service would return: a family's track is kept when the tagger scores it
    ``>= min_score`` (or always, ``min_score`` None = options.stems) and its estimate is
    ``>= silent_db`` of the input. Per (tag, cond, family, min_score, silent_db): how often a
    present family is returned and how good it is then, and how often an absent one is."""
    f_det = f"{a.work}/results/detect_panns.json"
    det = {(d["clip"], d["cond"]): d for d in json.load(open(f_det))} if os.path.exists(f_det) else {}
    out = []
    keys = sorted({(r["tag"], r["cond"], r["family"]) for r in rows})
    for tag, cond, fam in keys:
        rs = [r for r in rows if (r["tag"], r["cond"], r["family"]) == (tag, cond, fam) and r["state"] != "trace"]
        for min_score in (None, "present", 0.05, 0.1, 0.2):
            if min_score is not None and not any((r["clip"], cond) in det for r in rs):
                continue
            for silent_db in (-30.0, -25.0, -20.0, -15.0, -12.0):
                def kept(r):
                    tagged = min_score is None or detected(det.get((r["clip"], cond), {}), fam, min_score)
                    return tagged and r["est_db"] >= silent_db
                pres = [r for r in rs if r["state"] == "present"]
                absn = [r for r in rs if r["state"] == "absent"]
                kp = [r for r in pres if kept(r)]
                row = {"tag": tag, "cond": cond, "family": fam, "min_score": min_score, "silent_db": silent_db,
                       "n_present": len(pres), "n_absent": len(absn),
                       "present_returned": round(len(kp) / len(pres), 3) if pres else None,
                       "absent_returned": round(sum(kept(r) for r in absn) / len(absn), 3) if absn else None}
                if kp:
                    row["returned_usdr_mean"] = round(float(np.mean([r["usdr"] for r in kp])), 2)
                    row["returned_sdri_mean"] = round(float(np.mean([r["sdri"] for r in kp])), 2)
                    row["returned_csdr_median"] = round(float(np.median([r["csdr"] for r in kp if np.isfinite(r["csdr"])])), 2)
                out.append(row)
    return out


def _score_detection(a) -> dict:
    """Precision/recall per family of "tag score >= t" against presence (ref >= PRESENT_DB),
    with "trace" families (between ABSENT_DB and PRESENT_DB) left out."""
    det = json.load(open(f"{a.work}/results/detect_panns.json"))
    paths = {Path(p).stem: p for p in clip_paths(a.work)}
    out = {}
    for cond in sorted({d["cond"] for d in det}):
        labels = []
        for d in det:
            if d["cond"] != cond:
                continue
            mix, refs = load_clip(paths[d["clip"]], cond)
            e_mix = _e(mix)
            for f in refs:
                if f == "rest":
                    continue
                db = _db(_e(refs[f]), e_mix)
                state = "present" if db >= PRESENT_DB else ("absent" if db < ABSENT_DB else "trace")
                labels.append((f, state, d))
        res = {}
        for t in ("present", 0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
            per = {}
            for f in sorted({l[0] for l in labels}):
                ls = [l for l in labels if l[0] == f and l[1] != "trace"]
                tp = sum(l[1] == "present" and detected(l[2], f, t) for l in ls)
                fp = sum(l[1] == "absent" and detected(l[2], f, t) for l in ls)
                fn = sum(l[1] == "present" and not detected(l[2], f, t) for l in ls)
                per[f] = {"tp": tp, "fp": fp, "fn": fn, "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                          "recall": round(tp / (tp + fn), 3) if tp + fn else None}
            res[str(t)] = per
        out[cond] = res
    return out


# ----------------------------------------------------------------------------- full song

def cmd_fullsong(a):
    """Chain on a real song: htdemucs (4-stem, the live parent split) -> "other" -> nested
    system. Lists what comes out (energy and tags per family) and times a 4-minute "other"."""
    import torch
    from music_stems import DemucsRunner
    from sam_audio_service.labeling import detect_instruments, label_stems
    mix = read_audio(a.file)
    parent = DemucsRunner("htdemucs", 0, 0.25, a.device)
    parent.load()
    out = parent.run44(resample(torch.from_numpy(mix), SR, MODEL_SR))
    other = fit(resample(out["other"], MODEL_SR, SR).numpy(), mix.shape[1]).astype(np.float32)
    del parent
    torch.cuda.empty_cache()
    tagger = _tagger(a.device)
    lab = label_stems(tagger, [("other", other)], other, SR)[0]
    fams = [i.as_dict() for i in detect_instruments(tagger, other, SR, parent="other")]
    system = SYSTEMS[a.system](a)
    t0 = time.perf_counter()
    system.load()
    sync()
    load_s = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()
    wanted = [f for f in system.families]
    runs = []
    for _ in range(2):  # cold, warm
        t1 = time.perf_counter()
        est = system.run(other, wanted) if a.system != "banquet" else system.run(other, wanted, track=None)
        sync()
        runs.append(round(time.perf_counter() - t1, 2))
    run_s = runs[-1]
    e_in = _e(other)
    rows = {}
    for f, v in est.items():
        v2 = stereo(v)
        l2 = label_stems(tagger, [("other", v2)], other, SR)[0]
        rows[f] = {"est_db": round(_db(_e(v2), e_in), 2), "tags": l2.tags, "name": l2.name}
    res = {"system": a.system, "file": a.file, "duration_s": mix.shape[1] / SR, "load_s": load_s, "run_s": run_s, "runs_s": runs, **gpu_mem(),
           "other_tags": lab.tags, "other_name": lab.name, "detected_families": fams, "outputs": rows}
    os.makedirs(f"{a.work}/results", exist_ok=True)
    json.dump(res, open(f"{a.work}/results/fullsong_{a.tag or a.system}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--est", default=EST_ROOT)
    ap.add_argument("--device", default="cuda")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--slakh", default=f"{WORK}/data/babyslakh_16k")
    sub.add_parser("prepare-e2e")
    p = sub.add_parser("queries")
    p.add_argument("--slakh", default=f"{WORK}/data/babyslakh_16k")
    for name in ("separate", "fullsong"):
        p = sub.add_parser(name)
        p.add_argument("--system", choices=sorted(SYSTEMS), required=True)
        p.add_argument("--tag")
        p.add_argument("--conds", default="A,B,E,X")
        p.add_argument("--limit", type=int)
        p.add_argument("--overlap", type=float, default=0.25)
        p.add_argument("--banquet-root", default=f"{WORK}/src/query-bandit")
        p.add_argument("--banquet-ckpt", default=f"{WORK}/ckpt/ev-pre-aug.ckpt")
        p.add_argument("--banquet-hop", type=float, default=0.5)
        p.add_argument("--banquet-batch", type=int, default=12)
        p.add_argument("--audiosep-root", default="/workspace/AudioSep")
        p.add_argument("--model-dir", default="/workspace/torch-cache/audio-separator")
        p.add_argument("--file", default="/workspace/music-bench/data/fullsong.wav")
    p = sub.add_parser("detect")
    p.add_argument("--conds", default="A,B,E,X")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("score")
    p.add_argument("tags", nargs="+")
    p.add_argument("--name", default="nested")
    p.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    {"prepare": cmd_prepare, "prepare-e2e": cmd_prepare_e2e, "queries": cmd_queries, "separate": cmd_separate,
     "detect": cmd_detect, "score": cmd_score, "fullsong": cmd_fullsong}[a.cmd](a)


if __name__ == "__main__":
    main()
