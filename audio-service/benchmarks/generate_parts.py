#!/usr/bin/env python3
"""Phase E evaluation: add parts to a real song (from its stems) with every engine and report the
objective checks + write listening previews.

    python benchmarks/generate_parts.py --stems DIR --start 40 --seconds 75 --out RESULTS \
        [--engines symbolic,amt,musicgen] [--asks "add a piano part" ...]

DIR holds vocals.wav / drums.wav / bass.wav / other.wav (MUSDB layout) and optionally mixture.wav.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam_audio_service.audio_io import decode, encode_wav_pcm16  # noqa: E402
from sam_audio_service.generation import Generator, PartRequest  # noqa: E402
from sam_audio_service.generation.settings import GenSettings  # noqa: E402

DEFAULT_ASKS = ["add violins to the song", "add a piano part", "add a string pad", "add an 80s synth arpeggio", "add a funky bassline",
                "add a trumpet line in the chorus", "add brass stabs", "add a choir", "add a guitar riff"]


def mp3(path: Path, x: np.ndarray, sr: int, start: float, seconds: float, kbps: int = 96) -> None:
    seg = x[:, int(start * sr):int((start + seconds) * sr)]
    peak = np.abs(seg).max()
    if peak > 0.99:
        seg = seg * (0.99 / peak)
    wav = path.with_suffix(".tmp.wav")
    wav.write_bytes(encode_wav_pcm16(seg.astype(np.float32), sr))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-af", f"afade=t=in:d=0.3,afade=t=out:st={seconds - 0.6}:d=0.6",
                    "-codec:a", "libmp3lame", "-b:a", f"{kbps}k", str(path)], check=True)
    wav.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", required=True)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=75.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--engines", default="symbolic,amt,musicgen")
    ap.add_argument("--asks", nargs="*", default=DEFAULT_ASKS)
    ap.add_argument("--preview-start", type=float, default=None, help="seconds into the crop for the 20 s previews")
    ap.add_argument("--preview-seconds", type=float, default=20.0)
    ap.add_argument("--previews", default="", help="comma list of ask indices/engines to preview, e.g. 0:symbolic,1:amt")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = Path(args.stems)
    stems = {}
    for name in ("vocals", "drums", "bass", "other"):
        a = decode(path=d / f"{name}.wav")
        sr = a.sample_rate
        stems[name] = a.samples[:, int(args.start * sr):int((args.start + args.seconds) * sr)]
    mix = sum(stems.values())
    (out / "song.wav").write_bytes(encode_wav_pcm16(np.clip(mix, -1, 1), sr))
    gs = GenSettings()
    gs.allow_noncommercial = True  # evaluation only: MusicGen weights are CC-BY-NC-4.0
    gen = Generator(gs)
    t = time.perf_counter()
    analysis = gen.analyze(mix, sr, stems)
    t_an = time.perf_counter() - t
    analysis_mix = gen.analyze(mix, sr, None, melody=False)
    (out / "analysis.json").write_text(json.dumps(analysis, indent=1))
    (out / "analysis_mix_only.json").write_text(json.dumps(analysis_mix, indent=1))
    print(f"analysis {t_an:.1f}s tempo {analysis['tempo']} key {analysis['key']['label']} chords {len(analysis['chords'])} sections "
          f"{[(s['name'], s['start'], s['end']) for s in analysis['sections']]} melody notes {len(analysis['melody_midi'] or [])}")
    print("context", analysis["context"], "warnings", analysis["warnings"])
    results = {"song": {"stems": str(d), "start": args.start, "seconds": args.seconds}, "analysis_seconds": round(t_an, 2),
               "analysis_summary": {k: analysis[k] for k in ("tempo", "key", "space", "register", "context")}, "parts": []}
    engines = args.engines.split(",")
    for i, ask in enumerate(args.asks):
        for eng in engines:
            try:
                t = time.perf_counter()
                res = gen.generate(mix, sr, stems, PartRequest(prompt=ask, engine=eng, analysis=analysis, seed=i, audio_checks=True))
                took = time.perf_counter() - t
            except Exception as exc:  # noqa: BLE001
                print(f"[{eng}] {ask}: FAILED {type(exc).__name__}: {exc}")
                results["parts"].append({"ask": ask, "engine": eng, "error": f"{type(exc).__name__}: {exc}"})
                continue
            tag = f"{i:02d}_{eng}"
            (out / f"{tag}_part.wav").write_bytes(encode_wav_pcm16(res.audio, sr))
            if res.midi:
                (out / f"{tag}_part.mid").write_bytes(res.midi)
            row = {"ask": ask, "engine": eng, "instrument": res.spec.instrument, "role": res.spec.role, "renderer": (res.renderer or {}).get("used"),
                   "windows": res.windows, "seconds": round(took, 2), "elapsed": res.elapsed, "checks": res.checks, "mix": res.mix, "warnings": res.warnings,
                   "engine_info": res.engine}
            results["parts"].append(row)
            n = res.checks.get("notes", {})
            a = res.checks["audio"]
            print(f"[{eng:8s}] {ask:34s} {res.spec.instrument}/{res.spec.role} via {row['renderer']} {took:5.1f}s | notes ct {n.get('chord_tone_ratio')} key {n.get('in_key_ratio')} "
                  f"on30 {n.get('onsets_within_30ms')} clash {n.get('vocal_clash_ratio')} mask {n.get('vocal_masking_ratio')} | audio ct {a['chroma_chord_tone_ratio']} "
                  f"key {a['chroma_in_key_ratio']} on30 {a['audio_onsets_within_30ms']} LU {a.get('relative_lu')}")
            if args.preview_start is not None:
                mp3(out / f"{tag}_mix.mp3", np.clip(mix + res.audio, -1, 1), sr, args.preview_start, args.preview_seconds)
                mp3(out / f"{tag}_solo.mp3", res.audio, sr, args.preview_start, args.preview_seconds)
    if args.preview_start is not None:
        mp3(out / "song_original.mp3", mix, sr, args.preview_start, args.preview_seconds)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str))


if __name__ == "__main__":
    main()
