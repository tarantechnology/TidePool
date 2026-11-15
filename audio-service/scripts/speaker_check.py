"""Check Explode's speaker split on a real clip, on the box.

    python scripts/speaker_check.py clip.mp4 [more clips...] [--out DIR] [--no-demucs]

For each clip: Demucs dialogue (as Explode does), then the speaker split. Prints how many
people were found, how long each talks, every turn, and how alike the clusters' voices are
(centroid cosine; AUDIO_SPEAKER_SAME, default 0.70, is the merge line). Writes
<clip>_person<N>.wav into --out so you can listen to each person.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam_audio_service import speakers as S  # noqa: E402
from sam_audio_service.explode import DemucsVocals  # noqa: E402


def load(path: Path, rate: int = 44100) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "2", "-ar", str(rate), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def save(path: Path, x: np.ndarray, rate: int) -> None:
    with tempfile.NamedTemporaryFile(suffix=".f32") as f:
        f.write(np.ascontiguousarray(x.T, dtype=np.float32).tobytes())
        f.flush()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le", "-ar", str(rate), "-ac", str(x.shape[0]), "-i", f.name, str(path)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("speaker-check"))
    ap.add_argument("--no-demucs", action="store_true", help="split the raw mix (only for speech-only clips)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    enc, demucs, rate = S.VoiceEncoder(), DemucsVocals(), 44100
    print("speaker model available:", enc.available, "| demucs:", demucs.available)
    for clip in args.clips:
        mix = load(clip, rate)
        dialogue = mix if args.no_demucs else demucs.vocals(mix, rate)
        # Diagnostics: the clusters before merging and how alike they are.
        mono = S._resample(dialogue.mean(0), rate, S.RATE)
        active = S.speech_frames(mono)
        win, step = int(S.WINDOW_S * S.RATE // S.HOP), int(S.WINDOW_HOP_S * S.RATE // S.HOP)
        starts = [s for s in range(0, max(1, len(active) - win + 1), step) if active[s:s + win].mean() >= 0.5]
        if len(starts) >= 4:
            emb = enc.embed(np.stack([np.pad(mono, (0, win * S.HOP))[s * S.HOP:(s + win) * S.HOP] for s in starts]))
            labels = S._cluster(emb, 4)
            ids, sim = S._centroid_sims(emb, labels)
            pairs = [f"{a}-{b}:{sim[a, b]:.2f}" for a in range(len(ids)) for b in range(a + 1, len(ids))]
            print(f"\n== {clip.name}: speech {active.mean() * 100:.0f}% of {len(active) / 100:.0f}s, "
                  f"proposed {len(ids)} clusters {np.bincount(labels).tolist()}, centroid cosine {' '.join(pairs)}")
        out = S.split_speakers(dialogue, rate, enc)
        print(f"   -> {len(out.layers)} people, seconds {out.seconds}")
        for a, b, sp in out.turns:
            if b - a >= 0.3:
                print(f"   {a:7.2f}-{b:7.2f}  person {sp + 1}")
        for i, layer in enumerate(out.layers):
            save(args.out / f"{clip.stem}_person{i + 1}.wav", layer, rate)
        assert np.max(np.abs(sum(out.layers) - dialogue)) < 1e-4, "people do not sum to the dialogue"


if __name__ == "__main__":
    main()
