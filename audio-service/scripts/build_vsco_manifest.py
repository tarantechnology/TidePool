#!/usr/bin/env python3
"""Build ``manifest.json`` for the generate renderer from a (partial) VSCO-2 Community Edition
checkout (CC0, github.com/sgossner/VSCO-2-CE).

Every sample's pitch is *measured* (YIN-style f0) and snapped to the filename's note in the
nearest octave, because VSCO folders use different octave conventions. Velocity layers come from
``v1/v2/v3``, ``_p/_f`` or ``dyn1/dyn2``; round robins become extra zones. Loop points (for
sustains) are the steady part of the amplitude envelope. Gains level the presets to about
-20 dBFS RMS so composite sections (cellos/violas/violins) balance.

    python scripts/build_vsco_manifest.py /workspace/gen/samples/VSCO-2-CE
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sam_audio_service.generation import dsp  # noqa: E402
from sam_audio_service.generation.render import read_wav  # noqa: E402

PRESETS = {
    # name: (folder, sustain, attack_s, release_s)
    "violins": ("Strings/Violin Section/susVib", True, 0.16, 0.45),
    "violins_spic": ("Strings/Violin Section/Spic", False, 0.0, 0.12),
    "violins_pizz": ("Strings/Violin Section/Pizz", False, 0.0, 0.2),
    "violas": ("Strings/Viola Section/susvib", True, 0.16, 0.45),
    "cellos": ("Strings/Cello Section/susvib", True, 0.16, 0.5),
    "cellos_spic": ("Strings/Cello Section/spic", False, 0.0, 0.15),
    "solo_violin": ("Strings/Solo Violin/Arco Vib", True, 0.06, 0.35),
    "contrabass_pizz": ("Strings/Solo Contrabass/Pizz", False, 0.0, 0.25),
    "horn": ("Brass/F Horn/sus", True, 0.08, 0.35),
    "flute": ("Woodwinds/Flute/susvib", True, 0.05, 0.3),
    "piano": ("Keys/Upright Piano", False, 0.0, 0.35),
}
NOTE_RE = re.compile(r"(?<![A-Za-z])([A-G])(#|b)?(-?\d)(?![\d])")
PCS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def parse_note(name: str) -> int | None:
    m = None
    for m in NOTE_RE.finditer(name):
        pass  # last match: prefixes like "LLVln_ArcoVib_A3_f" put the note late
    if not m:
        return None
    pc = PCS[m.group(1)] + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)
    return 12 * (int(m.group(3)) + 1) + pc  # scientific pitch: C4 = 60


def parse_layer(name: str) -> int:
    n = name.lower()
    m = re.search(r"_v(\d)", n) or re.search(r"dyn(\d)", n)
    if m:
        return int(m.group(1))
    if re.search(r"_(pp|p)(_|\.|$)", n):
        return 1
    if re.search(r"_(ff|f)(_|\.|$)", n):
        return 3
    if re.search(r"_(mf|mp)(_|\.|$)", n):
        return 2
    return 1


def loop_points(x: np.ndarray, sr: int) -> list[float]:
    env = dsp.rms_envelope(x, sr, hop_s=0.01, win_s=0.05)
    peak = env.max() + 1e-9
    above = np.where(env > 0.6 * peak)[0]
    if above.size < 30:
        return [0.35, 0.88]
    a, b = above[0] + 10, above[-1] - 5
    n = len(env)
    return [round(float(a / n), 4), round(float(max(a + 20, b) / n), 4)]


def main(root: Path) -> None:
    piano_map = {}
    chart = root / "Keys/Upright Piano/MappingChart.txt"
    if chart.exists():
        for line in chart.read_text().splitlines():
            if "=" in line and line.split("=")[0].isdigit():
                k, v = line.split("=")
                piano_map[k.strip()] = int(v)
    presets = {}
    for name, (folder, sustain, attack, release) in PRESETS.items():
        files = sorted((root / folder).glob("*.wav"))
        if not files:
            print(f"skip {name}: no files in {folder}")
            continue
        zones, offsets, rms = [], [], []
        for f in files:
            x, sr = read_wav(f)
            mono = x.mean(axis=0)
            if name == "piano":
                idx = re.search(r"_(\d{3})\.wav$", f.name)
                nominal = piano_map.get(idx.group(1)) if idx else None
            else:
                nominal = parse_note(f.stem)
            f0 = dsp.estimate_f0(mono[: sr * 3], sr, fmin=27.0, fmax=2400.0)
            measured = float(dsp.hz_to_midi(f0)) if f0 else None
            pitch = float(nominal) if nominal is not None else measured
            if nominal is not None and measured is not None:
                k = round((measured - nominal) / 12)
                cents = measured - (nominal + 12 * k)
                offsets.append(k)
                if abs(cents) < 0.6:
                    pitch = nominal + 12 * k + cents
                else:
                    pitch = float(nominal + 12 * k)
            if pitch is None:
                continue
            zones.append({"file": str(f.relative_to(root)), "pitch": round(pitch, 3), "layer_raw": parse_layer(f.stem), "nominal": nominal,
                          "measured": round(measured, 2) if measured else None, "loop": loop_points(mono, sr) if sustain else None})
            rms.append(float(np.sqrt(np.mean(mono[: sr * 2] ** 2))))
        if offsets:
            # Octave offset: majority vote, then re-apply to every zone (f0 can octave-jump on single samples).
            k_major = Counter(offsets).most_common(1)[0][0]
            for z in zones:
                if z["nominal"] is not None:
                    frac = z["pitch"] - round(z["pitch"])
                    z["pitch"] = round(z["nominal"] + 12 * k_major + (frac if abs(frac) < 0.6 else 0), 3)
        layers = sorted({z["layer_raw"] for z in zones})
        for z in zones:
            z["layer"] = layers.index(z.pop("layer_raw"))
        gain = float(10 ** (-20 / 20) / (np.median(sorted(rms)[len(rms) // 2:]) + 1e-9))
        presets[name] = {"folder": folder, "sustain": sustain, "attack": attack, "release": release, "gain": round(min(gain, 20.0), 4),
                         "layers": len(layers), "range": [min(z["pitch"] for z in zones), max(z["pitch"] for z in zones)], "zones": zones}
        print(f"{name:16s} {len(zones):3d} zones, layers {len(layers)}, range {presets[name]['range']}, octave fix {Counter(offsets).most_common(2)}")
    (root / "manifest.json").write_text(json.dumps({"source": "VSCO-2 Community Edition (CC0-1.0)", "presets": presets}, indent=1))
    print("wrote", root / "manifest.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
