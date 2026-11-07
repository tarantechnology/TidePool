#!/usr/bin/env python3
"""Generate the Person 3 (editor) stem fixture.

Person 3's editor is not in this repository yet, so cross-team Test 1 (editor -> backend)
cannot be executed end to end. This script produces the next best thing: a directory of
plain WAV files plus a ``stems.json`` in Contract C shape (``docs/contracts/audio_contracts.md``)
that a Vite/Next dev server can serve as-is, so the timeline, mute/solo/volume and playback
sync can be built and reviewed before the GPU service exists.

It generates a ~3 s 48 kHz stereo mixture with FFmpeg's ``lavfi`` source (two amplitude
modulated "speech-like" tones, one per channel, plus a siren sweep present in both), then
runs it through this service's own :class:`~sam_audio_service.engine.Engine`. No new
dependencies and no HTTP server are involved.

The default backend is the MOCK backend: its stems are band-pass filtered copies of the
mixture, **not** source separation. What the fixture is faithful about is the *contract* --
stem count, naming, roles, timing, sample rate, channel count and the fact that the stems
sum back to the mixture.

Usage::

    cd audio-service
    .venv/bin/python scripts/make_editor_fixture.py                 # -> fixtures/generated/editor/
    .venv/bin/python scripts/make_editor_fixture.py --out /tmp/ed   # anywhere else
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

DEFAULT_OUT = SERVICE_ROOT / "fixtures" / "generated" / "editor"
DEFAULT_PROMPTS = ("speech", "siren")
SOURCE_NAME = "source.wav"
MANIFEST_NAME = "stems.json"

# Left: 220 Hz syllable-like tone + siren sweep. Right: 330 Hz tone + the same (quieter)
# sweep, so the mixture is genuinely stereo and `stereo_mode` behaviour is visible.
SIREN = "sin(2*PI*(900+300*sin(2*PI*1.5*t))*t)"
LEFT = f"0.22*sin(2*PI*220*t)*(0.55+0.45*sin(2*PI*3*t))+0.22*{SIREN}"
RIGHT = f"0.22*sin(2*PI*330*t)*(0.55+0.45*sin(2*PI*2.5*t))+0.16*{SIREN}"


def build_source(path: Path, *, seconds: float, ffmpeg: str) -> Path:
    """Render the mixture with FFmpeg as 48 kHz stereo PCM16 -- Person 4's upload format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg, "-v", "error", "-y",
            "-f", "lavfi", "-i", f"aevalsrc={LEFT}|{RIGHT}:s=48000:d={seconds:g}",
            "-ac", "2", "-ar", "48000", "-c:a", "pcm_s16le", str(path),
        ],
        check=True,
    )
    return path


def slugify(name: str, used: set[str]) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_") or "stem"
    candidate, n = base, 2
    while candidate in used:
        candidate, n = f"{base}_{n}", n + 1
    used.add(candidate)
    return candidate


def generate(out: Path, *, seconds: float, prompts: list[str], ffmpeg: str) -> dict:
    import numpy as np

    from sam_audio_service.audio_io import decode, encode_wav_pcm16
    from sam_audio_service.config import load_settings
    from sam_audio_service.contracts import (
        ModelInfo,
        SeparationMetadata,
        SeparationResponse,
        SourceInfo,
        StemResult,
    )
    from sam_audio_service.engine import Engine

    out.mkdir(parents=True, exist_ok=True)
    source_path = build_source(out / SOURCE_NAME, seconds=seconds, ffmpeg=ffmpeg)

    settings = load_settings()
    audio = decode(
        path=source_path,
        ffmpeg=settings.ffmpeg,
        ffprobe=settings.ffprobe,
        timeout=settings.ffmpeg_timeout,
    )
    request_id = uuid.uuid4().hex
    asset_id = "editor_fixture_001"
    metadata = SeparationMetadata.model_validate(
        {"prompts": list(prompts), "asset_id": asset_id, "request_id": request_id}
    )
    result = Engine(settings).separate(audio, metadata)

    stems: list[StemResult] = []
    files: list[str] = []
    used: set[str] = set()
    for stem in result.stems:
        wav = encode_wav_pcm16(stem.samples, audio.sample_rate)
        filename = slugify(stem.name, used) + ".wav"
        (out / filename).write_bytes(wav)
        files.append(filename)
        stems.append(
            StemResult(
                name=stem.name,
                stem_id=stem.stem_id,
                role=stem.role,
                prompt=stem.prompt,
                label=stem.label,
                start_time=0.0,
                duration=round(stem.samples.shape[1] / audio.sample_rate, 6),
                sample_rate=audio.sample_rate,
                channels=int(stem.samples.shape[0]),
                sample_count=int(stem.samples.shape[1]),
                spans=stem.spans,
                source_asset_id=asset_id,
            )
        )

    response = SeparationResponse(
        stems=stems,
        model=ModelInfo(**result.model),
        source=SourceInfo(
            asset_id=asset_id,
            duration=round(audio.duration, 6),
            sample_rate=audio.sample_rate,
            channels=audio.channels,
            sample_count=audio.frames,
        ),
        request_id=request_id,
        warnings=list(result.warnings),
    )

    # Contract C, with ``audio_base64`` replaced by a relative ``audio_url`` the browser can
    # fetch straight from the dev server. Everything else is exactly what /separate returns.
    payload = response.model_dump(exclude_none=True)
    for item, filename in zip(payload["stems"], files, strict=True):
        item.pop("audio_base64", None)
        item["audio_url"] = filename

    reconstruction = float(
        np.max(np.abs(sum(s.samples for s in result.stems) - audio.samples))
    )
    payload["fixture"] = {
        "generated_by": "audio-service/scripts/make_editor_fixture.py",
        "source_audio_url": SOURCE_NAME,
        "audio_url_base": "relative to this file",
        "is_real_separation": not result.model["is_mock"],
        "reconstruction_error_float": round(reconstruction, 9),
        "note": (
            "audio_url replaces audio_base64 for local development only; the HTTP service "
            "returns base64 (or a file:// audio_ref). Stem audio here comes from the "
            "MOCK backend unless is_real_separation is true."
        ),
    }
    (out / MANIFEST_NAME).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    peaks = {
        s.name: round(float(np.max(np.abs(s.samples))), 4) if s.samples.size else 0.0
        for s in result.stems
    }
    return {"payload": payload, "files": files, "reconstruction": reconstruction, "peaks": peaks}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--seconds", type=float, default=3.0, help="mixture duration in seconds (default: 3.0)")
    parser.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS), help="prompts to separate (default: speech siren)")
    parser.add_argument("--ffmpeg", default=os.getenv("AUDIO_FFMPEG", "ffmpeg"), help="ffmpeg executable")
    args = parser.parse_args(argv)

    # The fixture documents the contract, not model quality: default to the mock backend so
    # it can be regenerated on any laptop without a GPU or gated checkpoints.
    os.environ.setdefault("AUDIO_BACKEND", "mock")
    os.environ.setdefault("AUDIO_CHUNK_SECONDS", "0")

    info = generate(args.out.resolve(), seconds=args.seconds, prompts=list(args.prompts), ffmpeg=args.ffmpeg)
    payload = info["payload"]
    print(f"wrote {args.out.resolve()}")
    print(f"  {SOURCE_NAME}  {payload['source']['duration']}s {payload['source']['sample_rate']} Hz x{payload['source']['channels']}")
    for item in payload["stems"]:
        print(f"  {item['audio_url']:<24} name={item['name']!r} role={item['role']} peak={info['peaks'].get(item['name'], 0.0)}")
    print(f"  {MANIFEST_NAME} (Contract C, {len(payload['stems'])} stems)")
    print(f"  stems sum back to the source within {info['reconstruction']:.2e} (float, before PCM16 rounding)")
    if payload["model"]["is_mock"]:
        print("  MOCK backend: band-pass filtered copies of the mixture, not source separation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
