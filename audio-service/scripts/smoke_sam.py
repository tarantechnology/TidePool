#!/usr/bin/env python
"""Manual smoke test for the real SAM Audio backend. Never runs in CI.

Nothing here can work until checkpoint access is approved on huggingface.co and
``.[sam]`` is installed; it exists so the first GPU run is one command instead of a
notebook. It calls ``SamAudioBackend`` directly (no HTTP, no engine), so what it measures
is model load time, inference time and peak GPU memory for a single window.

Text prompting::

    .venv/bin/python scripts/smoke_sam.py --input mix.wav --prompt "a dog barking"

Span prompting (either predicted or given)::

    .venv/bin/python scripts/smoke_sam.py --input mix.wav --prompt siren --predict-spans
    .venv/bin/python scripts/smoke_sam.py --input mix.wav --prompt siren --anchor 0.5:2.0

Visual prompting, including the polarity A/B described in ``backends/sam_audio.py``::

    .venv/bin/python scripts/smoke_sam.py --input clip.mp4 --video clip.mp4 \
        --mask-video masks.mkv --interval 0:10 --prompt guitar --mask-mode both

``--frame-masks`` takes a JSON object mapping native frame index to Person 2's RLE mask
(``{"12": {"encoding": "rle", ...}}``) as an alternative to a mask video.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sam_audio_service.audio_io import MODEL_SAMPLE_RATE, decode, encode_wav_pcm16, peak_dbfs, to_mono  # noqa: E402
from sam_audio_service.backends.base import BackendOptions, BackendPrompt, VisualInput  # noqa: E402
from sam_audio_service.backends.sam_audio import SamAudioBackend  # noqa: E402
from sam_audio_service.visual import decode_rle_mask  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, help="WAV/MP4/anything FFmpeg reads; synthesised when omitted")
    p.add_argument("--synth-seconds", type=float, default=10.0, help="length of the synthetic mixture")
    p.add_argument("--prompt", default="a dog barking")
    p.add_argument("--anchor", action="append", default=[], metavar="START:END[:TOKEN]",
                   help="span anchor in seconds relative to the window; TOKEN is + (default) or -")
    p.add_argument("--predict-spans", action="store_true", help="let the span predictor find the spans")
    p.add_argument("--candidates", type=int, default=1, help="reranking candidates (8 = paper quality)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--interval", default=None, metavar="START:END",
                   help="seconds of the input to process; required shape for visual prompting")
    p.add_argument("--video", type=Path, help="source video for visual prompting")
    p.add_argument("--mask-video", type=Path, help="lossless mask video aligned frame-for-frame (>127 = object)")
    p.add_argument("--frame-masks", type=Path, help="JSON {frame_index: RLE mask} from Person 2's codec")
    p.add_argument("--frame-rate", type=float, default=None, help="native frame rate, if ffprobe cannot tell")
    p.add_argument("--label", default=None, help="object label; used as the description when --prompt is empty")
    p.add_argument("--mask-mode", choices=["keep_object", "notebook", "both"], default="keep_object")
    p.add_argument("--model-id", default="facebook/sam-audio-large")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--no-span-predictor", action="store_true", help="save memory; disables --predict-spans")
    p.add_argument("--no-rankers", action="store_true", help="save memory; disables reranking quality gains")
    p.add_argument("--frame-size", type=int, default=336)
    p.add_argument("--out", type=Path, default=Path("smoke_out"))
    return p.parse_args(argv)


def synthesise(seconds: float, rate: int = MODEL_SAMPLE_RATE) -> np.ndarray:
    """A three-source mixture: steady tone, swept 'siren', broadband noise bed."""
    t = np.arange(int(seconds * rate)) / rate
    tone = 0.25 * np.sin(2 * np.pi * 220.0 * t)
    siren = 0.3 * np.sin(2 * np.pi * (900 + 400 * np.sin(2 * np.pi * 0.5 * t)) * t)
    bed = 0.05 * np.random.default_rng(0).standard_normal(t.size)
    return (tone + siren + bed).astype(np.float32)


def parse_interval(raw: str | None, duration: float) -> tuple[float, float]:
    if not raw:
        return 0.0, duration
    start, _, end = raw.partition(":")
    return float(start or 0.0), min(float(end) if end else duration, duration)


def parse_anchors(raw: list[str]) -> list[tuple[str, float, float]]:
    anchors = []
    for item in raw:
        parts = item.split(":")
        if len(parts) not in (2, 3):
            raise SystemExit(f"--anchor {item!r} must be START:END or START:END:TOKEN")
        token = parts[2] if len(parts) == 3 else "+"
        if token not in ("+", "-"):
            raise SystemExit("anchor token must be '+' or '-'")
        anchors.append((token, float(parts[0]), float(parts[1])))
    return anchors


def load_frame_masks(path: Path) -> dict[int, np.ndarray]:
    payload = json.loads(path.read_text())
    masks: dict[int, np.ndarray] = {}
    for key, value in payload.items():
        masks[int(key)] = decode_rle_mask(value) if isinstance(value, dict) else np.asarray(value, dtype=np.uint8)
    return masks


def build_visual(args: argparse.Namespace, interval: tuple[float, float]) -> VisualInput | None:
    if args.video is None:
        if args.mask_video or args.frame_masks:
            raise SystemExit("--mask-video/--frame-masks need --video")
        return None
    if bool(args.mask_video) == bool(args.frame_masks):
        raise SystemExit("visual prompting needs exactly one of --mask-video or --frame-masks")
    return VisualInput(
        video_path=str(args.video),
        interval=interval,
        mask_video_path=str(args.mask_video) if args.mask_video else None,
        frame_masks=load_frame_masks(args.frame_masks) if args.frame_masks else None,
        frame_rate=args.frame_rate,
        label=args.label,
    )


def run_once(args: argparse.Namespace, mask_mode: str, window: np.ndarray, visual: VisualInput | None) -> None:
    seconds = len(window) / MODEL_SAMPLE_RATE
    backend = SamAudioBackend(
        model_id=args.model_id,
        device=args.device,
        dtype=args.dtype,
        mask_mode=mask_mode,  # type: ignore[arg-type]
        load_span_predictor=not args.no_span_predictor,
        load_rankers=not args.no_rankers,
        frame_size=args.frame_size,
    )
    started = time.perf_counter()
    backend.load()
    load_seconds = time.perf_counter() - started
    info = backend.info()
    print(f"\n=== mask_mode={mask_mode} ===")
    print(f"model      {info['model_id']} (revision={info['revision']}, {(info['parameters'] or 0) / 1e9:.2f}B params)")
    print(f"device     {info['device']}  dtype={info['dtype']}  torch={info['torch_version']}")
    print(f"flags      span_predictor={info['span_predictor_loaded']} visual_ranker={info['visual_ranker_loaded']} "
          f"text_ranker={info['text_ranker_loaded']}")
    print(f"load       {load_seconds:.1f}s")

    options = BackendOptions(
        predict_spans=args.predict_spans and not args.no_span_predictor,
        reranking_candidates=args.candidates,
        seed=args.seed,
    )
    prompt = BackendPrompt(text=args.prompt, anchors=parse_anchors(args.anchor) or None, visual=visual)
    started = time.perf_counter()
    result = backend.separate(window, prompt, options)
    infer_seconds = time.perf_counter() - started

    print(f"inference  {infer_seconds:.1f}s for {seconds:.1f}s audio  (real-time factor {infer_seconds / seconds:.2f}x)")
    after = backend.info()
    if "cuda_peak_allocated_bytes" in after:
        print(f"gpu peak   {after['cuda_peak_allocated_bytes'] / 2**30:.1f} GiB allocated / "
              f"{after['cuda_peak_reserved_bytes'] / 2**30:.1f} GiB reserved on {after['cuda_device_name']}")
    if result.spans:
        print("spans      " + ", ".join(f"{s:.2f}-{e:.2f}s" for s, e in result.spans))
    levels = f"target {peak_dbfs(result.target):.1f} dBFS"
    if result.residual is not None:
        levels += f", residual {peak_dbfs(result.residual):.1f} dBFS"
    print(f"levels     {levels}")
    print(f"extra      {result.extra}")

    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f"_{mask_mode}" if visual is not None else ""
    for name, data in (("target", result.target), ("residual", result.residual)):
        if data is None:
            continue
        path = args.out / f"{name}{suffix}.wav"
        path.write_bytes(encode_wav_pcm16(data, MODEL_SAMPLE_RATE))
        print(f"wrote      {path}")
    backend.unload()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input:
        audio = decode(path=args.input, max_channels=2)
        mixture, duration = to_mono(audio.samples), audio.duration
        print(f"input      {args.input} ({duration:.1f}s, {audio.source_channels}ch @ {audio.source_sample_rate} Hz)")
    else:
        mixture = synthesise(args.synth_seconds)
        duration = args.synth_seconds
        print(f"input      synthetic mixture ({duration:.1f}s: 220 Hz tone + swept siren + noise bed)")

    start, end = parse_interval(args.interval, duration)
    if end <= start:
        raise SystemExit(f"--interval {args.interval!r} is empty")
    window = mixture[int(start * MODEL_SAMPLE_RATE): int(end * MODEL_SAMPLE_RATE)]
    print(f"window     [{start:.2f}, {end:.2f})s -> {len(window)} samples @ {MODEL_SAMPLE_RATE} Hz")
    print(f"prompt     {args.prompt!r} anchors={args.anchor or None} predict_spans={args.predict_spans} "
          f"candidates={args.candidates}")

    visual = build_visual(args, (start, end))
    if visual is not None:
        print(f"visual     {visual.video_path} masks="
              f"{visual.mask_video_path or f'{len(visual.frame_masks)} frame masks'}")
    modes = ["keep_object", "notebook"] if args.mask_mode == "both" else [args.mask_mode]
    if visual is None and args.mask_mode == "both":
        modes = ["keep_object"]  # mask mode is irrelevant without a visual prompt
    for mode in modes:
        run_once(args, mode, window, visual)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
