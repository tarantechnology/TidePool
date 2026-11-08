"""Visual prompt preparation: video frames + object masks -> model-ready tensors.

What SAM Audio actually consumes
--------------------------------
``SAMAudioProcessor.mask_videos(videos, masks)`` computes ``video * mask.eq(0)``: it
**keeps** pixels where the mask is 0 and zeroes pixels where the mask is non-zero. The
resulting masked video goes into ``processor(..., masked_videos=[...])``; for tensor input
``processor`` maps the N frames *uniformly* onto the audio feature frames
(``torch.linspace(0, N - 1, feature_frames)``), so the frames handed over must cover
exactly the audio interval the backend was given, at the source frame rate. Anything else
silently time-warps the visual conditioning.

The vision encoder resizes every frame to ``image_size`` (336) without preserving aspect
ratio, so pre-resizing frames here is equivalent to letting the encoder do it and bounds
memory at ``N * 3 * size * size`` bytes for the frames plus ``N * size * size`` for masks.

Polarity (``mask_mode``)
------------------------
``prepare_masked_video`` returns masks **in processor polarity**, i.e. ready to hand to
``mask_videos`` unchanged:

* ``"keep_object"`` (default) returns ``1 - object_mask``, so ``mask_videos`` keeps the
  object and zeroes the background. This matches the paper's ``V ⊙ M`` and is the only
  polarity under which the ImageBind re-ranker (which scores audio against the *masked*
  video) is meaningful. Person 2's ``build_processor_batch`` inverts foreground masks the
  same way.
* ``"notebook"`` returns the object mask as-is, reproducing the official visual notebook,
  which passes object masks straight in (object zeroed, background kept). Upstream issue
  #75 on this discrepancy is unresolved, hence the switch.

Selected with ``AUDIO_VISUAL_MASK_MODE``. See ``backends/sam_audio.py`` for the A/B plan.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import subprocess
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .errors import InferenceFailure, InvalidMedia

if TYPE_CHECKING:  # pragma: no cover - torch is only needed for tensor building
    import torch

log = logging.getLogger("sam_audio_service")

MaskMode = Literal["keep_object", "notebook"]
MASK_MODES: tuple[str, ...] = ("keep_object", "notebook")

# Pixel values above this count as object in a mask *video* (generic contract with Person 2).
MASK_VIDEO_THRESHOLD = 127
_EPS = 1e-6


# --------------------------------------------------------------------------- RLE codec
def decode_rle_mask(value: dict[str, Any]) -> np.ndarray:
    """Decode Person 2's portable binary RLE into a ``uint8`` ``HxW`` mask (1 = object).

    Format (``vision/soundcanvas_cv/masks.py``)::

        {"encoding": "rle", "order": "C", "size": [h, w], "counts": [...],
         "foreground_value": 1}

    ``counts`` are row-major alternating run lengths starting with a *zero* run (a
    leading ``0`` when the first pixel is foreground) and must sum to ``h * w``.
    Pure numpy, no torch: safe to import from anywhere.

    Raises:
        ValueError: on any unsupported or inconsistent payload.
    """
    if not isinstance(value, dict):
        raise ValueError("mask must be an RLE object")
    if value.get("encoding") != "rle" or value.get("order") != "C" or value.get("foreground_value") != 1:
        raise ValueError("unsupported mask encoding (expected encoding=rle, order=C, foreground_value=1)")
    size = value.get("size")
    counts = value.get("counts")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("mask size must be [height, width]")
    h, w = size
    if not isinstance(h, int) or not isinstance(w, int) or isinstance(h, bool) or isinstance(w, bool) or h <= 0 or w <= 0:
        raise ValueError("mask size must be two positive integers")
    if not isinstance(counts, (list, tuple)) or not counts:
        raise ValueError("mask counts must be a non-empty list")
    if any(not isinstance(c, int) or isinstance(c, bool) or c < 0 for c in counts):
        raise ValueError("mask counts must be non-negative integers")
    if sum(counts) != h * w:
        raise ValueError(f"mask counts sum to {sum(counts)}, expected {h * w}")
    flat = np.repeat(np.arange(len(counts), dtype=np.uint8) % 2, counts)
    return flat.reshape(h, w)


def _as_mask_array(value: Any) -> np.ndarray:
    """Accept an ``HxW`` array or an RLE object and return ``uint8`` 0/1."""
    if isinstance(value, dict):
        return decode_rle_mask(value)
    arr = np.asarray(value)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"mask must be a non-empty HxW array, got shape {arr.shape}")
    return (arr != 0).astype(np.uint8)


# ------------------------------------------------------------------------- ffmpeg glue
def _capture(args: list[str], timeout: int, what: str) -> bytes:
    """Run ffmpeg/ffprobe and return stdout, mapping failures to service errors."""
    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise InferenceFailure(f"{args[0]} is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise InvalidMedia(f"{what} timed out after {timeout}s") from exc
    if proc.returncode != 0:
        raise InvalidMedia(f"{what} failed: " + proc.stderr.decode("utf-8", "replace")[-400:].strip())
    return proc.stdout


def probe_video(path: str, ffprobe: str = "ffprobe", timeout: int = 60) -> dict[str, Any]:
    """Native frame rate / dimensions / frame count of the first video stream."""
    out = _capture(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", str(path)],
        timeout,
        f"probing {path}",
    )
    data = json.loads(out or b"{}")
    streams = data.get("streams") or []
    if not streams:
        raise InvalidMedia(f"{path} has no video stream")
    stream = streams[0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/0"
    num, _, den = str(rate).partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (TypeError, ValueError, ZeroDivisionError):
        fps = 0.0
    nb = stream.get("nb_frames")
    return {
        "frame_rate": fps if fps > 0 else None,
        "width": int(stream["width"]) if stream.get("width") else None,
        "height": int(stream["height"]) if stream.get("height") else None,
        "frame_count": int(nb) if str(nb).isdigit() else None,
        "duration": float(data.get("format", {}).get("duration") or stream.get("duration") or 0.0),
    }


def verify_sha256(path: str, expected: str, *, chunk: int = 1 << 20) -> None:
    """Confirm the bytes we are about to decode are the ones Person 2 analysed."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            while block := handle.read(chunk):
                digest.update(block)
    except OSError as exc:
        raise InvalidMedia(f"could not read {path}: {exc}") from exc
    actual = digest.hexdigest()
    wanted = expected.split(":", 1)[-1].strip().lower()
    if actual != wanted:
        raise InvalidMedia(
            "video bytes do not match the analysed source; masks would be meaningless",
            details={"expected_sha256": wanted, "actual_sha256": actual, "path": str(path)},
        )


def _decode_rawvideo(
    path: str,
    start: float,
    duration: float,
    size: int,
    *,
    gray: bool,
    ffmpeg: str,
    timeout: int,
) -> np.ndarray:
    """Decode ``[start, start + duration)`` to ``(N, size, size, C)`` uint8.

    Frames are scaled to ``size x size`` without preserving aspect ratio, matching the
    vision encoder. Masks use nearest-neighbour so they stay binary.
    """
    channels = 1 if gray else 3
    scale = f"scale={size}:{size}" + (":flags=neighbor" if gray else "")
    args = [
        ffmpeg, "-v", "error", "-nostdin",
        "-ss", f"{start:.6f}", "-i", str(path), "-t", f"{max(duration, 0.0):.6f}",
        "-an", "-sn", "-dn", "-vf", scale,
        "-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24", "pipe:1",
    ]
    raw = _capture(args, timeout, f"decoding frames from {path}")
    frame_bytes = size * size * channels
    if not raw:
        return np.zeros((0, size, size, channels), dtype=np.uint8)
    count = len(raw) // frame_bytes
    return np.frombuffer(raw[: count * frame_bytes], dtype=np.uint8).reshape(count, size, size, channels)


def _resize_mask_nearest(mask: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour resize of a binary ``HxW`` mask to ``size x size``."""
    h, w = mask.shape
    if (h, w) == (size, size):
        return mask
    rows = np.minimum(((np.arange(size) + 0.5) * h / size).astype(np.int64), h - 1)
    cols = np.minimum(((np.arange(size) + 0.5) * w / size).astype(np.int64), w - 1)
    return mask[rows][:, cols]


def _frame_index_range(start: float, end: float, fps: float) -> tuple[int, int]:
    """Native frame indices whose timestamps ``k / fps`` fall in ``[start, end)``."""
    first = max(0, math.ceil(start * fps - _EPS))
    last_exclusive = max(first, math.ceil(end * fps - _EPS))
    return first, last_exclusive


# ------------------------------------------------------------------------ main entry
def prepare_masked_video(
    visual: Any,
    *,
    size: int = 336,
    mask_mode: MaskMode = "keep_object",
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    timeout: int = 600,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """Decode a ``VisualInput`` into ``(frames NCHW uint8, masks N1HW uint8)``.

    ``frames`` are RGB at ``size x size``; ``masks`` are already in processor polarity
    (see the module docstring), so the caller does exactly::

        masked = processor.mask_videos([frames], [masks])

    Only frames whose native timestamp falls in ``visual.interval`` are decoded — the
    engine crops the mixture to the same interval, which is what makes the processor's
    uniform frame-to-feature mapping correct.

    Raises:
        InvalidMedia: unreadable media, hash mismatch, no frames in the interval,
            mask/video geometry disagreement, or an interval where the object is never
            visible (an all-zero mask carries no information and, in ``keep_object``
            mode, would hand the model a fully black video).
        InferenceFailure: ffmpeg/ffprobe missing.
    """
    torch = _require_torch()
    if mask_mode not in MASK_MODES:
        raise ValueError(f"mask_mode must be one of {MASK_MODES}, got {mask_mode!r}")
    if size <= 0:
        raise ValueError("size must be positive")

    start, end = float(visual.interval[0]), float(visual.interval[1])
    if end <= start:
        raise InvalidMedia(f"visual interval {visual.interval} is empty")
    if visual.source_sha256:
        verify_sha256(visual.video_path, visual.source_sha256)

    info = probe_video(visual.video_path, ffprobe, min(timeout, 60))
    fps = float(visual.frame_rate or info["frame_rate"] or 0.0)
    if fps <= 0:
        raise InvalidMedia(f"could not determine the frame rate of {visual.video_path}")

    frames_hwc = _decode_rawvideo(visual.video_path, start, end - start, size, gray=False, ffmpeg=ffmpeg, timeout=timeout)
    n = int(frames_hwc.shape[0])
    if n == 0:
        raise InvalidMedia(
            f"no video frames in interval [{start:.3f}, {end:.3f})s of {visual.video_path}",
            details={"interval": [start, end], "video_duration": info["duration"], "frame_rate": fps},
        )

    first_index, last_exclusive = _frame_index_range(start, end, fps)
    if n != last_exclusive - first_index:
        raise InvalidMedia(
            f"decoded {n} frames, expected {last_exclusive - first_index}; "
            "visual frames and masks cannot be aligned (normalize to constant frame rate)",
        )

    masks = _build_masks(visual, n, first_index, size, info, ffmpeg=ffmpeg, ffprobe=ffprobe, timeout=timeout)
    if not masks.any():
        raise InvalidMedia(
            "the object mask is empty for every frame in the interval",
            details={"interval": [start, end], "frames": n, "object_id": visual.object_id, "label": visual.label},
        )

    frames = torch.from_numpy(np.ascontiguousarray(frames_hwc.transpose(0, 3, 1, 2)))
    processor_masks = (1 - masks) if mask_mode == "keep_object" else masks
    mask_tensor = torch.from_numpy(np.ascontiguousarray(processor_masks[:, None, :, :]))
    return frames, mask_tensor


def _build_masks(
    visual: Any,
    n: int,
    first_index: int,
    size: int,
    info: dict[str, Any],
    *,
    ffmpeg: str,
    ffprobe: str,
    timeout: int,
) -> np.ndarray:
    """Object masks (1 = object) as ``(n, size, size)`` uint8, aligned to the frames."""
    has_video_masks = bool(getattr(visual, "mask_video_path", None))
    frame_masks = getattr(visual, "frame_masks", None)
    if has_video_masks == bool(frame_masks):
        raise InvalidMedia("visual input needs exactly one of mask_video_path or frame_masks")

    if has_video_masks:
        return _masks_from_video(visual, n, size, info, ffmpeg=ffmpeg, ffprobe=ffprobe, timeout=timeout)

    expected = _expected_mask_shape(visual, info)
    out = np.zeros((n, size, size), dtype=np.uint8)
    for offset in range(n):
        raw = frame_masks.get(first_index + offset)
        if raw is None:
            continue  # object absent in this frame; never reuse a neighbouring mask
        try:
            mask = _as_mask_array(raw)
        except ValueError as exc:
            raise InvalidMedia(f"frame {first_index + offset}: {exc}") from exc
        if expected is not None and mask.shape != expected:
            raise InvalidMedia(
                f"frame {first_index + offset} mask is {mask.shape[1]}x{mask.shape[0]} but the video is "
                f"{expected[1]}x{expected[0]} (width x height); masks must be at native resolution",
                details={"mask_size": list(mask.shape), "video_size": list(expected)},
            )
        out[offset] = _resize_mask_nearest(mask, size)
    return out


def _expected_mask_shape(visual: Any, info: dict[str, Any]) -> tuple[int, int] | None:
    """Expected ``(height, width)`` of a per-frame mask, cross-checking declared sizes."""
    declared = getattr(visual, "frame_size", None)
    video = (info["height"], info["width"]) if info["width"] and info["height"] else None
    if declared:
        width, height = int(declared[0]), int(declared[1])
        if video is not None and (height, width) != video:
            raise InvalidMedia(
                f"masks were produced at {width}x{height} but {visual.video_path} is "
                f"{video[1]}x{video[0]} (width x height)",
                details={"declared_frame_size": [width, height], "video_size": [video[1], video[0]]},
            )
        return height, width
    return video


def _masks_from_video(
    visual: Any,
    n: int,
    size: int,
    info: dict[str, Any],
    *,
    ffmpeg: str,
    ffprobe: str,
    timeout: int,
) -> np.ndarray:
    """Decode the aligned mask video for the same interval; pixels > 127 are object."""
    mask_info = probe_video(visual.mask_video_path, ffprobe, min(timeout, 60))
    if info["width"] and mask_info["width"] and (mask_info["width"], mask_info["height"]) != (info["width"], info["height"]):
        raise InvalidMedia(
            f"mask video is {mask_info['width']}x{mask_info['height']} but the source video is "
            f"{info['width']}x{info['height']}; they must be frame-for-frame aligned",
            details={"mask_size": [mask_info["width"], mask_info["height"]], "video_size": [info["width"], info["height"]]},
        )
    start, end = float(visual.interval[0]), float(visual.interval[1])
    decoded = _decode_rawvideo(visual.mask_video_path, start, end - start, size, gray=True, ffmpeg=ffmpeg, timeout=timeout)
    if decoded.shape[0] == 0:
        raise InvalidMedia(
            f"no mask frames in interval [{start:.3f}, {end:.3f})s of {visual.mask_video_path}",
            details={"interval": [start, end]},
        )
    if decoded.shape[0] != n:
        raise InvalidMedia(
            f"mask video yielded {decoded.shape[0]} frames for interval [{start:.3f}, {end:.3f})s but the "
            f"source video yielded {n}; the mask video must be aligned frame-for-frame",
            details={"mask_frames": int(decoded.shape[0]), "video_frames": n},
        )
    return (decoded[:, :, :, 0] > MASK_VIDEO_THRESHOLD).astype(np.uint8)


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is installed wherever the model runs
        raise InferenceFailure(
            "visual prompting needs torch; install the service with the 'sam' extra"
        ) from exc
    return torch


__all__ = ["MASK_MODES", "MaskMode", "decode_rle_mask", "prepare_masked_video", "probe_video", "verify_sha256"]
