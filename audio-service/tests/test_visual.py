"""Visual prompt preparation: Person 2's RLE codec, frame/mask decoding, polarity."""
from __future__ import annotations

import hashlib
import subprocess

import numpy as np
import pytest

from conftest import have_ffmpeg

from sam_audio_service.backends.base import VisualInput
from sam_audio_service.errors import InvalidMedia
from sam_audio_service.visual import (
    _frame_index_range,
    _resize_mask_nearest,
    decode_rle_mask,
    prepare_masked_video,
    probe_video,
    verify_sha256,
)

FPS = 10.0
SIZE = 64  # synthetic videos are 64x64; prepare_masked_video resizes to `size`


# ------------------------------------------------------- Person 2's reference encoder
def encode_mask(mask):
    """Re-implementation of ``vision/soundcanvas_cv/masks.py::encode_mask``.

    Kept here on purpose: the test must pin our decoder against *their* encoder rather
    than against our own decoder.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or not mask.size or not np.isin(mask, [0, 1]).all():
        raise ValueError("Mask must be a binary HxW array")
    flat = mask.astype(np.uint8).ravel()
    edges = np.flatnonzero(np.diff(flat)) + 1
    counts = np.diff(np.r_[0, edges, flat.size]).tolist()
    if flat.size and flat[0]:
        counts.insert(0, 0)
    return {"encoding": "rle", "order": "C", "size": list(mask.shape),
            "counts": counts, "foreground_value": 1}


# ---------------------------------------------------------------------- RLE decoding
def test_rle_matches_contract_example():
    """The example in vision/docs/CONTRACT.md decodes to rows [0,1,1] and [0,0,1]."""
    value = {"encoding": "rle", "order": "C", "size": [2, 3], "counts": [1, 2, 2, 1], "foreground_value": 1}
    assert decode_rle_mask(value).tolist() == [[0, 1, 1], [0, 0, 1]]


@pytest.mark.parametrize("seed", range(6))
def test_rle_roundtrip_against_their_encoder(seed):
    rng = np.random.default_rng(seed)
    mask = (rng.random((7, 11)) > 0.6).astype(np.uint8)
    decoded = decode_rle_mask(encode_mask(mask))
    assert decoded.dtype == np.uint8
    assert np.array_equal(decoded, mask)


@pytest.mark.parametrize("mask", [np.ones((3, 4), np.uint8), np.zeros((3, 4), np.uint8)])
def test_rle_roundtrip_degenerate(mask):
    """All-foreground needs the leading zero-length run; all-background is a single run."""
    assert np.array_equal(decode_rle_mask(encode_mask(mask)), mask)


@pytest.mark.parametrize("value", [
    {"encoding": "coco", "order": "C", "size": [2, 2], "counts": [4], "foreground_value": 1},
    {"encoding": "rle", "order": "F", "size": [2, 2], "counts": [4], "foreground_value": 1},
    {"encoding": "rle", "order": "C", "size": [2, 2], "counts": [4], "foreground_value": 255},
    {"encoding": "rle", "order": "C", "size": [2, 2], "counts": [3], "foreground_value": 1},
    {"encoding": "rle", "order": "C", "size": [2, 2], "counts": [2, -1, 3], "foreground_value": 1},
    {"encoding": "rle", "order": "C", "size": [0, 2], "counts": [1], "foreground_value": 1},
    {"encoding": "rle", "order": "C", "size": [2], "counts": [4], "foreground_value": 1},
    {"encoding": "rle", "order": "C", "size": [2, 2], "counts": [], "foreground_value": 1},
    "not a dict",
])
def test_rle_rejects_bad_input(value):
    with pytest.raises(ValueError):
        decode_rle_mask(value)


# -------------------------------------------------------------------------- helpers
def test_frame_index_range_is_half_open():
    assert _frame_index_range(0.0, 1.0, 10.0) == (0, 10)
    assert _frame_index_range(0.3, 0.7, 10.0) == (3, 7)
    assert _frame_index_range(0.25, 0.55, 10.0) == (3, 6)


def test_resize_mask_nearest_keeps_binary_and_layout():
    mask = np.zeros((4, 4), np.uint8)
    mask[:2] = 1
    out = _resize_mask_nearest(mask, 8)
    assert out.shape == (8, 8)
    assert set(np.unique(out)) <= {0, 1}
    assert out[:4].all() and not out[4:].any()
    assert _resize_mask_nearest(mask, 4) is mask


def test_verify_sha256(tmp_path):
    path = tmp_path / "bytes.bin"
    path.write_bytes(b"soundcanvas")
    digest = hashlib.sha256(b"soundcanvas").hexdigest()
    verify_sha256(str(path), digest)
    verify_sha256(str(path), f"sha256:{digest}")  # Person 2 prefixes digests
    with pytest.raises(InvalidMedia):
        verify_sha256(str(path), "0" * 64)


# ------------------------------------------------- frame + mask tensor preparation
# The RLE codec above is pure numpy and always runs. Tensor preparation needs torch (for
# the output tensors) and ffmpeg (to decode frames), so those tests skip cleanly without.
try:
    import torch
except ImportError:  # pragma: no cover - torch is present wherever the model runs
    torch = None

requires_tensors = pytest.mark.skipif(
    torch is None or not have_ffmpeg(), reason="visual tensor preparation needs torch and ffmpeg"
)


def _video(path, *, seconds=1.0, rate=FPS, size=SIZE):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"testsrc=size={size}x{size}:rate={rate:g}:duration={seconds:g}",
         "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True,
    )
    return str(path)


def _mask_video(path, *, colour="white", seconds=1.0, rate=FPS, size=SIZE):
    """Lossless grayscale mask video (ffv1), the format the generic contract asks for."""
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", f"color=c={colour}:s={size}x{size}:r={rate:g}:d={seconds:g}",
         "-c:v", "ffv1", "-pix_fmt", "gray", str(path)],
        check=True, capture_output=True,
    )
    return str(path)


def _half_mask():
    """Top half = object, so polarity is observable."""
    mask = np.zeros((SIZE, SIZE), np.uint8)
    mask[: SIZE // 2] = 1
    return mask


@pytest.fixture
def video(tmp_path):
    return _video(tmp_path / "clip.mp4")


@requires_tensors
def test_probe_video_reports_native_geometry(video):
    info = probe_video(video)
    assert info["frame_rate"] == pytest.approx(FPS)
    assert (info["width"], info["height"]) == (SIZE, SIZE)


@requires_tensors
def test_frame_masks_full_interval(video):
    masks = {i: _half_mask() for i in range(10)}
    visual = VisualInput(video_path=video, interval=(0.0, 1.0), frame_masks=masks, frame_rate=FPS)
    frames, prepared = prepare_masked_video(visual, size=336)
    assert frames.shape == (10, 3, 336, 336) and frames.dtype == torch.uint8
    assert prepared.shape == (10, 1, 336, 336) and prepared.dtype == torch.uint8
    # keep_object: the processor keeps pixels where the mask is 0, so the object half is 0.
    assert not prepared[:, 0, :168].any()
    assert prepared[:, 0, 168:].all()


@requires_tensors
def test_frame_masks_cropped_interval_uses_native_indices(video):
    """Only frames 3..6 are in [0.3, 0.7); masks must be looked up by native index."""
    masks = {3: _half_mask(), 5: _half_mask()}
    visual = VisualInput(video_path=video, interval=(0.3, 0.7), frame_masks=masks, frame_rate=FPS)
    frames, prepared = prepare_masked_video(visual, size=64)
    assert frames.shape[0] == 4
    object_present = [bool((prepared[i, 0] == 0).any()) for i in range(4)]
    assert object_present == [True, False, True, False]  # frames 3, 4, 5, 6
    # A frame with no mask entry means "object absent": nothing is kept, nothing reused.
    assert prepared[1, 0].all() and prepared[3, 0].all()


@requires_tensors
def test_notebook_mode_inverts_polarity(video):
    masks = {i: _half_mask() for i in range(10)}
    visual = VisualInput(video_path=video, interval=(0.0, 1.0), frame_masks=masks, frame_rate=FPS)
    _, keep = prepare_masked_video(visual, size=64, mask_mode="keep_object")
    _, notebook = prepare_masked_video(visual, size=64, mask_mode="notebook")
    assert torch.equal(notebook, 1 - keep)
    assert notebook[:, 0, :32].all() and not notebook[:, 0, 32:].any()


@requires_tensors
def test_rle_masks_are_accepted_directly(video):
    """frame_masks values may be Person 2's RLE objects as well as arrays."""
    masks = {i: encode_mask(_half_mask()) for i in range(10)}
    visual = VisualInput(video_path=video, interval=(0.0, 1.0), frame_masks=masks, frame_rate=FPS)
    _, prepared = prepare_masked_video(visual, size=64)
    assert not prepared[:, 0, :32].any() and prepared[:, 0, 32:].all()


@requires_tensors
def test_mask_video_path_variant(tmp_path, video):
    mask_video = _mask_video(tmp_path / "mask.mkv")  # fully white => object everywhere
    visual = VisualInput(video_path=video, interval=(0.0, 1.0), mask_video_path=mask_video, frame_rate=FPS)
    frames, prepared = prepare_masked_video(visual, size=64)
    assert frames.shape == (10, 3, 64, 64)
    assert not prepared.any()  # keep_object: everything is kept


@requires_tensors
def test_mask_video_all_black_is_rejected(tmp_path, video):
    mask_video = _mask_video(tmp_path / "black.mkv", colour="black")
    visual = VisualInput(video_path=video, interval=(0.0, 1.0), mask_video_path=mask_video, frame_rate=FPS)
    with pytest.raises(InvalidMedia, match="empty for every frame"):
        prepare_masked_video(visual, size=64)


@requires_tensors
def test_mask_video_size_mismatch_is_rejected(tmp_path, video):
    mask_video = _mask_video(tmp_path / "small.mkv", size=32)
    visual = VisualInput(video_path=video, interval=(0.0, 1.0), mask_video_path=mask_video, frame_rate=FPS)
    with pytest.raises(InvalidMedia, match="frame-for-frame"):
        prepare_masked_video(visual, size=64)


@requires_tensors
def test_mask_video_frame_count_mismatch_is_rejected(tmp_path, video):
    mask_video = _mask_video(tmp_path / "short.mkv", seconds=0.5)
    visual = VisualInput(video_path=video, interval=(0.0, 1.0), mask_video_path=mask_video, frame_rate=FPS)
    with pytest.raises(InvalidMedia, match="aligned frame-for-frame"):
        prepare_masked_video(visual, size=64)


@requires_tensors
def test_declared_frame_size_must_match_the_video(video):
    visual = VisualInput(
        video_path=video, interval=(0.0, 1.0), frame_masks={0: _half_mask()},
        frame_rate=FPS, frame_size=(128, 96),
    )
    with pytest.raises(InvalidMedia, match="masks were produced at"):
        prepare_masked_video(visual, size=64)


@requires_tensors
def test_mask_resolution_must_be_native(video):
    visual = VisualInput(
        video_path=video, interval=(0.0, 1.0), frame_rate=FPS,
        frame_masks={0: np.ones((16, 16), np.uint8)},
    )
    with pytest.raises(InvalidMedia, match="must be at native resolution"):
        prepare_masked_video(visual, size=64)


@requires_tensors
def test_interval_past_the_end_has_no_frames(video):
    visual = VisualInput(video_path=video, interval=(5.0, 6.0), frame_masks={50: _half_mask()}, frame_rate=FPS)
    with pytest.raises(InvalidMedia, match="no video frames in interval"):
        prepare_masked_video(visual, size=64)


@requires_tensors
def test_empty_interval_is_rejected(video):
    visual = VisualInput(video_path=video, interval=(1.0, 1.0), frame_masks={0: _half_mask()}, frame_rate=FPS)
    with pytest.raises(InvalidMedia, match="is empty"):
        prepare_masked_video(visual, size=64)


@requires_tensors
def test_exactly_one_mask_source_is_required(tmp_path, video):
    mask_video = _mask_video(tmp_path / "mask.mkv")
    both = VisualInput(video_path=video, interval=(0.0, 1.0), mask_video_path=mask_video,
                       frame_masks={0: _half_mask()}, frame_rate=FPS)
    neither = VisualInput(video_path=video, interval=(0.0, 1.0), frame_rate=FPS)
    for visual in (both, neither):
        with pytest.raises(InvalidMedia, match="exactly one of"):
            prepare_masked_video(visual, size=64)


@requires_tensors
def test_source_hash_is_verified(video):
    visual = VisualInput(video_path=video, interval=(0.0, 1.0), frame_masks={0: _half_mask()},
                         frame_rate=FPS, source_sha256="0" * 64)
    with pytest.raises(InvalidMedia, match="do not match the analysed source"):
        prepare_masked_video(visual, size=64)


@requires_tensors
def test_frame_rate_is_probed_when_not_declared(video):
    masks = {i: _half_mask() for i in range(10)}
    visual = VisualInput(video_path=video, interval=(0.0, 1.0), frame_masks=masks)
    frames, _ = prepare_masked_video(visual, size=64)
    assert frames.shape[0] == 10


@requires_tensors
def test_unknown_mask_mode_is_rejected(video):
    visual = VisualInput(video_path=video, interval=(0.0, 1.0), frame_masks={0: _half_mask()}, frame_rate=FPS)
    with pytest.raises(ValueError, match="mask_mode"):
        prepare_masked_video(visual, size=64, mask_mode="invert")  # type: ignore[arg-type]
