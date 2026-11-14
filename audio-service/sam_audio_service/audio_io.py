"""Audio decode/encode and normalisation for model input.

Scope: only what inference needs (PCM WAV in and out, resampling, channel handling).
General media processing and final rendering belong to Person 4 and use FFmpeg there.
"""
from __future__ import annotations

import io
import json
import struct
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import InvalidMedia
from .errors import InferenceFailure

MODEL_SAMPLE_RATE = 48_000


@dataclass
class DecodedAudio:
    samples: np.ndarray  # float32, shape (channels, frames), range about [-1, 1]
    sample_rate: int
    source_sample_rate: int
    source_channels: int

    @property
    def channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[1])

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate


def _run(args: list[str], stdin: bytes | None, timeout: int) -> bytes:
    try:
        proc = subprocess.run(args, input=stdin, capture_output=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise InferenceFailure(f"{args[0]} is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise InvalidMedia("media decoding timed out") from exc
    if proc.returncode != 0:
        message = proc.stderr.decode("utf-8", "replace")[-500:].strip()
        for arg in args:  # keep server-side paths (temp spool files) out of client-facing errors
            if "/" in arg and arg in message:
                message = message.replace(arg, "<input>")
        raise InvalidMedia("could not decode audio: " + message)
    return proc.stdout


def probe(path: Path, ffprobe: str = "ffprobe", timeout: int = 60) -> dict:
    out = _run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], None, timeout)
    data = json.loads(out or b"{}")
    streams = data.get("streams", [])
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if audio is None:
        raise InvalidMedia("media has no audio stream")
    fps = None
    if video and video.get("avg_frame_rate") and video["avg_frame_rate"] != "0/0":
        num, _, den = video["avg_frame_rate"].partition("/")
        fps = float(num) / float(den or 1)
    return {
        "duration": float(data.get("format", {}).get("duration") or audio.get("duration") or 0.0),
        "sample_rate": int(audio["sample_rate"]),
        "channels": int(audio["channels"]),
        "has_video": video is not None,
        "frame_rate": fps,
        "frame_count": int(video["nb_frames"]) if video and str(video.get("nb_frames", "")).isdigit() else None,
        "width": int(video["width"]) if video else None,
        "height": int(video["height"]) if video else None,
    }


def _try_wave(data: bytes) -> tuple[np.ndarray, int] | None:
    """Fast path for PCM WAV (8/16/24/32-bit integer) without spawning FFmpeg."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
            raw = wav.readframes(frames)
    except (wave.Error, EOFError, struct.error):
        return None
    if channels < 1 or frames == 0:
        return None
    if width == 2:
        arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        arr = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    elif width == 1:
        arr = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        arr = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) ^ 0x800000) - 0x800000
        arr = arr.astype(np.float32) / 8388608.0
    else:
        return None
    usable = (len(arr) // channels) * channels
    return arr[:usable].reshape(-1, channels).T.copy(), rate


def _ffmpeg_decode(data: bytes | None, path: Path | None, target_rate: int, channels: int | None, ffmpeg: str, timeout: int) -> np.ndarray:
    args = [ffmpeg, "-v", "error", "-nostdin", "-i", str(path) if path else "pipe:0", "-vn", "-ar", str(target_rate)]
    if channels:
        args += ["-ac", str(channels)]
    args += ["-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    out = _run(args, data, timeout)
    if not out:
        raise InvalidMedia("media contains no decodable audio")
    arr = np.frombuffer(out, dtype="<f4")
    if channels is None:
        channels = _probe_channels_from_bytes(data, path, ffmpeg.replace("ffmpeg", "ffprobe"), timeout)
    usable = (len(arr) // channels) * channels
    return arr[:usable].reshape(-1, channels).T.copy()


def _probe_channels_from_bytes(data: bytes | None, path: Path | None, ffprobe: str, timeout: int) -> int:
    args = [ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=channels", "-of", "csv=p=0", str(path) if path else "pipe:0"]
    out = _run(args, data, timeout).decode().strip()
    try:
        return max(1, int(out.splitlines()[0]))
    except (ValueError, IndexError) as exc:
        raise InvalidMedia("could not determine channel count") from exc


def decode(data: bytes | None = None, path: Path | None = None, *, target_rate: int = MODEL_SAMPLE_RATE, max_channels: int = 2, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", timeout: int = 600) -> DecodedAudio:
    """Decode any FFmpeg-readable media to float32 (channels, frames) at ``target_rate``.

    Channel count is preserved up to ``max_channels`` (more are downmixed to that count).
    The model consumes mono; stereo handling is decided later by the engine so the
    residual can be computed against the original channel layout.
    """
    if (data is None) == (path is None):
        raise ValueError("pass exactly one of data or path")
    if path is not None and not Path(path).exists():
        raise InvalidMedia(f"media file not found: {path}")
    fast = _try_wave(data) if data is not None else None
    if fast is None and path is not None:
        try:
            fast = _try_wave(Path(path).read_bytes())
        except OSError:
            fast = None
    if fast is not None and fast[0].shape[0] > max_channels:
        fast = None  # let ffmpeg do a proper matrix downmix (-ac)
    if fast is not None:
        samples, rate = fast
        source_channels = samples.shape[0]
        if rate != target_rate:
            samples = resample(samples, rate, target_rate, ffmpeg=ffmpeg, timeout=timeout)
        return DecodedAudio(np.ascontiguousarray(samples, dtype=np.float32), target_rate, rate, source_channels)
    info = probe(path, ffprobe, min(timeout, 60)) if path is not None else None
    source_channels = info["channels"] if info else _probe_channels_from_bytes(data, None, ffprobe, timeout)
    source_rate = info["sample_rate"] if info else None
    channels = min(source_channels, max_channels)
    samples = _ffmpeg_decode(data, path, target_rate, channels, ffmpeg, timeout)
    if source_rate is None:
        source_rate = target_rate
    return DecodedAudio(np.ascontiguousarray(samples, dtype=np.float32), target_rate, source_rate, source_channels)


def resample(samples: np.ndarray, source_rate: int, target_rate: int, *, ffmpeg: str = "ffmpeg", timeout: int = 600) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    channels = samples.shape[0]
    raw = np.ascontiguousarray(samples.T, dtype="<f4").tobytes()
    args = [ffmpeg, "-v", "error", "-nostdin", "-f", "f32le", "-ar", str(source_rate), "-ac", str(channels), "-i", "pipe:0", "-ar", str(target_rate), "-f", "f32le", "pipe:1"]
    out = _run(args, raw, timeout)
    arr = np.frombuffer(out, dtype="<f4")
    usable = (len(arr) // channels) * channels
    return arr[:usable].reshape(-1, channels).T.copy()


def to_mono(samples: np.ndarray) -> np.ndarray:
    return samples.mean(axis=0) if samples.shape[0] > 1 else samples[0]


def sanitise(samples: np.ndarray) -> np.ndarray:
    """Replace NaN/inf (a model can emit them) with silence; never changes gain."""
    if not np.isfinite(samples).all():
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    return samples


def encode_wav_pcm16(samples: np.ndarray, sample_rate: int) -> bytes:
    if samples.ndim == 1:
        samples = samples[None, :]
    clipped = np.clip(sanitise(samples), -1.0, 1.0)
    pcm = (clipped.T * 32767.0).round().astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(samples.shape[0])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


def encode_wav_float32(samples: np.ndarray, sample_rate: int) -> bytes:
    """IEEE float WAV preserves model values outside [-1, 1] for lossless stem mixing."""
    if samples.ndim == 1:
        samples = samples[None, :]
    channels = samples.shape[0]
    data = np.ascontiguousarray(sanitise(samples).T, dtype='<f4').tobytes()
    # RIFF WAVE, format 3 = IEEE float; do not apply per-stem normalization.
    fmt = struct.pack('<HHIIHH', 3, channels, sample_rate, sample_rate * channels * 4, channels * 4, 32)
    return b'RIFF' + struct.pack('<I', 36 + len(data)) + b'WAVEfmt ' + struct.pack('<I', 16) + fmt + b'data' + struct.pack('<I', len(data)) + data


def peak_dbfs(samples: np.ndarray) -> float:
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return 20 * np.log10(peak) if peak > 0 else -np.inf


def rms_dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    return 20 * np.log10(rms) if rms > 0 else -np.inf
