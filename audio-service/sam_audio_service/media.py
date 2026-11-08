"""Resolve media references to local files.

Person 4 owns storage. This module only *reads* what Person 4 (or Person 2) points at:
``file://`` paths that must live under ``AUDIO_MEDIA_ROOT`` (a shared volume) or http(s)
URLs on an allow-listed host. It never writes into Person 4's storage layout.
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from .config import Settings
from .errors import InvalidRequest, MissingAsset, StorageFailure


def resolve_ref(ref: str, settings: Settings, *, workdir: Path | None = None) -> Path:
    parsed = urlparse(ref)
    if parsed.scheme in ("", "file"):
        if settings.media_root is None:
            raise InvalidRequest("file references are disabled: set AUDIO_MEDIA_ROOT to a shared media directory")
        raw = unquote(parsed.path if parsed.scheme == "file" else ref)
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = settings.media_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise MissingAsset(f"media not found: {ref}") from exc
        if settings.media_root not in resolved.parents and resolved != settings.media_root:
            raise InvalidRequest("file reference is outside AUDIO_MEDIA_ROOT")
        if not resolved.is_file():
            raise MissingAsset(f"media is not a file: {ref}")
        return resolved
    if parsed.scheme in ("http", "https"):
        host = parsed.hostname or ""
        if host not in settings.allowed_media_hosts:
            raise InvalidRequest(f"host {host!r} is not in AUDIO_ALLOWED_MEDIA_HOSTS")
        target_dir = workdir or Path(tempfile.mkdtemp(prefix="sam-audio-media-"))
        suffix = Path(parsed.path).suffix or ".bin"
        target = target_dir / (hashlib.sha256(ref.encode()).hexdigest()[:24] + suffix)
        try:
            with httpx.stream("GET", ref, timeout=120, follow_redirects=False) as response:
                if response.status_code == 404:
                    raise MissingAsset(f"media not found: {ref}")
                response.raise_for_status()
                with target.open("wb") as out:
                    size = 0
                    for chunk in response.iter_bytes(1024 * 1024):
                        size += len(chunk)
                        if size > settings.max_input_bytes:
                            raise InvalidRequest("referenced media exceeds AUDIO_MAX_INPUT_BYTES")
                        out.write(chunk)
        except httpx.HTTPError as exc:
            raise StorageFailure(f"could not fetch media: {exc}") from exc
        return target
    raise InvalidRequest(f"unsupported media reference scheme: {parsed.scheme!r}")


def to_file_ref(path: Path) -> str:
    return path.resolve().as_uri()
