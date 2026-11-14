"""Contract E (proposed): shared processing error format.

Every failure leaves the service as JSON of the form::

    {"detail": "<message>", "error": {"code": "<CODE>", "message": "<message>",
                                       "retryable": false, "details": {...}}}

``detail`` keeps the FastAPI shape Person 4's backend already understands; ``error``
carries the stable machine-readable code Person 4 can translate into user-facing text.
"""
from __future__ import annotations

from typing import Any


class AudioServiceError(Exception):
    """Unexpected failure inside the service (INTERNAL_ERROR)."""

    code = "INTERNAL_ERROR"
    status = 500
    retryable = False

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_payload(self) -> dict[str, Any]:
        return {
            "detail": self.message,
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "details": self.details,
            },
        }


class InvalidRequest(AudioServiceError):
    """Malformed request or metadata (bad JSON, unknown fields, no prompt). Not retryable."""

    code = "INVALID_REQUEST"
    status = 400


class InvalidMedia(AudioServiceError):
    """Audio/video could not be decoded, is empty, or exceeds the byte limit."""

    code = "INVALID_MEDIA"
    status = 422


class UnsupportedOperation(AudioServiceError):
    """Valid request the service cannot honour: too long, unknown model, unsupported prompting mode."""

    code = "UNSUPPORTED_OPERATION"
    status = 400


class MissingAsset(AudioServiceError):
    """A referenced media file or document does not exist."""

    code = "MISSING_ASSET"
    status = 404


class MissingModel(AudioServiceError):
    """Model weights are not loaded or not accessible (gated checkpoint, bad token). Retry later."""

    code = "MISSING_MODEL"
    status = 503
    retryable = True


class InsufficientGpuMemory(AudioServiceError):
    """The GPU ran out of memory loading the model or separating a window. Retry with shorter input or lower settings."""

    code = "INSUFFICIENT_GPU_MEMORY"
    status = 503
    retryable = True


class InferenceFailure(AudioServiceError):
    """The model raised during separation. Not retryable without changes."""

    code = "INFERENCE_FAILURE"
    status = 500


class StorageFailure(AudioServiceError):
    """Could not read referenced media or write stems. Usually transient."""

    code = "STORAGE_FAILURE"
    status = 500
    retryable = True


class Cancelled(AudioServiceError):
    """The request was cancelled while it ran (stem splitting checks between chunks). Retry if still wanted."""

    code = "CANCELLED"
    status = 409
    retryable = True


class ServiceBusy(AudioServiceError):
    """Too many concurrent requests. Retry later."""

    code = "SERVICE_BUSY"
    status = 503
    retryable = True


# --------------------------------------------------------------- OOM classification
# Under memory pressure CUDA does not always say "out of memory": the allocation failure
# surfaces through whichever library asked for the block first. These are the texts seen
# in practice, matched case-insensitively against "<ClassName>: <message>":
#
# * ``CUDA out of memory`` / ``CUDA error: out of memory`` - the caching allocator itself;
# * ``CUBLAS_STATUS_ALLOC_FAILED`` - cuBLAS workspace allocation (a GEMM inside the
#   transformer or the codec), raised as a plain RuntimeError;
# * ``CUDNN_STATUS_ALLOC_FAILED`` - cuDNN workspace allocation in a convolution;
# * ``cudaErrorMemoryAllocation`` - the raw driver code, seen from kernels and from
#   ``cudaMalloc`` inside third-party ops;
# * ``CUDA error: an illegal memory access`` - what an allocation failure degrades into
#   when a kernel was launched against memory it could not get.
#
# All of them mean the same thing for us: this batch did not fit. They are retryable at
# AUDIO_CHUNK_BATCH=1, so they must be classified as INSUFFICIENT_GPU_MEMORY and not as
# INFERENCE_FAILURE, which kills the request. ``torch.OutOfMemoryError`` (and its alias
# ``torch.cuda.OutOfMemoryError``) is matched by class name, including subclasses, because
# it is not importable without torch.
_OOM_HINTS: tuple[str, ...] = (
    "out of memory",
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
    "cudaerrormemoryallocation",
    "cuda error: an illegal memory access",
)


def is_out_of_memory(exc: BaseException) -> bool:
    """True when ``exc`` means "the GPU could not give us the memory", not "we have a bug".

    Shared by the backend (which turns it into :class:`InsufficientGpuMemory` after freeing
    the batch) and the engine (which retries the whole signal at batch 1), so the two can
    never disagree about what counts as an OOM.
    """
    if any(cls.__name__ == "OutOfMemoryError" for cls in type(exc).__mro__):
        return True
    blob = f"{type(exc).__name__}: {exc}".lower()
    return any(hint in blob for hint in _OOM_HINTS)


ERROR_CODES: dict[str, dict[str, Any]] = {
    cls.code: {"http_status": cls.status, "retryable": cls.retryable, "description": (cls.__doc__ or "").strip()}
    for cls in (
        InvalidRequest, InvalidMedia, UnsupportedOperation, MissingAsset, MissingModel,
        InsufficientGpuMemory, InferenceFailure, StorageFailure, Cancelled, ServiceBusy, AudioServiceError,
    )
}
