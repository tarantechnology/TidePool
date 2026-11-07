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
    code = "INVALID_REQUEST"
    status = 400


class InvalidMedia(AudioServiceError):
    code = "INVALID_MEDIA"
    status = 422


class UnsupportedOperation(AudioServiceError):
    code = "UNSUPPORTED_OPERATION"
    status = 400


class MissingAsset(AudioServiceError):
    code = "MISSING_ASSET"
    status = 404


class MissingModel(AudioServiceError):
    code = "MISSING_MODEL"
    status = 503
    retryable = True


class InsufficientGpuMemory(AudioServiceError):
    code = "INSUFFICIENT_GPU_MEMORY"
    status = 503
    retryable = True


class InferenceFailure(AudioServiceError):
    code = "INFERENCE_FAILURE"
    status = 500


class StorageFailure(AudioServiceError):
    code = "STORAGE_FAILURE"
    status = 500
    retryable = True


class ServiceBusy(AudioServiceError):
    code = "SERVICE_BUSY"
    status = 503
    retryable = True


ERROR_CODES: dict[str, dict[str, Any]] = {
    cls.code: {"http_status": cls.status, "retryable": cls.retryable, "description": (cls.__doc__ or "").strip()}
    for cls in (
        InvalidRequest, InvalidMedia, UnsupportedOperation, MissingAsset, MissingModel,
        InsufficientGpuMemory, InferenceFailure, StorageFailure, ServiceBusy, AudioServiceError,
    )
}
