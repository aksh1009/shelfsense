"""Errors that map to consistent JSON responses: {"status": "error", "error": {"code", "message"}}."""
from __future__ import annotations


class ShelfAIError(Exception):
    """Base error with an HTTP status and a stable machine-readable code."""

    status = 500
    code = "INTERNAL_ERROR"

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self, request_id: str | None = None) -> dict:
        body = {"status": "error", "error": {"code": self.code, "message": self.message}}
        if self.details:
            body["error"]["details"] = self.details
        if request_id:
            body["request_id"] = request_id
        return body


class InvalidRequest(ShelfAIError):
    status, code = 400, "INVALID_REQUEST"


class InvalidImage(ShelfAIError):
    status, code = 400, "INVALID_IMAGE"


class ImageTooLarge(ShelfAIError):
    status, code = 413, "IMAGE_TOO_LARGE"


class UnsupportedMediaType(ShelfAIError):
    status, code = 415, "UNSUPPORTED_MEDIA_TYPE"


class ImageFetchFailed(ShelfAIError):
    status, code = 422, "IMAGE_FETCH_FAILED"


class ServiceUnavailable(ShelfAIError):
    status, code = 503, "SERVICE_UNAVAILABLE"


class UpstreamTimeout(ShelfAIError):
    status, code = 504, "UPSTREAM_TIMEOUT"


class NotFound(ShelfAIError):
    status, code = 404, "NOT_FOUND"
