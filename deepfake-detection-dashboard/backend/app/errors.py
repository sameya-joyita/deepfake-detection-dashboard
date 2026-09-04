"""Application-specific errors returned through the JSON API."""


class ApiError(Exception):
    def __init__(self, message, status_code=400, code="bad_request"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class ArtifactError(RuntimeError):
    """Raised when a required trusted model artifact is missing or invalid."""


class VideoReadError(ValueError):
    """Raised when OpenCV cannot decode the supplied video."""
