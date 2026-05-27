class RagAppError(Exception):
    """Base class for expected service errors."""


class ServiceUnavailableError(RagAppError):
    """A required external service is unavailable or misconfigured."""


class NoIndexedDocumentsError(RagAppError):
    """No document chunks are available for retrieval."""
