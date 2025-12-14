from .base import BaseStreamingService
from .memory import InMemoryStreamService

# Default instance
STREAM_SERVICE = InMemoryStreamService()

__all__ = ["BaseStreamingService", "InMemoryStreamService", "STREAM_SERVICE"]
