from .streamer_factory import create_streaming_service

# Default streaming service instance (in-memory)
STREAM_SERVICE = create_streaming_service(backend="memory")

__all__ = ["create_streaming_service", "STREAM_SERVICE"]
