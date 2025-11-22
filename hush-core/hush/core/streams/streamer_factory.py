from .base_streamer import BaseStreamingService
from .inmem_streamer import InMemoryStreamService


def create_streaming_service(backend: str = "memory", **kwargs) -> BaseStreamingService:
    """Factory function to create streaming service instances.

    Args:
        backend: Streaming backend to use ("memory", "redis", "kafka", etc.)
        **kwargs: Backend-specific arguments

    Returns:
        BaseStreamingService instance

    Examples:
        >>> # Create in-memory service
        >>> service = create_streaming_service(backend="memory")

        >>> # Create Redis service (requires hush-storage)
        >>> service = create_streaming_service(backend="redis", registry_key="myapp")

        >>> # Create Kafka service (requires hush-streaming-kafka)
        >>> service = create_streaming_service(
        ...     backend="kafka",
        ...     bootstrap_servers=["localhost:9092"]
        ... )
    """
    if backend == "memory":
        return InMemoryStreamService()

    elif backend == "redis":
        # Redis support requires hush-storage package
        try:
            from hush.storage.streams.redis_streamer import RedisStreamingService
            registry_key = kwargs.get("registry_key")
            if not registry_key:
                raise ValueError("registry_key required for Redis backend")
            return RedisStreamingService(registry_key)
        except ImportError:
            raise ImportError(
                "Redis streaming requires hush-storage package. "
                "Install with: pip install hush-storage"
            )

    elif backend == "kafka":
        # Kafka support requires hush-streaming-kafka package
        try:
            from hush.streaming.kafka import KafkaStreamingService
            return KafkaStreamingService(**kwargs)
        except ImportError:
            raise ImportError(
                "Kafka streaming requires hush-streaming-kafka package. "
                "Install with: pip install hush-streaming-kafka"
            )

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Supported: 'memory', 'redis', 'kafka'"
        )
