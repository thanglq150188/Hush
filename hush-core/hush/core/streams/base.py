from abc import ABC, abstractmethod
from typing import AsyncGenerator, Any, Optional


class BaseStreamingService(ABC):
    """Abstract base class for streaming services.

    This provides a common interface for different streaming backends
    including in-memory, Redis, Kafka, RabbitMQ, NATS, Pulsar, etc.

    Supports streaming any data type: dicts, chunks, events, or custom objects.
    """

    @abstractmethod
    async def push(
        self,
        request_id: str,
        channel_name: str,
        data: Any,
        session_id: Optional[str] = None
    ) -> None:
        """Push data to the specified channel.

        Args:
            request_id: Request identifier
            channel_name: Channel identifier
            data: Data to push (can be dict, object, or any serializable type)
            session_id: Optional session identifier for multi-tenant scenarios
        """
        pass

    @abstractmethod
    async def end(
        self,
        request_id: str,
        channel_name: str,
        session_id: Optional[str] = None
    ) -> None:
        """Signal end of stream for the specified channel.

        Pushes a sentinel value to indicate no more data will be sent.
        Consumers will stop iteration when they receive this signal.

        Args:
            request_id: Request identifier
            channel_name: Channel identifier
            session_id: Optional session identifier
        """
        pass

    @abstractmethod
    async def get(
        self,
        request_id: str,
        channel_name: str,
        session_id: Optional[str] = None,
        timeout: float = 0.01,
        max_idle_time: Optional[float] = None
    ) -> AsyncGenerator[Any, None]:
        """Get AsyncGenerator for consuming data from the specified channel.

        Args:
            request_id: Request identifier
            channel_name: Channel identifier
            session_id: Optional session identifier
            timeout: Timeout for each queue.get() operation in seconds
            max_idle_time: Maximum time to wait for new data before stopping.
                          If None, waits indefinitely until END signal.
                          Useful for timeout-based termination.

        Yields:
            Data items from the stream until END signal or timeout

        Example:
            # Wait for END signal (default)
            async for data in stream.get(req_id, channel):
                process(data)

            # Stop after 5 seconds of no data
            async for data in stream.get(req_id, channel, max_idle_time=5.0):
                process(data)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close/cleanup resources."""
        pass
