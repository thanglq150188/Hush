import asyncio
import time
from typing import AsyncGenerator, Dict, Set, Optional, Any
from collections import defaultdict
from hush.core.loggings import LOGGER
from .base import BaseStreamingService


class InMemoryStreamService(BaseStreamingService):
    """In-memory streaming service using asyncio.Queue.

    High-performance in-memory implementation for single-process workflows.
    Organizes streams by session_id, request_id, and channel_name.

    For distributed systems, use Redis, Kafka, or other distributed backends.
    """

    DEFAULT_SESSION = "default"

    def __init__(self):
        # Structure: {session_id: {request_id: {channel_name: asyncio.Queue}}}
        self._queues: Dict[str, Dict[str, Dict[str, asyncio.Queue]]] = defaultdict(lambda: defaultdict(dict))
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        LOGGER.info("InMemoryStreamService initialized")

    async def push(
        self,
        request_id: str,
        channel_name: str,
        data: Any,
        session_id: Optional[str] = None
    ) -> None:
        """Push data to the specified channel queue.

        Args:
            request_id: Request identifier
            channel_name: Channel identifier
            data: Data to push (any type)
            session_id: Optional session identifier (defaults to "default")
        """
        if not request_id:
            raise ValueError("request_id cannot be None or empty")

        session_id = session_id or self.DEFAULT_SESSION

        try:
            async with self._lock:
                # Ensure queue exists
                if channel_name not in self._queues[session_id][request_id]:
                    self._queues[session_id][request_id][channel_name] = asyncio.Queue()

            # Push to queue (non-blocking)
            await self._queues[session_id][request_id][channel_name].put(data)

        except Exception as e:
            LOGGER.error(f"Error pushing data to {session_id}/{request_id}/{channel_name}: {e}")
            raise

    async def end(self, request_id: str, channel_name: str, session_id: Optional[str] = None) -> None:
        """Signal end of stream for the specified channel.

        Args:
            request_id: Request identifier
            channel_name: Channel identifier
            session_id: Optional session identifier (defaults to "default")
        """
        if not request_id:
            raise ValueError("request_id cannot be None or empty")

        session_id = session_id or self.DEFAULT_SESSION

        try:
            async with self._lock:
                if channel_name not in self._queues[session_id][request_id]:
                    self._queues[session_id][request_id][channel_name] = asyncio.Queue()

            # Push END sentinel
            await self._queues[session_id][request_id][channel_name].put("__END__")
            LOGGER.debug(f"Pushed END signal to {session_id}/{request_id}/{channel_name}")

        except Exception as e:
            LOGGER.error(f"Error pushing END signal to {session_id}/{request_id}/{channel_name}: {e}")
            raise

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
            session_id: Optional session identifier (defaults to "default")
            timeout: Timeout for each queue.get() operation in seconds
            max_idle_time: Maximum time to wait for new data before stopping.
                          If None, waits indefinitely until END signal.

        Yields:
            Data from the queue until END signal or max_idle_time exceeded
        """
        if not request_id:
            raise ValueError("request_id cannot be None or empty")

        session_id = session_id or self.DEFAULT_SESSION
        last_data_time = time.time()

        try:
            # Ensure queue exists
            async with self._lock:
                if channel_name not in self._queues[session_id][request_id]:
                    self._queues[session_id][request_id][channel_name] = asyncio.Queue()

            queue = self._queues[session_id][request_id][channel_name]

            while True:
                try:
                    # Wait for data with timeout
                    data = await asyncio.wait_for(queue.get(), timeout=timeout)

                    # Check for END sentinel
                    if data == "__END__":
                        break

                    # Update last data time and yield
                    last_data_time = time.time()
                    yield data

                except asyncio.TimeoutError:
                    # Check if max_idle_time exceeded
                    if max_idle_time is not None:
                        idle_time = time.time() - last_data_time
                        if idle_time >= max_idle_time:
                            LOGGER.debug(
                                f"Max idle time ({max_idle_time}s) exceeded for "
                                f"{session_id}/{request_id}/{channel_name}, stopping"
                            )
                            break
                    continue

                except Exception as e:
                    LOGGER.warning(f"Failed to process data from {session_id}/{request_id}/{channel_name}: {e}")
                    continue

        except Exception as e:
            LOGGER.error(f"Error consuming from {session_id}/{request_id}/{channel_name}: {e}")
            raise
        finally:
            # Cleanup
            async with self._lock:
                if (session_id in self._queues and
                    request_id in self._queues[session_id] and
                    channel_name in self._queues[session_id][request_id]):
                    del self._queues[session_id][request_id][channel_name]

    def close(self) -> None:
        """Cleanup resources."""
        try:
            self._queues.clear()
            LOGGER.info("InMemoryStreamService closed")
        except Exception as e:
            LOGGER.error(f"Error closing InMemoryStreamService: {e}")
