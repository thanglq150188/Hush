"""Base tracer interface for all observability backends."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


class BaseTracer(ABC):
    """Abstract base class for all tracer implementations.

    This interface ensures all tracer backends (Langfuse, Phoenix, Opik, etc.)
    provide a consistent API for tracing operations.

    All tracers use a buffering approach where trace items are collected
    and then flushed to the backend in hierarchical order.
    """

    @abstractmethod
    def add_span(
        self,
        request_id: str,
        name: str,
        parent: Optional[str] = None,
        **kwargs
    ) -> None:
        """Add a span to the trace buffer.

        Args:
            request_id: Unique identifier for the request/trace
            name: Unique name of the span within the request
            parent: Name of parent span/generation, None for root
            **kwargs: Span parameters (input, output, metadata, etc.)
                     May include trace metadata (user_id, session_id, tags)

        Example:
            tracer.add_span(
                request_id="req-123",
                name="data-processing",
                parent=None,
                input={"data": [1, 2, 3]},
                user_id="user-456",
                tags=["production"]
            )
        """
        pass

    @abstractmethod
    def add_generation(
        self,
        request_id: str,
        name: str,
        parent: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> None:
        """Add a generation (LLM call) to the trace buffer.

        Args:
            request_id: Unique identifier for the request/trace
            name: Unique name of the generation within the request
            parent: Name of parent span/generation, None for root
            model: Model name (e.g., "gpt-4", "claude-3-sonnet")
            **kwargs: Generation parameters (input, output, usage_details, etc.)
                     May include trace metadata (user_id, session_id, tags)

        Example:
            tracer.add_generation(
                request_id="req-123",
                name="llm-call",
                parent="root-span",
                model="gpt-4",
                input=[{"role": "user", "content": "Hello"}],
                output="Hi there!",
                usage_details={"prompt_tokens": 10, "completion_tokens": 5}
            )
        """
        pass

    @abstractmethod
    def add_event(
        self,
        request_id: str,
        name: str,
        parent: Optional[str] = None,
        **kwargs
    ) -> None:
        """Add an event to the trace buffer.

        Args:
            request_id: Unique identifier for the request/trace
            name: Unique name of the event within the request
            parent: Name of parent span/generation, None for root
            **kwargs: Event parameters (level, message, etc.)

        Example:
            tracer.add_event(
                request_id="req-123",
                name="cache-hit",
                parent="db-span",
                level="info",
                message="Cache hit for key: user-456"
            )
        """
        pass

    @abstractmethod
    def update_item(
        self,
        request_id: str,
        name: str,
        **kwargs
    ) -> bool:
        """Update an existing item in the trace buffer.

        Useful for adding outputs or metadata after initial creation.

        Args:
            request_id: Unique identifier for the request/trace
            name: Name of the item to update
            **kwargs: Data to update (output, metadata, etc.)

        Returns:
            True if item was found and updated, False otherwise

        Example:
            # Add generation without output
            tracer.add_generation(request_id="req-1", name="llm", model="gpt-4", ...)

            # Later, update with output
            tracer.update_item(
                request_id="req-1",
                name="llm",
                output="Response text",
                usage_details={...}
            )
        """
        pass

    @abstractmethod
    async def flush(self, request_id: str) -> bool:
        """Flush buffered traces for a request to the backend.

        Processes all buffered items for the request in hierarchical order
        (parents before children) and sends them to the observability backend.

        Args:
            request_id: Request ID to flush

        Returns:
            True if request was found and flushed successfully, False otherwise

        Raises:
            ValueError: If hierarchy validation fails
            Exception: If backend operations fail

        Example:
            # Add trace items
            tracer.add_span(request_id="req-1", ...)
            tracer.add_generation(request_id="req-1", ...)

            # Flush to backend
            await tracer.flush("req-1")
        """
        pass

    @abstractmethod
    async def flush_all(self) -> Dict[str, bool]:
        """Flush all buffered requests to the backend.

        Returns:
            Dictionary mapping request_id -> success status

        Example:
            results = await tracer.flush_all()
            # {"req-1": True, "req-2": False, ...}
        """
        pass

    @abstractmethod
    def clear_request(self, request_id: str) -> bool:
        """Clear buffered data for a request without flushing.

        Useful for discarding traces that shouldn't be sent to the backend.

        Args:
            request_id: Request ID to clear

        Returns:
            True if request was found and cleared, False otherwise

        Example:
            # Clear request without sending to backend
            tracer.clear_request("req-1")
        """
        pass

    def __repr__(self) -> str:
        """String representation of the tracer."""
        return f"<{self.__class__.__name__}>"
