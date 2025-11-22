"""Langfuse tracer implementation."""

import os
from typing import Optional, Dict, Any

try:
    from langfuse import Langfuse
    import httpx
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

from .base import BaseTracer
from .buffer import AsyncTraceBuffer, BackendAdapter, TraceItem
from ..config.langfuse import LangfuseConfig


class LangfuseBackendAdapter:
    """Backend adapter for Langfuse.

    Handles the creation of Langfuse-specific trace objects.
    Note: Since the buffer processes one request at a time within flush(),
    we can use a single _current_trace variable that gets reset for each flush.
    """

    def __init__(self, client: 'Langfuse'):
        """Initialize adapter with Langfuse client."""
        self.client = client
        self._current_trace = None  # Reset for each flush operation

    def create_span(self, item: TraceItem, parent_obj: Optional[Any]) -> Any:
        """Create a Langfuse span."""
        data = item.data.copy()
        name = data.pop("name", "unnamed-span")

        if parent_obj:
            # Child span
            span = parent_obj.span(name=name, **data)
        else:
            # Root span - create a trace first
            if self._current_trace is None:
                trace_metadata = item.trace_metadata.copy() if item.trace_metadata else {}
                trace_name = trace_metadata.pop("name", name)
                self._current_trace = self.client.trace(name=trace_name, **trace_metadata)
            span = self._current_trace.span(name=name, **data)

        return span

    def create_generation(self, item: TraceItem, parent_obj: Optional[Any]) -> Any:
        """Create a Langfuse generation."""
        data = item.data.copy()
        name = data.pop("name", "unnamed-generation")

        if parent_obj:
            # Child generation
            generation = parent_obj.generation(name=name, **data)
        else:
            # Root generation - create a trace first
            if self._current_trace is None:
                trace_metadata = item.trace_metadata.copy() if item.trace_metadata else {}
                trace_name = trace_metadata.pop("name", name)
                self._current_trace = self.client.trace(name=trace_name, **trace_metadata)
            generation = self._current_trace.generation(name=name, **data)

        return generation

    def end_item(self, obj: Any) -> None:
        """End a Langfuse span or generation."""
        if hasattr(obj, 'end'):
            obj.end()

    def flush_backend(self) -> None:
        """Flush Langfuse client."""
        self.client.flush()
        self._current_trace = None  # Reset for next flush


class LangfuseTracer(BaseTracer):
    """Langfuse implementation of BaseTracer.

    Provides full Langfuse observability integration with buffered tracing.

    Example:
        ```python
        from hush.observability import LangfuseTracer, LangfuseConfig

        config = LangfuseConfig(
            public_key="pk-...",
            secret_key="sk-...",
            host="https://cloud.langfuse.com"
        )

        tracer = LangfuseTracer(config)

        # Add traces
        tracer.add_span(
            request_id="req-1",
            name="root",
            input={"query": "Hello"},
            user_id="user-123"
        )

        tracer.add_generation(
            request_id="req-1",
            name="llm-call",
            parent="root",
            model="gpt-4",
            input=[{"role": "user", "content": "Hello"}],
            output="Hi there!"
        )

        # Flush to Langfuse
        await tracer.flush("req-1")
        ```
    """

    def __init__(self, config: LangfuseConfig):
        """Initialize Langfuse tracer.

        Args:
            config: Langfuse configuration

        Raises:
            ImportError: If langfuse package is not installed
        """
        if not LANGFUSE_AVAILABLE:
            raise ImportError(
                "langfuse package is required for LangfuseTracer. "
                "Install it with: pip install langfuse"
            )

        self.config = config
        self.client = self._initialize_client()

        # Create backend adapter and buffer
        adapter = LangfuseBackendAdapter(self.client)
        self.buffer = AsyncTraceBuffer(backend=adapter)

    def _initialize_client(self) -> 'Langfuse':
        """Initialize and validate Langfuse client."""
        if self.config.no_proxy:
            os.environ.update({"NO_PROXY": self.config.no_proxy})

        client = Langfuse(
            public_key=self.config.public_key,
            secret_key=self.config.secret_key,
            host=self.config.host
        )

        # Validate authentication
        client.auth_check()

        return client

    def add_span(
        self,
        request_id: str,
        name: str,
        parent: Optional[str] = None,
        **kwargs
    ) -> None:
        """Add a span to the trace buffer."""
        self.buffer.add_span(request_id, name, parent, **kwargs)

    def add_generation(
        self,
        request_id: str,
        name: str,
        parent: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ) -> None:
        """Add a generation to the trace buffer."""
        self.buffer.add_generation(request_id, name, parent, model, **kwargs)

    def add_event(
        self,
        request_id: str,
        name: str,
        parent: Optional[str] = None,
        **kwargs
    ) -> None:
        """Add an event to the trace buffer."""
        self.buffer.add_event(request_id, name, parent, **kwargs)

    def update_item(
        self,
        request_id: str,
        name: str,
        **kwargs
    ) -> bool:
        """Update an existing item in the buffer."""
        return self.buffer.update_item(request_id, name, **kwargs)

    async def flush(self, request_id: str) -> bool:
        """Flush buffered traces for a request to Langfuse."""
        return await self.buffer.flush(request_id)

    async def flush_all(self) -> Dict[str, bool]:
        """Flush all buffered requests to Langfuse."""
        return await self.buffer.flush_all()

    def clear_request(self, request_id: str) -> bool:
        """Clear buffered data for a request without flushing."""
        return self.buffer.clear_request(request_id)

    def __repr__(self) -> str:
        """String representation."""
        return f"<LangfuseTracer host={self.config.host}>"
