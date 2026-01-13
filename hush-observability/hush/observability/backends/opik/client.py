"""Opik client for ResourceHub integration."""

from typing import Any, Dict, List, Optional, Union

from hush.observability.backends.opik.config import OpikConfig


class OpikClient:
    """Enhanced Opik client for LLM observability.

    This client wraps the Opik SDK and is registered to ResourceHub.
    It provides tracing, span creation, and evaluation functionality.

    Opik is an open-source LLM observability platform by Comet that supports:
    - Comprehensive tracing of LLM calls
    - Evaluation metrics (hallucination, relevance, etc.)
    - Production monitoring dashboards

    Example:
        ```python
        from hush.core.registry import get_hub

        # Get client from ResourceHub
        client = get_hub().opik("default")

        # Create a trace
        trace = client.trace(name="my-workflow", input={"query": "Hello"})

        # Add a span
        trace.span(name="llm-call", type="llm", model="gpt-4")

        # Flush traces
        client.flush()
        ```

    References:
        - Documentation: https://www.comet.com/docs/opik/
        - GitHub: https://github.com/comet-ml/opik
    """

    def __init__(self, config: OpikConfig):
        """Initialize Opik client.

        Args:
            config: OpikConfig with API credentials and settings
        """
        self._config = config
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Opik client."""
        if self._client is None:
            try:
                from opik import Opik
            except ImportError:
                raise ImportError(
                    "opik package is required for OpikClient. "
                    "Install it with: pip install opik"
                )

            # Build kwargs, only including non-None values
            kwargs = {}
            if self._config.project_name:
                kwargs["project_name"] = self._config.project_name
            if self._config.workspace:
                kwargs["workspace"] = self._config.workspace
            if self._config.host:
                kwargs["host"] = self._config.host
            if self._config.api_key:
                kwargs["api_key"] = self._config.api_key

            self._client = Opik(**kwargs)
        return self._client

    @property
    def config(self) -> OpikConfig:
        """Get the configuration."""
        return self._config

    # Delegate common methods to underlying client

    def trace(
        self,
        id: Optional[str] = None,
        name: Optional[str] = None,
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        project_name: Optional[str] = None,
        **kwargs,
    ):
        """Create a new trace.

        Args:
            id: Optional trace ID (auto-generated if not provided)
            name: Name of the trace
            input: Input data for the trace
            output: Output data for the trace
            metadata: Additional metadata
            tags: Tags for categorization
            project_name: Override default project name
            **kwargs: Additional arguments passed to Opik

        Returns:
            Opik Trace object
        """
        return self.client.trace(
            id=id,
            name=name,
            input=input,
            output=output,
            metadata=metadata,
            tags=tags,
            project_name=project_name,
            **kwargs,
        )

    def span(
        self,
        trace_id: Optional[str] = None,
        id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        name: Optional[str] = None,
        type: str = "general",
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        model: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Create a new span.

        Args:
            trace_id: ID of parent trace
            id: Optional span ID (auto-generated if not provided)
            parent_span_id: ID of parent span (for nested spans)
            name: Name of the span
            type: Span type ("general", "llm", "tool", etc.)
            input: Input data for the span
            output: Output data for the span
            metadata: Additional metadata
            tags: Tags for categorization
            model: Model name (for LLM spans)
            usage: Token usage information
            **kwargs: Additional arguments passed to Opik

        Returns:
            Opik Span object
        """
        return self.client.span(
            trace_id=trace_id,
            id=id,
            parent_span_id=parent_span_id,
            name=name,
            type=type,
            input=input,
            output=output,
            metadata=metadata,
            tags=tags,
            model=model,
            usage=usage,
            **kwargs,
        )

    def flush(self, timeout: Optional[int] = None):
        """Flush all pending traces to Opik server.

        Args:
            timeout: Optional timeout in seconds
        """
        if timeout:
            return self.client.flush(timeout=timeout)
        return self.client.flush()

    def end(self):
        """End the client session and flush remaining data."""
        return self.client.end()

    def __repr__(self) -> str:
        """String representation."""
        if self._config.host:
            return f"<OpikClient host={self._config.host}>"
        return f"<OpikClient workspace={self._config.workspace}>"
