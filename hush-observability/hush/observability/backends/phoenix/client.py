"""Phoenix client for tracing.

This client wraps the phoenix.otel module to provide tracing capabilities
with lazy initialization for subprocess-based flushing.

Phoenix is built on OpenTelemetry and provides:
- Automatic instrumentation for LLM frameworks
- Manual span creation
- Trace visualization and debugging
- Evaluation capabilities

References:
    - Documentation: https://arize.com/docs/phoenix
    - Phoenix OTEL: https://arize.com/docs/phoenix/sdk-api-reference/python/arize-phoenix-otel
"""

from typing import Any, Optional

from hush.core.loggings import LOGGER

from hush.observability.backends.phoenix.config import PhoenixConfig


class PhoenixClient:
    """Client for Arize Phoenix observability platform.

    This client uses lazy initialization to create the TracerProvider
    only when needed, which is important for subprocess-based flushing.

    Example:
        ```python
        from hush.core.registry import get_hub

        # Get client from ResourceHub
        client = get_hub().phoenix("local")

        # Get the tracer for manual span creation
        tracer = client.tracer

        # Create spans
        with tracer.start_as_current_span("my-operation") as span:
            span.set_attribute("key", "value")
            # ... do work
        ```

    Attributes:
        config: PhoenixConfig instance with connection settings.
    """

    def __init__(self, config: PhoenixConfig):
        """Initialize the Phoenix client.

        Args:
            config: PhoenixConfig with endpoint, project_name, api_key, etc.
        """
        self.config = config
        self._tracer_provider: Optional[Any] = None
        self._tracer: Optional[Any] = None

    @property
    def tracer_provider(self) -> Any:
        """Get the OpenTelemetry TracerProvider (lazy initialization).

        Returns:
            The TracerProvider configured for Phoenix.

        Raises:
            ImportError: If phoenix.otel is not installed.
        """
        if self._tracer_provider is None:
            self._initialize()
        return self._tracer_provider

    @property
    def tracer(self) -> Any:
        """Get an OpenTelemetry Tracer for creating spans.

        Returns:
            An OpenTelemetry Tracer instance.
        """
        if self._tracer is None:
            from opentelemetry import trace

            self.tracer_provider  # Ensure initialized
            self._tracer = trace.get_tracer("hush.observability.phoenix")
        return self._tracer

    def _initialize(self) -> None:
        """Initialize the Phoenix TracerProvider.

        This method is called lazily when the tracer_provider is first accessed.
        """
        try:
            from phoenix.otel import register

            LOGGER.debug(
                "Initializing Phoenix TracerProvider for project '%s' at endpoint '%s'",
                self.config.project_name,
                self.config.endpoint,
            )

            headers = self.config.get_headers()

            self._tracer_provider = register(
                endpoint=self.config.endpoint,
                project_name=self.config.project_name,
                headers=headers if headers else None,
                batch=self.config.batch,
                verbose=False,  # Don't print to stdout
            )

            LOGGER.info(
                "Phoenix TracerProvider initialized for project '%s'",
                self.config.project_name,
            )

        except ImportError as e:
            LOGGER.error(
                "arize-phoenix-otel is required for PhoenixClient. "
                "Install it with: pip install arize-phoenix-otel. Error: %s",
                str(e),
            )
            raise

    def flush(self, timeout_millis: int = 30000) -> bool:
        """Flush all pending spans to Phoenix.

        Args:
            timeout_millis: Maximum time to wait for flush in milliseconds.

        Returns:
            True if flush was successful.
        """
        if self._tracer_provider is None:
            return True  # Nothing to flush

        try:
            # TracerProvider has a force_flush method
            if hasattr(self._tracer_provider, "force_flush"):
                return self._tracer_provider.force_flush(timeout_millis)
            return True
        except Exception as e:
            LOGGER.warning("Failed to flush Phoenix spans: %s", str(e))
            return False

    def shutdown(self) -> None:
        """Shutdown the TracerProvider and release resources."""
        if self._tracer_provider is not None:
            try:
                if hasattr(self._tracer_provider, "shutdown"):
                    self._tracer_provider.shutdown()
                LOGGER.debug("Phoenix TracerProvider shut down")
            except Exception as e:
                LOGGER.warning("Error shutting down Phoenix TracerProvider: %s", str(e))
            finally:
                self._tracer_provider = None
                self._tracer = None

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"<PhoenixClient endpoint={self.config.endpoint} "
            f"project={self.config.project_name}>"
        )
