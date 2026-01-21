"""Local tracer that stores traces in SQLite only.

This tracer is useful for:
- Local development and debugging
- Testing the tracing infrastructure
- Offline trace collection for later analysis

Traces are stored in ~/.hush/traces.db by default.
"""

from typing import Any, Dict

from hush.core.tracers.base import BaseTracer, register_tracer


@register_tracer
class LocalTracer(BaseTracer):
    """Tracer that stores traces locally in SQLite without external flushing.

    This is the simplest tracer implementation - it just marks traces as
    complete and lets the background process write them to SQLite.
    The flush() method does nothing since traces are already persisted.

    Useful for:
    - Local development
    - Testing
    - Offline analysis via SQL queries

    Example:
        ```python
        from hush.core.tracers import LocalTracer

        tracer = LocalTracer()

        # Use with workflow engine
        engine = Hush(graph)
        result = await engine.run(inputs={...}, tracer=tracer)

        # Traces are now in ~/.hush/traces.db
        # Query with: SELECT * FROM traces WHERE request_id = '...'
        ```
    """

    def __init__(self, name: str = "local"):
        """Initialize LocalTracer.

        Args:
            name: Optional name for this tracer instance
        """
        self.name = name

    def _get_tracer_config(self) -> Dict[str, Any]:
        """Return tracer configuration.

        Returns:
            Dict with tracer name
        """
        return {"name": self.name}

    @staticmethod
    def flush(flush_data: Dict[str, Any]) -> None:
        """No-op flush - traces are already in SQLite.

        This method is called by the background process but does nothing
        since LocalTracer only stores traces locally.

        Args:
            flush_data: Dictionary containing trace data (ignored)
        """
        # Do nothing - traces are already persisted in SQLite
        # Just log that we "flushed" for debugging
        from hush.core.loggings import LOGGER

        request_id = flush_data.get("request_id", "unknown")
        workflow_name = flush_data.get("workflow_name", "unknown")

        LOGGER.debug(
            "[title]\\[%s][/title] LocalTracer: traces stored for [highlight]%s[/highlight]",
            request_id,
            workflow_name,
        )

    def __repr__(self) -> str:
        """String representation."""
        return f"<LocalTracer name={self.name}>"