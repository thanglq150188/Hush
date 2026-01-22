"""OpenTelemetry tracer implementation using subprocess-based flushing.

This tracer inherits from hush.core.tracers.BaseTracer and uses
ResourceHub to get the OTELClient in the subprocess.
"""

from typing import Any, Dict, List, Optional

from hush.core.tracers import BaseTracer, register_tracer


@register_tracer
class OTELTracer(BaseTracer):
    """Tracer that sends workflow traces via OpenTelemetry.

    This tracer uses subprocess-based flushing to avoid blocking
    the main workflow execution. It references an OTELClient
    from ResourceHub using a resource_key.

    OpenTelemetry is a vendor-neutral observability framework that
    can export traces to any OTLP-compatible backend (Jaeger, Zipkin,
    Datadog, New Relic, Grafana Tempo, etc.).

    Example:
        ```python
        from hush.observability import OTELTracer

        # Use with ResourceHub (recommended)
        tracer = OTELTracer(resource_key="otel:jaeger", tags=["prod", "ml-team"])

        # Use with workflow engine
        workflow = MyWorkflow(tracer=tracer)
        result = await workflow.run(inputs={...})
        # Traces are automatically flushed in background
        ```

    References:
        - Documentation: https://opentelemetry.io/docs/languages/python/
        - GitHub: https://github.com/open-telemetry/opentelemetry-python
    """

    def __init__(self, resource_key: str = "otel:default", tags: Optional[List[str]] = None):
        """Initialize the OpenTelemetry tracer.

        Args:
            resource_key: ResourceHub key for OTELClient (e.g., "otel:jaeger")
            tags: Optional list of static tags for filtering/grouping traces
        """
        super().__init__(tags=tags)
        self.resource_key = resource_key

    def _get_tracer_config(self) -> Dict[str, Any]:
        """Return configuration for subprocess.

        Returns resource_key so subprocess can load client from ResourceHub.
        """
        return {"resource_key": self.resource_key}

    @staticmethod
    def flush(flush_data: Dict[str, Any]) -> None:
        """Execute flush logic in a separate process.

        This method runs in a subprocess, so it must:
        - Re-import all dependencies
        - Use only the data provided in flush_data
        - Load OTELClient from ResourceHub

        Args:
            flush_data: Dictionary containing all data needed for flushing
        """
        from hush.core.loggings import LOGGER

        try:
            from opentelemetry import trace
            from opentelemetry.trace import Status, StatusCode

            from hush.core.registry import get_hub

            tracer_config = flush_data["tracer_config"]
            resource_key = tracer_config["resource_key"]

            # Get OTELClient from ResourceHub
            client = get_hub().otel(resource_key)

            workflow_name = flush_data["workflow_name"]
            req_id = flush_data["request_id"]
            user_id = flush_data.get("user_id")
            session_id = flush_data.get("session_id")
            tags = flush_data.get("tags", [])
            execution_order = flush_data["execution_order"]
            nodes_trace_data = flush_data["nodes_trace_data"]

            LOGGER.info(
                "Workflow: %s, Request ID: %s, Creating OpenTelemetry trace hierarchy...",
                workflow_name,
                req_id,
            )

            # Get the tracer from client
            otel_tracer = client.tracer

            # Track created spans for parent-child linking
            spans: Dict[str, Any] = {}

            for execution in execution_order:
                node_id = execution["node"]
                parent_id = execution["parent"]
                context_id = execution.get("context_id")
                contain_generation = execution.get("contain_generation", False)

                # Get trace data for this node
                trace_key = f"{node_id}:{context_id}" if context_id else node_id
                trace_data = nodes_trace_data.get(trace_key)
                if trace_data is None:
                    continue

                # Remove media_attachments (OTEL doesn't support media like Langfuse)
                trace_data.pop("media_attachments", None)

                # Build unique key for context-aware nodes
                if context_id:
                    node_id = f"{node_id}:{context_id}"

                # Check for context-aware parent
                if parent_id and context_id:
                    context_parent_id = f"{parent_id}:{context_id}"
                    if context_parent_id in spans:
                        parent_id = context_parent_id

                # Build span attributes
                attributes = {
                    "workflow.name": workflow_name,
                    "workflow.request_id": req_id,
                    "node.name": trace_data.get("name", node_id),
                }

                if user_id:
                    attributes["user.id"] = user_id
                if session_id:
                    attributes["session.id"] = session_id
                if tags:
                    attributes["tags"] = ",".join(tags)

                # Add LLM-specific attributes for generations
                if contain_generation:
                    attributes["llm.request.type"] = "generation"
                    if "model" in trace_data:
                        attributes["llm.model"] = trace_data["model"]
                    if "usage" in trace_data:
                        usage = trace_data["usage"]
                        if "input" in usage:
                            attributes["llm.usage.prompt_tokens"] = usage["input"]
                        if "output" in usage:
                            attributes["llm.usage.completion_tokens"] = usage["output"]
                        if "total" in usage:
                            attributes["llm.usage.total_tokens"] = usage["total"]

                # Add input/output as attributes (serialized)
                if trace_data.get("input"):
                    try:
                        import json

                        input_str = json.dumps(trace_data["input"])
                        if len(input_str) < 10000:  # Limit attribute size
                            attributes["input"] = input_str
                    except (TypeError, ValueError):
                        pass

                if trace_data.get("output"):
                    try:
                        import json

                        output_str = json.dumps(trace_data["output"])
                        if len(output_str) < 10000:  # Limit attribute size
                            attributes["output"] = output_str
                    except (TypeError, ValueError):
                        pass

                # Add metadata as attributes
                if trace_data.get("metadata"):
                    for key, value in trace_data["metadata"].items():
                        if isinstance(value, (str, int, float, bool)):
                            attributes[f"metadata.{key}"] = value

                # Create span
                if parent_id is None:
                    # Root span
                    span = otel_tracer.start_span(
                        name=workflow_name,
                        attributes=attributes,
                    )
                    spans[node_id] = span
                else:
                    # Child span - need to use parent context
                    parent_span = spans.get(parent_id)
                    if parent_span is None:
                        LOGGER.warning(
                            "Parent '%s' not found for node '%s'",
                            parent_id,
                            node_id,
                        )
                        continue

                    # Create child span with parent context
                    ctx = trace.set_span_in_context(parent_span)
                    span = otel_tracer.start_span(
                        name=trace_data.get("name", node_id),
                        context=ctx,
                        attributes=attributes,
                    )
                    spans[node_id] = span

            # End all spans (in reverse order - children first)
            for span in reversed(list(spans.values())):
                span.set_status(Status(StatusCode.OK))
                span.end()

            # Flush to ensure traces are sent
            client.flush()

            LOGGER.info(
                "Workflow: %s, Request ID: %s, OpenTelemetry trace created successfully.",
                workflow_name,
                req_id,
            )

        except ImportError as e:
            from hush.core.loggings import LOGGER

            LOGGER.error(
                "opentelemetry packages are required for OTELTracer. "
                "Install them with: pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-otlp-proto-grpc. Error: %s",
                str(e),
            )

        except Exception as e:
            import traceback

            from hush.core.loggings import LOGGER

            LOGGER.error(
                "OpenTelemetry flush failed: %s\nTraceback:\n%s",
                str(e),
                traceback.format_exc(),
            )

    def __repr__(self) -> str:
        """String representation."""
        return f"<OTELTracer resource_key={self.resource_key}>"
