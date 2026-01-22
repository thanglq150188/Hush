"""Arize Phoenix tracer implementation using subprocess-based flushing.

This tracer inherits from hush.core.tracers.BaseTracer and uses
ResourceHub to get the PhoenixClient in the subprocess.

Phoenix is built on OpenTelemetry, so the trace structure is similar
to OTELTracer but with Phoenix-specific features.
"""

from typing import Any, Dict, List, Optional

from hush.core.tracers import BaseTracer, register_tracer


@register_tracer
class PhoenixTracer(BaseTracer):
    """Tracer that sends workflow traces to Arize Phoenix.

    This tracer uses subprocess-based flushing to avoid blocking
    the main workflow execution. It references a PhoenixClient
    from ResourceHub using a resource_key.

    Phoenix provides:
    - LLM observability with prompt/response tracking
    - Embeddings visualization
    - Evaluation capabilities
    - Local and cloud deployment options

    Example:
        ```python
        from hush.observability import PhoenixTracer

        # Use with ResourceHub (recommended)
        tracer = PhoenixTracer(resource_key="phoenix:local", tags=["prod", "ml-team"])

        # Use with workflow engine
        workflow = MyWorkflow(tracer=tracer)
        result = await workflow.run(inputs={...})
        # Traces are automatically flushed in background
        ```

    References:
        - Documentation: https://arize.com/docs/phoenix
        - GitHub: https://github.com/Arize-ai/phoenix
    """

    def __init__(self, resource_key: str = "phoenix:default", tags: Optional[List[str]] = None):
        """Initialize the Phoenix tracer.

        Args:
            resource_key: ResourceHub key for PhoenixClient (e.g., "phoenix:local")
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
        - Load PhoenixClient from ResourceHub

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

            # Get PhoenixClient from ResourceHub
            client = get_hub().phoenix(resource_key)

            workflow_name = flush_data["workflow_name"]
            req_id = flush_data["request_id"]
            user_id = flush_data.get("user_id")
            session_id = flush_data.get("session_id")
            tags = flush_data.get("tags", [])
            execution_order = flush_data["execution_order"]
            nodes_trace_data = flush_data["nodes_trace_data"]

            LOGGER.info(
                "Workflow: %s, Request ID: %s, Creating Phoenix trace hierarchy...",
                workflow_name,
                req_id,
            )

            # Get the tracer from client
            phoenix_tracer = client.tracer

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

                # Remove media_attachments (Phoenix doesn't support media like Langfuse)
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

                # Add LLM-specific attributes for generations (OpenInference semantic conventions)
                if contain_generation:
                    attributes["openinference.span.kind"] = "LLM"
                    if "model" in trace_data:
                        attributes["llm.model_name"] = trace_data["model"]
                    if "usage" in trace_data:
                        usage = trace_data["usage"]
                        if "input" in usage:
                            attributes["llm.token_count.prompt"] = usage["input"]
                        if "output" in usage:
                            attributes["llm.token_count.completion"] = usage["output"]
                        if "total" in usage:
                            attributes["llm.token_count.total"] = usage["total"]

                # Add input/output as attributes (serialized)
                if trace_data.get("input"):
                    try:
                        import json

                        input_str = json.dumps(trace_data["input"])
                        if len(input_str) < 10000:  # Limit attribute size
                            attributes["input.value"] = input_str
                    except (TypeError, ValueError):
                        pass

                if trace_data.get("output"):
                    try:
                        import json

                        output_str = json.dumps(trace_data["output"])
                        if len(output_str) < 10000:  # Limit attribute size
                            attributes["output.value"] = output_str
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
                    span = phoenix_tracer.start_span(
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
                    span = phoenix_tracer.start_span(
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
                "Workflow: %s, Request ID: %s, Phoenix trace created successfully.",
                workflow_name,
                req_id,
            )

        except ImportError as e:
            from hush.core.loggings import LOGGER

            LOGGER.error(
                "arize-phoenix-otel is required for PhoenixTracer. "
                "Install it with: pip install arize-phoenix-otel. Error: %s",
                str(e),
            )

        except Exception as e:
            import traceback

            from hush.core.loggings import LOGGER

            LOGGER.error(
                "Phoenix flush failed: %s\nTraceback:\n%s",
                str(e),
                traceback.format_exc(),
            )

    def __repr__(self) -> str:
        """String representation."""
        return f"<PhoenixTracer resource_key={self.resource_key}>"
