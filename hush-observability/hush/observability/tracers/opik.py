"""Opik tracer implementation using subprocess-based flushing.

This tracer inherits from hush.core.tracers.BaseTracer and uses
ResourceHub to get the OpikClient in the subprocess.
"""

from typing import Any, Dict, Optional

from hush.core.tracers import BaseTracer, register_tracer


@register_tracer
class OpikTracer(BaseTracer):
    """Tracer that sends workflow traces to Opik.

    This tracer uses subprocess-based flushing to avoid blocking
    the main workflow execution. It references an OpikClient
    from ResourceHub using a resource_key.

    Opik is an open-source LLM observability platform by Comet that supports:
    - Comprehensive tracing of LLM calls
    - Evaluation metrics
    - Production monitoring dashboards

    Example:
        ```python
        from hush.observability import OpikTracer

        # Use with ResourceHub (recommended)
        tracer = OpikTracer(resource_key="opik:default")

        # Use with workflow engine
        workflow = MyWorkflow(tracer=tracer)
        result = await workflow.run(inputs={...})
        # Traces are automatically flushed in background
        ```

    References:
        - Documentation: https://www.comet.com/docs/opik/
        - GitHub: https://github.com/comet-ml/opik
    """

    def __init__(self, resource_key: str = "opik:default"):
        """Initialize the Opik tracer.

        Args:
            resource_key: ResourceHub key for OpikClient (e.g., "opik:default")
        """
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
        - Load OpikClient from ResourceHub

        Args:
            flush_data: Dictionary containing all data needed for flushing
        """
        from hush.core.loggings import LOGGER

        try:
            from hush.core.registry import get_hub

            tracer_config = flush_data["tracer_config"]
            resource_key = tracer_config["resource_key"]

            # Get OpikClient from ResourceHub
            client = get_hub().opik(resource_key)

            workflow_name = flush_data["workflow_name"]
            req_id = flush_data["request_id"]
            user_id = flush_data.get("user_id")
            session_id = flush_data.get("session_id")
            execution_order = flush_data["execution_order"]
            nodes_trace_data = flush_data["nodes_trace_data"]

            LOGGER.info(
                "Workflow: %s, Request ID: %s, Creating Opik trace hierarchy...",
                workflow_name,
                req_id,
            )

            # Track created Opik objects for parent-child linking
            opik_objects: Dict[str, Any] = {}
            root_trace = None

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

                # Remove media_attachments (Opik doesn't support media like Langfuse)
                trace_data.pop("media_attachments", None)

                # Build unique key for context-aware nodes
                if context_id:
                    node_id = f"{node_id}:{context_id}"

                # Check for context-aware parent
                if parent_id and context_id:
                    context_parent_id = f"{parent_id}:{context_id}"
                    if context_parent_id in opik_objects:
                        parent_id = context_parent_id

                if parent_id is None:
                    # Root node - create trace
                    root_trace = client.trace(
                        id=req_id,
                        name=workflow_name,
                        input=trace_data.get("input"),
                        output=trace_data.get("output"),
                        metadata={
                            **(trace_data.get("metadata") or {}),
                            "user_id": user_id,
                            "session_id": session_id,
                        },
                        tags=[workflow_name] if workflow_name else None,
                    )
                    opik_objects[node_id] = {
                        "trace": root_trace,
                        "trace_id": req_id,
                        "span_id": None,
                    }
                else:
                    # Child node - create span
                    parent_info = opik_objects.get(parent_id)
                    if parent_info is None:
                        LOGGER.warning(
                            "Parent '%s' not found for node '%s'",
                            parent_id,
                            node_id,
                        )
                        continue

                    # Determine span type based on whether it contains generation
                    span_type = "llm" if contain_generation else "general"

                    # Build span kwargs
                    span_kwargs: Dict[str, Any] = {
                        "trace_id": parent_info["trace_id"],
                        "name": trace_data.get("name", node_id),
                        "type": span_type,
                        "input": trace_data.get("input"),
                        "output": trace_data.get("output"),
                        "metadata": trace_data.get("metadata"),
                    }

                    # Add parent_span_id for nested spans
                    if parent_info.get("span_id"):
                        span_kwargs["parent_span_id"] = parent_info["span_id"]

                    # Add model info for LLM spans
                    if contain_generation:
                        if "model" in trace_data:
                            span_kwargs["model"] = trace_data["model"]
                        if "usage" in trace_data:
                            span_kwargs["usage"] = trace_data["usage"]

                    span = client.span(**span_kwargs)

                    opik_objects[node_id] = {
                        "trace": root_trace,
                        "trace_id": parent_info["trace_id"],
                        "span_id": span.id if hasattr(span, "id") else None,
                    }

            # Ensure all traces are sent to Opik
            client.flush()

            LOGGER.info(
                "Workflow: %s, Request ID: %s, Opik trace created successfully.",
                workflow_name,
                req_id,
            )

        except ImportError as e:
            from hush.core.loggings import LOGGER

            LOGGER.error(
                "opik package is required for OpikTracer. "
                "Install it with: pip install opik. Error: %s",
                str(e),
            )

        except Exception as e:
            import traceback

            from hush.core.loggings import LOGGER

            LOGGER.error(
                "Opik flush failed: %s\nTraceback:\n%s",
                str(e),
                traceback.format_exc(),
            )

    def __repr__(self) -> str:
        """String representation."""
        return f"<OpikTracer resource_key={self.resource_key}>"
