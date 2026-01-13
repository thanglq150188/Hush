"""Langfuse tracer implementation using subprocess-based flushing.

This tracer inherits from hush.core.tracers.BaseTracer and uses
ResourceHub to get the LangfuseClient in the subprocess.
"""

import base64
from typing import Any, Dict, Optional

from hush.core.tracers import BaseTracer, register_tracer


@register_tracer
class LangfuseTracer(BaseTracer):
    """Tracer that sends workflow traces to Langfuse.

    This tracer uses subprocess-based flushing to avoid blocking
    the main workflow execution. It references a LangfuseClient
    from ResourceHub using a resource_key.

    Example:
        ```python
        from hush.observability import LangfuseTracer

        # Use with ResourceHub (recommended)
        tracer = LangfuseTracer(resource_key="langfuse:vpbank")

        # Use with workflow engine
        workflow = MyWorkflow(tracer=tracer)
        result = await workflow.run(inputs={...})
        # Traces are automatically flushed in background
        ```
    """

    def __init__(self, resource_key: str = "langfuse:default"):
        """Initialize the Langfuse tracer.

        Args:
            resource_key: ResourceHub key for LangfuseClient (e.g., "langfuse:vpbank")
        """
        self.resource_key = resource_key

    def _get_tracer_config(self) -> Dict[str, Any]:
        """Return configuration for subprocess.

        Returns resource_key so subprocess can load client from ResourceHub.
        """
        return {"resource_key": self.resource_key}

    @staticmethod
    def _resolve_media(
        trace_data: Dict[str, Any],
        media_attachments: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Resolve media attachments to LangfuseMedia objects.

        Args:
            trace_data: Trace data dict (input, output, metadata)
            media_attachments: Serialized media attachments

        Returns:
            Modified trace_data with LangfuseMedia objects injected
        """
        if not media_attachments:
            return trace_data

        try:
            from langfuse.media import LangfuseMedia
        except ImportError:
            return trace_data

        # Group attachments by target location
        by_location: Dict[str, Dict[str, Any]] = {
            "input": {},
            "output": {},
            "metadata": {},
        }

        for key, attachment in media_attachments.items():
            attach_to = attachment.get("attach_to", "metadata")
            content_type = attachment["content_type"]

            # Get content bytes
            if "base64" in attachment:
                content_bytes = base64.b64decode(attachment["base64"])
            elif "path" in attachment:
                with open(attachment["path"], "rb") as f:
                    content_bytes = f.read()
            else:
                continue

            # Create LangfuseMedia object
            media_obj = LangfuseMedia(
                content_bytes=content_bytes, content_type=content_type
            )
            by_location[attach_to][key] = media_obj

        # Inject into trace_data
        result = trace_data.copy()

        if by_location["input"]:
            if isinstance(result.get("input"), dict):
                result["input"] = {**result["input"], **by_location["input"]}
            else:
                result["input"] = {
                    "_original": result.get("input"),
                    **by_location["input"],
                }

        if by_location["output"]:
            if isinstance(result.get("output"), dict):
                result["output"] = {**result["output"], **by_location["output"]}
            else:
                result["output"] = {
                    "_original": result.get("output"),
                    **by_location["output"],
                }

        if by_location["metadata"]:
            if isinstance(result.get("metadata"), dict):
                result["metadata"] = {**result["metadata"], **by_location["metadata"]}
            else:
                result["metadata"] = by_location["metadata"]

        return result

    @staticmethod
    def flush(flush_data: Dict[str, Any]) -> None:
        """Execute flush logic in a separate process.

        This method runs in a subprocess, so it must:
        - Re-import all dependencies
        - Use only the data provided in flush_data
        - Load LangfuseClient from ResourceHub

        Args:
            flush_data: Dictionary containing all data needed for flushing
        """
        from hush.core.loggings import LOGGER

        try:
            from hush.core.registry import get_hub

            tracer_config = flush_data["tracer_config"]
            resource_key = tracer_config["resource_key"]

            # Get LangfuseClient from ResourceHub
            client = get_hub().langfuse(resource_key)

            workflow_name = flush_data["workflow_name"]
            req_id = flush_data["request_id"]
            user_id = flush_data.get("user_id")
            session_id = flush_data.get("session_id")
            execution_order = flush_data["execution_order"]
            nodes_trace_data = flush_data["nodes_trace_data"]

            LOGGER.info(
                "Workflow: %s, Request ID: %s, Creating Langfuse trace hierarchy...",
                workflow_name,
                req_id,
            )

            # Track created Langfuse objects for parent-child linking
            langfuse_objects = {}
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

                # Resolve media attachments to LangfuseMedia objects
                media_attachments = trace_data.pop("media_attachments", None)
                trace_data = LangfuseTracer._resolve_media(trace_data, media_attachments)

                # Build unique key for context-aware nodes
                if context_id:
                    node_id = f"{node_id}:{context_id}"

                # Check for context-aware parent
                if parent_id and context_id:
                    context_parent_id = f"{parent_id}:{context_id}"
                    if context_parent_id in langfuse_objects:
                        parent_id = context_parent_id

                if parent_id is None:
                    # Root node - create trace
                    root_trace = client.trace(
                        id=req_id,
                        name=workflow_name,
                        user_id=user_id,
                        session_id=session_id,
                        input=trace_data.get("input"),
                        output=trace_data.get("output"),
                        metadata=trace_data.get("metadata"),
                    )
                    langfuse_objects[node_id] = root_trace
                else:
                    # Child node - create span or generation
                    parent = langfuse_objects.get(parent_id)
                    if parent is None:
                        LOGGER.warning(
                            "Parent '%s' not found for node '%s'",
                            parent_id,
                            node_id,
                        )
                        continue

                    if contain_generation:
                        langfuse_objects[node_id] = parent.generation(**trace_data)
                    else:
                        langfuse_objects[node_id] = parent.span(**trace_data)

            # Ensure all traces are sent to Langfuse
            client.flush()

            # Generate and log trace URL
            trace_url = None
            if root_trace:
                trace_url = root_trace.get_trace_url()

            if trace_url:
                LOGGER.info(
                    "Workflow: %s, Request ID: %s, Langfuse trace created. View: %s",
                    workflow_name,
                    req_id,
                    trace_url,
                )
            else:
                LOGGER.warning(
                    "Workflow: %s, Request ID: %s, Failed to generate Langfuse trace URL.",
                    workflow_name,
                    req_id,
                )

        except ImportError as e:
            from hush.core.loggings import LOGGER

            LOGGER.error(
                "langfuse package is required for LangfuseTracer. "
                "Install it with: pip install langfuse. Error: %s",
                str(e),
            )

        except Exception as e:
            import traceback

            from hush.core.loggings import LOGGER

            LOGGER.error(
                "Langfuse flush failed: %s\nTraceback:\n%s",
                str(e),
                traceback.format_exc(),
            )

    def __repr__(self) -> str:
        """String representation."""
        return f"<LangfuseTracer resource_key={self.resource_key}>"
