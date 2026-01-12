"""Langfuse tracer implementation using subprocess-based flushing.

This tracer inherits from hush.core.tracers.BaseTracer and implements
the Langfuse-specific flush logic that runs in a separate process.
"""
from typing import Dict, Any

from hush.core.tracers import BaseTracer, register_tracer
from hush.observability.langfuse.config import LangfuseConfig
from hush.observability.langfuse.media import resolve_media_for_langfuse


@register_tracer
class LangfuseTracer(BaseTracer):
    """Tracer that sends workflow traces to Langfuse.

    This tracer uses subprocess-based flushing to avoid blocking
    the main workflow execution. Traces are collected during workflow
    execution and then flushed to Langfuse in a background process.

    Example:
        ```python
        from hush.observability.langfuse import LangfuseTracer, LangfuseConfig

        tracer = LangfuseTracer(
            config=LangfuseConfig(
                public_key="pk-...",
                secret_key="sk-..."
            )
        )

        # Use with workflow engine
        workflow = MyWorkflow(tracer=tracer)
        result = await workflow.run(inputs={...})
        # Traces are automatically flushed in background
        ```
    """

    def __init__(self, config: LangfuseConfig):
        """Initialize the Langfuse tracer.

        Args:
            config: Langfuse configuration with API keys and host
        """
        self.config = config

    def _get_tracer_config(self) -> Dict[str, Any]:
        """Return Langfuse-specific configuration for serialization.

        This config is passed to the subprocess flush function.
        """
        return {
            "public_key": self.config.public_key,
            "secret_key": self.config.secret_key,
            "host": self.config.host,
            "no_proxy": self.config.no_proxy,
        }

    @staticmethod
    def _resolve_media_attachments(trace_data: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve media attachments to LangfuseMedia objects.

        Args:
            trace_data: Trace data dict that may contain media_attachments

        Returns:
            Modified trace_data with LangfuseMedia objects injected
        """
        media_attachments = trace_data.pop("media_attachments", None)
        if not media_attachments:
            return trace_data

        return resolve_media_for_langfuse(trace_data, media_attachments)

    @staticmethod
    def flush(flush_data: Dict[str, Any]) -> None:
        """Execute flush logic in a separate process.

        This method runs in a subprocess, so it must:
        - Re-import all dependencies
        - Use only the data provided in flush_data
        - Not access any instance attributes

        Args:
            flush_data: Dictionary containing all data needed for flushing
        """
        import os
        from hush.core.loggings import LOGGER

        try:
            # Import langfuse here since we're in a subprocess
            from langfuse import Langfuse

            tracer_config = flush_data["tracer_config"]

            # Set proxy bypass if configured
            if tracer_config.get("no_proxy"):
                os.environ["NO_PROXY"] = tracer_config["no_proxy"]

            # Initialize Langfuse client
            langfuse_client = Langfuse(
                public_key=tracer_config["public_key"],
                secret_key=tracer_config["secret_key"],
                host=tracer_config["host"],
            )

            workflow_name = flush_data["workflow_name"]
            req_id = flush_data["request_id"]
            user_id = flush_data["user_id"]
            session_id = flush_data["session_id"]
            execution_order = flush_data["execution_order"]
            nodes_trace_data = flush_data["nodes_trace_data"]

            LOGGER.info(
                "Workflow: %s, Request ID: %s, Creating Langfuse trace hierarchy...",
                workflow_name, req_id
            )

            # Track created Langfuse objects for parent-child linking
            langfuse_objects = {}
            root_trace = None

            for execution in execution_order:
                node_id = execution["node"]
                parent_id = execution["parent"]
                context_id = execution["context_id"]
                contain_generation = execution["contain_generation"]

                # Get trace data for this node
                trace_data = nodes_trace_data.get(
                    f"{node_id}:{context_id}" if context_id else node_id
                )
                if trace_data is None:
                    continue

                # Resolve media attachments to LangfuseMedia objects
                trace_data = LangfuseTracer._resolve_media_attachments(trace_data)

                # Build unique key for context-aware nodes
                if context_id:
                    node_id = f"{node_id}:{context_id}"

                # Check for context-aware parent
                context_parent_id = f"{parent_id}:{context_id}"
                if context_parent_id in langfuse_objects:
                    parent_id = context_parent_id

                if parent_id is None:
                    # Root node - create trace
                    root_trace = langfuse_client.trace(
                        id=req_id,
                        name=workflow_name,
                        user_id=user_id,
                        session_id=session_id,
                        start_time=trace_data.get("start_time"),
                        end_time=trace_data.get("end_time"),
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
                            parent_id, node_id
                        )
                        continue

                    if contain_generation:
                        langfuse_objects[node_id] = parent.generation(**trace_data)
                    else:
                        langfuse_objects[node_id] = parent.span(**trace_data)

            # Ensure all traces are sent to Langfuse
            langfuse_client.flush()

            # Generate and log trace URL
            trace_url = None
            if root_trace:
                trace_url = root_trace.get_trace_url()

            if trace_url:
                LOGGER.info(
                    "Workflow: %s, Request ID: %s, Langfuse trace created. View: %s",
                    workflow_name, req_id, trace_url
                )
            else:
                LOGGER.warning(
                    "Workflow: %s, Request ID: %s, Failed to generate Langfuse trace URL.",
                    workflow_name, req_id
                )

        except ImportError:
            from hush.core.loggings import LOGGER
            LOGGER.error(
                "langfuse package is required for LangfuseTracer. "
                "Install it with: pip install langfuse"
            )

        except Exception as e:
            import traceback
            from hush.core.loggings import LOGGER
            LOGGER.error(
                "Langfuse flush failed: %s\nTraceback:\n%s",
                str(e),
                traceback.format_exc()
            )

    def __repr__(self) -> str:
        """String representation."""
        return f"<LangfuseTracer host={self.config.host}>"