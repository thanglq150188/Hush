"""Langfuse tracer implementation for hush workflows.

This module provides the LangfuseTracer that integrates with Langfuse
for workflow observability using the subprocess-based tracing approach.

Example:
    ```python
    from hush.observability.langfuse import LangfuseTracer, LangfuseConfig

    # Create tracer
    tracer = LangfuseTracer(
        config=LangfuseConfig(
            public_key="pk-...",
            secret_key="sk-...",
            host="https://cloud.langfuse.com"
        )
    )

    # Use with workflow
    workflow = MyWorkflow(tracer=tracer)
    await workflow.run(inputs={...})
    ```
"""
from hush.observability.langfuse.tracer import LangfuseTracer
from hush.observability.langfuse.config import LangfuseConfig
from hush.observability.langfuse.media import resolve_media_for_langfuse

__all__ = [
    "LangfuseTracer",
    "LangfuseConfig",
    "resolve_media_for_langfuse",
]