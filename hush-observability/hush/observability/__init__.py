"""
Hush Observability Package

Backend-agnostic observability with support for multiple tracing frameworks.

This package provides tracer implementations that work with hush-core's
BaseTracer interface. Tracers use subprocess-based flushing to avoid
blocking workflow execution.

Example:
    ```python
    from hush.observability.langfuse import LangfuseTracer, LangfuseConfig

    tracer = LangfuseTracer(
        config=LangfuseConfig(
            public_key="pk-...",
            secret_key="sk-..."
        )
    )

    # Use with workflow
    workflow = MyWorkflow(tracer=tracer)
    await workflow.run(inputs={...})
    ```
"""

# Import from hush.core for convenience
from hush.core.tracers import (
    BaseTracer,
    register_tracer,
    get_registered_tracers,
    MEDIA_KEY,
    MediaAttachment,
    serialize_media_attachments,
)

# Langfuse tracer (subprocess-based)
from hush.observability.langfuse import (
    LangfuseTracer,
    LangfuseConfig,
    resolve_media_for_langfuse,
)

# Legacy configs (for backward compatibility)
from .config import (
    TracerConfig,
    LangfuseConfig as LangfuseConfigLegacy,
    PhoenixConfig,
    OpikConfig,
    LangSmithConfig,
)

# Models
from .models import BaseTraceInfo, MessageTraceInfo, WorkflowTraceInfo

# Legacy tracers (buffer-based, for backward compatibility)
from .tracers import (
    AsyncTraceBuffer,
    BackendAdapter,
    BaseTracer as LegacyBaseTracer,
    LangfuseTracer as LangfuseTracerLegacy,
)

# Registry plugins - commented out until ResourcePlugin is available in hush-core
# from .registry import LangfusePlugin, LangSmithPlugin, OpikPlugin, PhoenixPlugin

__version__ = "0.1.0"

__all__ = [
    # Core tracer interface (from hush.core)
    "BaseTracer",
    "register_tracer",
    "get_registered_tracers",
    "MEDIA_KEY",
    "MediaAttachment",
    "serialize_media_attachments",
    # Langfuse tracer (recommended)
    "LangfuseTracer",
    "LangfuseConfig",
    "resolve_media_for_langfuse",
    # Legacy configs
    "TracerConfig",
    "PhoenixConfig",
    "OpikConfig",
    "LangSmithConfig",
    # Models
    "BaseTraceInfo",
    "MessageTraceInfo",
    "WorkflowTraceInfo",
    # Legacy tracers (buffer-based)
    "AsyncTraceBuffer",
    "BackendAdapter",
    "LegacyBaseTracer",
    "LangfuseTracerLegacy",
]
