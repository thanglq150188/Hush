"""
Hush Observability Package

Backend-agnostic observability with support for multiple tracing frameworks.

This package provides:
- Backend clients (LangfuseClient, OpikClient, OTELClient, PhoenixClient) registered to ResourceHub
- Tracers (LangfuseTracer, OpikTracer, OTELTracer, PhoenixTracer) that use ResourceHub to get clients

Example:
    ```python
    from hush.observability import LangfuseTracer, OpikTracer, OTELTracer, PhoenixTracer

    # Langfuse tracer
    tracer = LangfuseTracer(resource_key="langfuse:vpbank")

    # Opik tracer
    tracer = OpikTracer(resource_key="opik:default")

    # OpenTelemetry tracer (exports to Jaeger, Zipkin, etc.)
    tracer = OTELTracer(resource_key="otel:jaeger")

    # Phoenix tracer (open-source LLM observability)
    tracer = PhoenixTracer(resource_key="phoenix:local")

    # Use with workflow
    workflow = MyWorkflow(tracer=tracer)
    await workflow.run(inputs={...})
    ```

    ```python
    # Direct client access
    from hush.core.registry import get_hub

    # Langfuse client
    langfuse = get_hub().langfuse("vpbank")
    prompt = langfuse.get_prompt("my-prompt")

    # Opik client
    opik = get_hub().opik("default")
    trace = opik.trace(name="my-trace")

    # OpenTelemetry client
    otel = get_hub().otel("jaeger")
    with otel.start_span("my-operation") as span:
        span.set_attribute("key", "value")

    # Phoenix client
    phoenix = get_hub().phoenix("local")
    tracer = phoenix.tracer
    ```
"""

# Auto-register backends to ResourceHub on import
from hush.observability.plugin import ObservabilityPlugin  # noqa: F401

# Backends (configs + clients)
from hush.observability.backends import (
    LangfuseConfig,
    LangfuseClient,
    OpikConfig,
    OpikClient,
    OTELConfig,
    OTELClient,
    PhoenixConfig,
    PhoenixClient,
)

# Tracers
from hush.observability.tracers import (
    LangfuseTracer,
    OpikTracer,
    OTELTracer,
    PhoenixTracer,
)

# Re-export core utilities for convenience
from hush.core.tracers import (
    BaseTracer,
    register_tracer,
    get_registered_tracers,
    MEDIA_KEY,
    MediaAttachment,
    serialize_media_attachments,
)

__version__ = "0.1.0"

__all__ = [
    # Backends - Configs
    "LangfuseConfig",
    "OpikConfig",
    "OTELConfig",
    "PhoenixConfig",
    # Backends - Clients
    "LangfuseClient",
    "OpikClient",
    "OTELClient",
    "PhoenixClient",
    # Tracers
    "LangfuseTracer",
    "OpikTracer",
    "OTELTracer",
    "PhoenixTracer",
    # Core utilities (re-exported)
    "BaseTracer",
    "register_tracer",
    "get_registered_tracers",
    "MEDIA_KEY",
    "MediaAttachment",
    "serialize_media_attachments",
]
