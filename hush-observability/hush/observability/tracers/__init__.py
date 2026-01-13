"""Tracers for various observability backends.

Each tracer extends hush.core.tracers.BaseTracer and uses
ResourceHub to get the backend client in the subprocess.

Available tracers:
- LangfuseTracer: Langfuse observability platform
- OpikTracer: Opik observability platform (by Comet)
- OTELTracer: OpenTelemetry (vendor-neutral, exports to Jaeger/Zipkin/Datadog/etc.)
- PhoenixTracer: Arize Phoenix (open-source LLM observability)
"""

from hush.observability.tracers.langfuse import LangfuseTracer
from hush.observability.tracers.opik import OpikTracer
from hush.observability.tracers.otel import OTELTracer
from hush.observability.tracers.phoenix import PhoenixTracer

__all__ = [
    "LangfuseTracer",
    "OpikTracer",
    "OTELTracer",
    "PhoenixTracer",
]
