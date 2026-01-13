"""Observability backends for various providers.

Each backend provides:
- Config class (YamlModel): For ResourceHub registration
- Client class: For interacting with the backend

Available backends:
- langfuse: Langfuse observability platform
- opik: Opik observability platform (by Comet)
- otel: OpenTelemetry (vendor-neutral, exports to Jaeger/Zipkin/Datadog/etc.)
- phoenix: Arize Phoenix (open-source LLM observability)
"""

from hush.observability.backends.langfuse import LangfuseConfig, LangfuseClient
from hush.observability.backends.opik import OpikConfig, OpikClient
from hush.observability.backends.otel import OTELConfig, OTELClient
from hush.observability.backends.phoenix import PhoenixConfig, PhoenixClient

__all__ = [
    # Langfuse
    "LangfuseConfig",
    "LangfuseClient",
    # Opik
    "OpikConfig",
    "OpikClient",
    # OpenTelemetry
    "OTELConfig",
    "OTELClient",
    # Phoenix
    "PhoenixConfig",
    "PhoenixClient",
]
