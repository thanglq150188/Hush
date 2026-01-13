"""Arize Phoenix backend for hush observability.

Phoenix is an open-source AI observability platform that provides
tracing, evaluation, and debugging capabilities for LLM applications.

This module provides:
- PhoenixConfig: Configuration for ResourceHub
- PhoenixClient: Client for tracing

Deployment options:
- Local development (localhost:6006)
- Phoenix Cloud (app.phoenix.arize.com)
- Self-hosted deployment

References:
    - Documentation: https://arize.com/docs/phoenix
    - GitHub: https://github.com/Arize-ai/phoenix
"""

from hush.observability.backends.phoenix.config import PhoenixConfig
from hush.observability.backends.phoenix.client import PhoenixClient

__all__ = [
    "PhoenixConfig",
    "PhoenixClient",
]
