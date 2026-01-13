"""Opik backend for hush observability.

Opik is an open-source LLM observability platform by Comet.

This module provides:
- OpikConfig: Configuration for ResourceHub
- OpikClient: Client for tracing and evaluation

References:
    - Documentation: https://www.comet.com/docs/opik/
    - GitHub: https://github.com/comet-ml/opik
"""

from hush.observability.backends.opik.config import OpikConfig
from hush.observability.backends.opik.client import OpikClient

__all__ = [
    "OpikConfig",
    "OpikClient",
]
