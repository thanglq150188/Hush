"""
Hush Observability Package

Backend-agnostic observability with support for multiple tracing frameworks.
"""

from .config import LangfuseConfig, LangSmithConfig, OpikConfig, PhoenixConfig, TracerConfig
from .models import BaseTraceInfo, MessageTraceInfo, WorkflowTraceInfo
from .registry import LangfusePlugin, LangSmithPlugin, OpikPlugin, PhoenixPlugin
from .tracers import AsyncTraceBuffer, BackendAdapter, BaseTracer, LangfuseTracer

__version__ = "0.1.0"

__all__ = [
    # Config
    "TracerConfig",
    "LangfuseConfig",
    "PhoenixConfig",
    "OpikConfig",
    "LangSmithConfig",
    # Models
    "BaseTraceInfo",
    "MessageTraceInfo",
    "WorkflowTraceInfo",
    # Tracers
    "BaseTracer",
    "LangfuseTracer",
    "AsyncTraceBuffer",
    "BackendAdapter",
    # Registry
    "LangfusePlugin",
    "PhoenixPlugin",
    "OpikPlugin",
    "LangSmithPlugin",
]
