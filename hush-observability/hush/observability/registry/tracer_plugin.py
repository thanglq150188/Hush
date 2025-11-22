"""Tracer plugins for ResourceHub integration."""

from typing import Any, Type
from hush.core.registry import ResourcePlugin, ResourceConfig

from ..config.langfuse import LangfuseConfig
from ..config.phoenix import PhoenixConfig
from ..config.opik import OpikConfig
from ..config.langsmith import LangSmithConfig


class LangfusePlugin(ResourcePlugin):
    """Plugin for Langfuse tracer."""

    @classmethod
    def config_class(cls) -> Type[ResourceConfig]:
        return LangfuseConfig

    @classmethod
    def resource_type(cls) -> str:
        return "tracer"

    @classmethod
    def create(cls, config: ResourceConfig) -> Any:
        from ..tracers.langfuse_tracer import LangfuseTracer
        if not isinstance(config, LangfuseConfig):
            raise ValueError(f"Expected LangfuseConfig, got {type(config).__name__}")
        return LangfuseTracer(config)

    @classmethod
    def generate_key(cls, config: ResourceConfig) -> str:
        # Keep the key as "langfuse:..." to match YAML
        # Extract custom name from somewhere or use default
        return f"langfuse:{config.host.split('.')[ 0].split('//')[-1]}"


class PhoenixPlugin(ResourcePlugin):
    """Plugin for Phoenix tracer."""

    @classmethod
    def config_class(cls) -> Type[ResourceConfig]:
        return PhoenixConfig

    @classmethod
    def resource_type(cls) -> str:
        return "tracer"

    @classmethod
    def create(cls, config: ResourceConfig) -> Any:
        raise NotImplementedError("Phoenix tracer not yet implemented")

    @classmethod
    def generate_key(cls, config: ResourceConfig) -> str:
        return "phoenix:default"


class OpikPlugin(ResourcePlugin):
    """Plugin for Opik tracer."""

    @classmethod
    def config_class(cls) -> Type[ResourceConfig]:
        return OpikConfig

    @classmethod
    def resource_type(cls) -> str:
        return "tracer"

    @classmethod
    def create(cls, config: ResourceConfig) -> Any:
        raise NotImplementedError("Opik tracer not yet implemented")

    @classmethod
    def generate_key(cls, config: ResourceConfig) -> str:
        return "opik:default"


class LangSmithPlugin(ResourcePlugin):
    """Plugin for LangSmith tracer."""

    @classmethod
    def config_class(cls) -> Type[ResourceConfig]:
        return LangSmithConfig

    @classmethod
    def resource_type(cls) -> str:
        return "tracer"

    @classmethod
    def create(cls, config: ResourceConfig) -> Any:
        raise NotImplementedError("LangSmith tracer not yet implemented")

    @classmethod
    def generate_key(cls, config: ResourceConfig) -> str:
        return "langsmith:default"
