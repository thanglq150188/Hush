"""LLM resource plugin for ResourceHub."""

from typing import Any, Type

from hush.core.registry import ResourcePlugin, ResourceConfig
from hush.providers.llms.config import LLMConfig
from hush.providers.llms.factory import LLMFactory


class LLMPlugin(ResourcePlugin):
    """Plugin for LLM resources."""

    @classmethod
    def config_class(cls) -> Type[ResourceConfig]:
        """Return LLMConfig as the config class."""
        return LLMConfig

    @classmethod
    def create(cls, config: ResourceConfig) -> Any:
        """Create LLM instance from config."""
        if not isinstance(config, LLMConfig):
            raise ValueError(f"Expected LLMConfig, got {type(config)}")
        return LLMFactory.create(config)

    @classmethod
    def resource_type(cls) -> str:
        """Return 'llm' as the resource type."""
        return "llm"

    @classmethod
    def generate_key(cls, config: ResourceConfig) -> str:
        """Generate key like 'llm:openai:gpt-4' or 'llm:gpt-4'."""
        if not isinstance(config, LLMConfig):
            return super().generate_key(config)

        # Include provider type in key for clarity
        if hasattr(config, 'api_type') and config.api_type:
            # Extract the enum value if it's an enum
            api_type = config.api_type.value if hasattr(config.api_type, 'value') else config.api_type
            return f"llm:{api_type}:{config.model}"

        return f"llm:{config.model}"
