"""LLM resource plugin for ResourceHub.

Auto-registers LLM config classes and factory handlers with hush-core.
"""

from hush.core.registry import (
    register_config_class,
    register_config_classes,
    register_factory_handler,
)
from hush.providers.llms.config import LLMConfig, OpenAIConfig, AzureConfig, GeminiConfig
from hush.providers.llms.factory import LLMFactory


class LLMPlugin:
    """Plugin for auto-registering LLM resources with ResourceHub.

    Call LLMPlugin.register() to register all LLM config classes and factory handlers.

    Example:
        ```python
        from hush.providers.registry import LLMPlugin

        # Register once at startup
        LLMPlugin.register()

        # Now ResourceHub can create LLM instances from configs
        from hush.core.registry import get_hub
        hub = get_hub()
        llm = hub.llm("gpt-4")
        ```
    """

    _registered = False

    @classmethod
    def register(cls):
        """Register all LLM config classes and factory handler."""
        if cls._registered:
            return

        # Register all config classes for deserialization
        register_config_classes(
            LLMConfig,
            OpenAIConfig,
            AzureConfig,
            GeminiConfig,
        )

        # Register factory handler for creating instances
        register_factory_handler(LLMConfig, LLMFactory.create)
        register_factory_handler(OpenAIConfig, LLMFactory.create)
        register_factory_handler(AzureConfig, LLMFactory.create)
        register_factory_handler(GeminiConfig, LLMFactory.create)

        cls._registered = True

    @classmethod
    def is_registered(cls) -> bool:
        """Check if plugin has been registered."""
        return cls._registered


# Auto-register on import
LLMPlugin.register()
