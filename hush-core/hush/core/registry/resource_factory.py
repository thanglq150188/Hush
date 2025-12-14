"""Extensible resource factory for creating instances from configurations."""

import logging
from typing import Any, Callable, Dict, Optional, Type

from hush.core.utils.yaml_model import YamlModel

logger = logging.getLogger(__name__)


# Registry of config classes: class_name -> class
CLASS_NAME_MAP: Dict[str, Type[YamlModel]] = {}

# Registry of factory handlers: config_class -> handler_function
FACTORY_HANDLERS: Dict[Type[YamlModel], Callable[[YamlModel], Any]] = {}


def register_config_class(cls: Type[YamlModel]):
    """Register a config class for deserialization.

    Call this from external packages to register their config classes.

    Args:
        cls: Config class (must inherit from YamlModel)

    Example:
        from hush.core.registry import register_config_class
        from my_package.configs import MyConfig

        register_config_class(MyConfig)
    """
    CLASS_NAME_MAP[cls.__name__] = cls
    logger.debug(f"Registered config class: {cls.__name__}")


def register_config_classes(*classes: Type[YamlModel]):
    """Register multiple config classes at once.

    Args:
        *classes: Config classes to register

    Example:
        register_config_classes(LLMConfig, EmbeddingConfig, RedisConfig)
    """
    for cls in classes:
        register_config_class(cls)


def register_factory_handler(
    config_class: Type[YamlModel],
    handler: Callable[[YamlModel], Any]
):
    """Register a factory handler for a config type.

    The handler is called to create resource instances from configs.

    Args:
        config_class: Config class this handler supports
        handler: Function that takes config and returns resource instance

    Example:
        from hush.core.registry import register_factory_handler
        from my_package.configs import LLMConfig
        from my_package.factory import LLMFactory

        register_factory_handler(LLMConfig, LLMFactory.create)
    """
    FACTORY_HANDLERS[config_class] = handler
    logger.debug(f"Registered factory handler for: {config_class.__name__}")


def get_config_class(class_name: str) -> Optional[Type[YamlModel]]:
    """Get a registered config class by name.

    Args:
        class_name: Name of the config class

    Returns:
        Config class or None if not found
    """
    return CLASS_NAME_MAP.get(class_name)


class ResourceFactory:
    """Factory for creating resource instances from configurations.

    Uses registered handlers to create instances. External packages register
    their handlers via register_factory_handler().

    Example:
        # In hush-providers package:
        register_factory_handler(LLMConfig, LLMFactory.create)
        register_factory_handler(EmbeddingConfig, EmbeddingFactory.create)

        # Then ResourceFactory can create any registered resource:
        config = OpenAIConfig(model="gpt-4", api_key="sk-xxx")
        llm = ResourceFactory.create(config)
    """

    @classmethod
    def create(cls, config: YamlModel) -> Optional[Any]:
        """Create a resource instance from configuration.

        Looks up the appropriate handler based on config type (including parent classes).

        Args:
            config: Resource configuration object

        Returns:
            Resource instance, or None if creation fails

        Raises:
            ValueError: If no handler registered for this config type
        """
        config_type = type(config)

        # Look for handler matching this config type or its parent classes
        handler = None
        for check_type in config_type.__mro__:
            if check_type in FACTORY_HANDLERS:
                handler = FACTORY_HANDLERS[check_type]
                break

        if not handler:
            raise ValueError(
                f"No factory handler registered for {config_type.__name__}. "
                f"Register one using register_factory_handler()."
            )

        try:
            return handler(config)
        except Exception as e:
            logger.error(f"Failed to create resource for {config_type.__name__}: {e}")
            return None
