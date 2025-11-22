"""Plugin system for extending the resource registry."""

from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Optional, Type, TypeVar, TYPE_CHECKING

from hush.core.utils.yaml_model import YamlModel

if TYPE_CHECKING:
    from .resource_hub import ResourceHub

# Base config type that all resource configs must inherit from
# This is an alias for YamlModel - all existing configs already inherit from YamlModel
ResourceConfig = YamlModel

T = TypeVar('T', bound=ResourceConfig)


class PluginMeta(ABCMeta):
    """Metaclass for automatic plugin registration.

    When a plugin class is created (not instantiated), it automatically
    registers itself with the global RESOURCE_HUB if available.
    """

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Only auto-register concrete plugin classes (not the base class)
        if name != 'ResourcePlugin' and not getattr(cls, '__abstractmethods__', None):
            # Import here to avoid circular imports
            from .resource_hub import _get_global_hub

            # Get or create the global hub
            hub = _get_global_hub()
            if hub is not None:
                try:
                    hub.register_plugin(cls)
                except Exception:
                    # Silently fail if registration fails (e.g., during module import)
                    pass

        return cls


class ResourcePlugin(ABC, metaclass=PluginMeta):
    """Plugin interface for resource factories.

    Each hush-* package can register plugins to handle their specific resource types.

    Example:
        # In hush-providers:
        class LLMPlugin(ResourcePlugin):
            @classmethod
            def config_class(cls) -> Type[ResourceConfig]:
                return LLMConfig

            @classmethod
            def create(cls, config: ResourceConfig) -> Any:
                return LLMFactory.create(config)

            @classmethod
            def resource_type(cls) -> str:
                return "llm"
    """

    @classmethod
    @abstractmethod
    def config_class(cls) -> Type[ResourceConfig]:
        """Return the config class this plugin handles.

        Returns:
            The Pydantic model class for this resource type
        """
        pass

    @classmethod
    @abstractmethod
    def create(cls, config: ResourceConfig) -> Any:
        """Create a resource instance from config.

        Args:
            config: The resource configuration

        Returns:
            The instantiated resource
        """
        pass

    @classmethod
    @abstractmethod
    def resource_type(cls) -> str:
        """Return the resource type identifier (e.g., 'llm', 'embedding', 'redis').

        This is used to generate registry keys like 'llm:gpt-4' or 'redis:default'.

        Returns:
            Resource type string
        """
        pass

    @classmethod
    def generate_key(cls, config: ResourceConfig) -> str:
        """Generate a registry key for this config.

        Override this method to customize key generation.
        Default uses: {resource_type}:{model_or_hash}

        Tries to use these fields in order:
        1. model
        2. name
        3. id
        4. hash of config

        Args:
            config: The resource configuration

        Returns:
            Registry key string
        """
        import hashlib

        type_name = cls.resource_type()

        # Try common identifier fields in order
        for field_name in ['model', 'name', 'id']:
            if hasattr(config, field_name):
                value = getattr(config, field_name)
                if value:
                    return f"{type_name}:{value}"

        # Fall back to hash
        config_hash = hashlib.md5(
            config.model_dump_json().encode()
        ).hexdigest()[:8]
        return f"{type_name}:{config_hash}"
