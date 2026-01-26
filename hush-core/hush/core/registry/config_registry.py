"""Unified config registry for all resource types."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type

from hush.core.loggings import LOGGER
from hush.core.utils.yaml_model import YamlModel


@dataclass
class ConfigEntry:
    """Entry containing config class and its factory."""
    config_class: Type[YamlModel]
    factory: Callable[[YamlModel], Any]


class ConfigRegistry:
    """Unified registry for all config types.

    Replaces the old CLASS_NAME_MAP, TYPE_ALIAS_MAP, and FACTORY_HANDLERS.

    Usage:
        # Register a config class with its factory
        REGISTRY.register(OpenAIConfig, LLMFactory.create)

        # Get config class by type and category
        config_class = REGISTRY.get_class("openai", category="llm")

        # Create instance from config
        instance = REGISTRY.create(config)
    """

    _instance: Optional['ConfigRegistry'] = None

    def __init__(self):
        # category -> type -> ConfigEntry
        self._entries: Dict[str, Dict[str, ConfigEntry]] = {}

    @classmethod
    def instance(cls) -> 'ConfigRegistry':
        """Get global singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (for testing)."""
        cls._instance = None

    def register(
        self,
        config_class: Type[YamlModel],
        factory: Callable[[YamlModel], Any]
    ) -> None:
        """Register config class with factory.

        Args:
            config_class: Config class with _type and _category attributes
            factory: Callable to create resource instance from config
        """
        category = getattr(config_class, '_category', '_default')
        type_name = getattr(config_class, '_type', config_class.__name__)

        entry = ConfigEntry(config_class=config_class, factory=factory)

        # Check duplicate in same category
        if category in self._entries:
            existing = self._entries[category].get(type_name)
            if existing and existing.config_class != config_class:
                raise ValueError(
                    f"Duplicate type '{type_name}' in category '{category}': "
                    f"{existing.config_class.__name__} vs {config_class.__name__}"
                )

        # Register by category + type
        self._entries.setdefault(category, {})[type_name] = entry

        # Also register by class name (for lookup by class name)
        self._entries.setdefault('_class', {})[config_class.__name__] = entry

        LOGGER.debug("Registered: %s:%s -> %s", category, type_name, config_class.__name__)

    def get_entry(
        self,
        type_name: str,
        category: Optional[str] = None
    ) -> Optional[ConfigEntry]:
        """Lookup ConfigEntry by type + category."""
        # 1. Try category namespace
        if category and category in self._entries:
            entry = self._entries[category].get(type_name)
            if entry:
                return entry

        # 2. Fallback to class name
        return self._entries.get('_class', {}).get(type_name)

    def get_class(
        self,
        type_name: str,
        category: Optional[str] = None
    ) -> Optional[Type[YamlModel]]:
        """Get config class by type + category."""
        entry = self.get_entry(type_name, category)
        return entry.config_class if entry else None

    def get_factory(
        self,
        config_class: Type[YamlModel]
    ) -> Optional[Callable]:
        """Get factory for a config class."""
        entry = self._entries.get('_class', {}).get(config_class.__name__)
        return entry.factory if entry else None

    def create(self, config: YamlModel) -> Any:
        """Create resource instance from config.

        Args:
            config: Config object to create instance from

        Returns:
            Resource instance

        Raises:
            ValueError: If no factory registered for config type
        """
        config_type = type(config)

        # Try exact match first
        entry = self._entries.get('_class', {}).get(config_type.__name__)

        # If not found, try parent classes (for inheritance)
        if not entry:
            for parent_type in config_type.__mro__[1:]:
                entry = self._entries.get('_class', {}).get(parent_type.__name__)
                if entry:
                    break

        if not entry:
            raise ValueError(f"No factory registered for {config_type.__name__}")

        return entry.factory(config)

    def clear(self):
        """Clear all registrations."""
        self._entries.clear()

    def categories(self) -> list[str]:
        """List all registered categories."""
        return [c for c in self._entries.keys() if c != '_class']

    def types_in_category(self, category: str) -> list[str]:
        """List all types in a category."""
        return list(self._entries.get(category, {}).keys())


# Global singleton
REGISTRY = ConfigRegistry.instance()
