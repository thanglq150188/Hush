"""Centralized resource registry with plugin-based extensibility."""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional, ClassVar, Any, List, Type

from .plugin import ResourcePlugin, ResourceConfig
from .storage import ConfigStorage, FileConfigStorage, InMemoryConfigStorage


# Global hub instance - will be lazily initialized
_GLOBAL_HUB: Optional['ResourceHub'] = None


def _get_global_hub() -> Optional['ResourceHub']:
    """Get or create the global ResourceHub instance.

    This function implements lazy initialization of the global hub.
    It tries to load configuration from:
    1. HUSH_CONFIG environment variable
    2. ./resources.yaml (current directory)
    3. ~/.hush/resources.yaml (user home)
    4. Falls back to in-memory storage

    Returns:
        The global ResourceHub instance, or None if initialization fails
    """
    global _GLOBAL_HUB

    if _GLOBAL_HUB is None:
        try:
            # Try to find config file
            config_path = None

            # 1. Check environment variable
            env_config = os.getenv('HUSH_CONFIG')
            if env_config and Path(env_config).exists():
                config_path = Path(env_config)

            # 2. Check current directory
            elif Path('resources.yaml').exists():
                config_path = Path('resources.yaml')

            # 3. Check user home directory
            elif (Path.home() / '.hush' / 'resources.yaml').exists():
                config_path = Path.home() / '.hush' / 'resources.yaml'

            # Create hub from file or in-memory
            if config_path:
                _GLOBAL_HUB = ResourceHub.from_yaml(config_path)
            else:
                _GLOBAL_HUB = ResourceHub.from_memory()

        except Exception:
            # If anything fails, create in-memory hub
            _GLOBAL_HUB = ResourceHub.from_memory()

    return _GLOBAL_HUB


def get_hub() -> 'ResourceHub':
    """Get the global ResourceHub instance.

    This is the primary way to access the global hub.

    Returns:
        The global ResourceHub instance

    Example:
        from hush.core.registry import get_hub

        hub = get_hub()
        llm = hub.llm("gpt-4")
    """
    hub = _get_global_hub()
    if hub is None:
        raise RuntimeError("Failed to initialize global ResourceHub")
    return hub


def set_global_hub(hub: 'ResourceHub'):
    """Set a custom global ResourceHub instance.

    Use this to override the default global hub with a custom one.

    Args:
        hub: ResourceHub instance to use as global

    Example:
        from hush.core.registry import ResourceHub, set_global_hub

        custom_hub = ResourceHub.from_yaml("my_config.yaml")
        set_global_hub(custom_hub)
    """
    global _GLOBAL_HUB
    _GLOBAL_HUB = hub


class ResourceHub:
    """Centralized registry for managing application resources with plugin architecture.

    The ResourceHub provides:
    - Unified access to various resources (LLMs, databases, caches, etc.)
    - Plugin-based extensibility for different resource types
    - Multiple storage backends (file, memory, or custom)
    - Singleton pattern support

    Example:
        # Basic usage
        hub = ResourceHub.from_yaml("configs/resources.yaml")
        llm = hub.get("llm:gpt-4")

        # Register plugins from other packages
        from hush.providers import LLMPlugin, EmbeddingPlugin
        hub.register_plugin(LLMPlugin)
        hub.register_plugin(EmbeddingPlugin)

        # Access with type-specific methods
        llm = hub.llm("gpt-4")
        embedding = hub.embedding("bge-m3")
    """

    _instance: ClassVar[Optional['ResourceHub']] = None

    def __init__(self, storage: ConfigStorage):
        """Initialize hub with storage backend.

        Args:
            storage: Storage backend for persisting configurations
        """
        self._storage = storage
        self._instances: Dict[str, Any] = {}
        self._configs: Dict[str, ResourceConfig] = {}
        self._plugins: Dict[str, Type[ResourcePlugin]] = {}
        self._class_to_plugin: Dict[str, Type[ResourcePlugin]] = {}

        # Load existing configs
        self._load_all_configs()

    # ========================================================================
    # Factory Methods
    # ========================================================================

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'ResourceHub':
        """Create hub with YAML file storage.

        Args:
            path: Path to YAML configuration file

        Returns:
            ResourceHub instance with YAML storage
        """
        storage = FileConfigStorage(Path(path), format='yaml')
        return cls(storage)

    @classmethod
    def from_json(cls, path: str | Path) -> 'ResourceHub':
        """Create hub with JSON file storage.

        Args:
            path: Path to JSON configuration file

        Returns:
            ResourceHub instance with JSON storage
        """
        storage = FileConfigStorage(Path(path), format='json')
        return cls(storage)

    @classmethod
    def from_memory(cls) -> 'ResourceHub':
        """Create hub with in-memory storage (for testing).

        Returns:
            ResourceHub instance with in-memory storage
        """
        storage = InMemoryConfigStorage()
        return cls(storage)

    @classmethod
    def instance(cls) -> 'ResourceHub':
        """Get the singleton instance.

        Note: You must call set_instance() first to configure the singleton.

        Returns:
            The global ResourceHub singleton

        Raises:
            RuntimeError: If singleton not initialized
        """
        if cls._instance is None:
            raise RuntimeError(
                "ResourceHub singleton not initialized. "
                "Call ResourceHub.set_instance() first."
            )
        return cls._instance

    @classmethod
    def set_instance(cls, hub: 'ResourceHub'):
        """Set the global singleton instance.

        Args:
            hub: ResourceHub instance to use as singleton
        """
        cls._instance = hub

    # ========================================================================
    # Plugin Management
    # ========================================================================

    def register_plugin(self, plugin: Type[ResourcePlugin]):
        """Register a resource plugin.

        Args:
            plugin: Plugin class to register

        Example:
            from hush.providers import LLMPlugin
            hub.register_plugin(LLMPlugin)
        """
        resource_type = plugin.resource_type()
        config_class = plugin.config_class()

        self._plugins[resource_type] = plugin

        # Register the config class and all its subclasses
        # This allows plugins to handle config hierarchies (e.g., OpenAIConfig extends LLMConfig)
        def register_config_class(cls):
            self._class_to_plugin[cls.__name__] = plugin
            # Recursively register subclasses
            for subclass in cls.__subclasses__():
                register_config_class(subclass)

        register_config_class(config_class)

        # Reload configs to instantiate resources for this plugin
        self._load_all_configs()

    def register_plugins(self, *plugins: Type[ResourcePlugin]):
        """Register multiple plugins at once.

        Args:
            *plugins: Plugin classes to register

        Example:
            from hush.providers import LLMPlugin, EmbeddingPlugin, RerankPlugin
            hub.register_plugins(LLMPlugin, EmbeddingPlugin, RerankPlugin)
        """
        for plugin in plugins:
            self.register_plugin(plugin)

    def has_plugin(self, resource_type: str) -> bool:
        """Check if a plugin is registered for a resource type.

        Args:
            resource_type: Resource type identifier

        Returns:
            True if plugin is registered
        """
        return resource_type in self._plugins

    # ========================================================================
    # Config & Instance Management
    # ========================================================================

    def _load_all_configs(self):
        """Load configurations from storage and instantiate resources."""
        raw_configs = self._storage.load_all()

        for key, config_dict in raw_configs.items():
            if not isinstance(config_dict, dict):
                continue

            config_class_name = config_dict.get('_class')
            if not config_class_name:
                continue

            # Check if we have a plugin for this config type
            plugin = self._class_to_plugin.get(config_class_name)
            if not plugin:
                # No plugin registered yet, skip for now
                continue

            try:
                # Parse config (remove _class field)
                config_data = {k: v for k, v in config_dict.items() if k != '_class'}
                config_class = plugin.config_class()
                config = config_class.model_validate(config_data)

                # Store config and create instance
                self._configs[key] = config
                self._instances[key] = plugin.create(config)

            except Exception as e:
                # Log error but continue loading other configs
                print(f"Warning: Failed to load config '{key}': {e}")

    def keys(self) -> List[str]:
        """Return all registered resource keys.

        Returns:
            List of registry keys
        """
        return list(self._instances.keys())

    def has(self, key: str) -> bool:
        """Check if resource exists in registry.

        Args:
            key: Registry key

        Returns:
            True if resource exists
        """
        return key in self._instances

    def get(self, key: str) -> Any:
        """Retrieve resource instance by key.

        Args:
            key: Registry key for the resource

        Returns:
            The instantiated resource

        Raises:
            KeyError: If key not found
        """
        if key not in self._instances:
            raise KeyError(f"Resource '{key}' not found in registry")
        return self._instances[key]

    def get_config(self, key: str) -> ResourceConfig:
        """Get the config object for a resource.

        Args:
            key: Registry key

        Returns:
            The resource configuration

        Raises:
            KeyError: If key not found
        """
        if key not in self._configs:
            raise KeyError(f"Config '{key}' not found in registry")
        return self._configs[key]

    def register(
        self,
        config: ResourceConfig,
        registry_key: Optional[str] = None,
        persist: bool = True
    ) -> str:
        """Register a new resource configuration.

        Args:
            config: Resource configuration object
            registry_key: Optional custom key (auto-generated if not provided)
            persist: Whether to save to storage backend

        Returns:
            The registry key used

        Raises:
            ValueError: If no plugin registered for this config type
        """
        config_class_name = type(config).__name__

        # Find plugin for this config
        plugin = self._class_to_plugin.get(config_class_name)
        if not plugin:
            raise ValueError(
                f"No plugin registered for config type: {config_class_name}. "
                f"Register a plugin using hub.register_plugin(YourPlugin)"
            )

        # Generate key if not provided
        if not registry_key:
            registry_key = plugin.generate_key(config)

        # Store config and create instance
        self._configs[registry_key] = config
        self._instances[registry_key] = plugin.create(config)

        # Persist to storage
        if persist:
            config_dict = json.loads(config.model_dump_json(exclude_none=True))
            config_dict['_class'] = config_class_name
            self._storage.save(registry_key, config_dict)

        return registry_key

    def remove(self, key: str, persist: bool = True) -> bool:
        """Remove a resource from registry.

        Args:
            key: Registry key to remove
            persist: Whether to remove from storage backend

        Returns:
            True if removed, False if not found
        """
        if key not in self._instances:
            return False

        del self._instances[key]
        del self._configs[key]

        if persist:
            self._storage.remove(key)

        return True

    def clear(self, persist: bool = True):
        """Remove all resources from registry.

        Args:
            persist: Whether to clear storage backend as well
        """
        keys = list(self._instances.keys())
        self._instances.clear()
        self._configs.clear()

        if persist:
            for key in keys:
                self._storage.remove(key)

    def close(self):
        """Close storage connections and cleanup."""
        if self._storage:
            self._storage.close()

    # ========================================================================
    # Convenience Methods (Can be extended by subclasses or composition)
    # ========================================================================

    def _get_with_prefix(self, key: str, prefix: str) -> Any:
        """Helper to get resource with automatic prefix handling."""
        if not key.startswith(f"{prefix}:"):
            key = f"{prefix}:{key}"
        return self.get(key)

    def llm(self, key: str) -> Any:
        """Get LLM instance by key.

        Automatically adds 'llm:' prefix if not present.

        Args:
            key: LLM identifier (e.g., 'gpt-4' or 'llm:gpt-4')

        Returns:
            LLM instance
        """
        # Handle provider prefixes (azure:, openai:, gemini:)
        if ':' in key and not key.startswith('llm:'):
            key = f"llm:{key}"
        return self._get_with_prefix(key, 'llm')

    def embedding(self, key: str) -> Any:
        """Get embedding model by key."""
        return self._get_with_prefix(key, 'embedding')

    def reranker(self, key: str) -> Any:
        """Get reranker instance by key."""
        return self._get_with_prefix(key, 'reranking')

    def redis(self, key: str) -> Any:
        """Get Redis client by key."""
        return self._get_with_prefix(key, 'redis')

    def mongo(self, key: str) -> Any:
        """Get MongoDB client by key."""
        return self._get_with_prefix(key, 'mongo')

    def milvus(self, key: str) -> Any:
        """Get Milvus client by key."""
        return self._get_with_prefix(key, 'milvus')

    def s3(self, key: str) -> Any:
        """Get S3 client by key."""
        return self._get_with_prefix(key, 's3')

    def langfuse(self, key: str) -> Any:
        """Get Langfuse client by key."""
        return self._get_with_prefix(key, 'langfuse')

    def mcp(self, key: str) -> Any:
        """Get MCP server by key."""
        return self._get_with_prefix(key, 'mcp')
