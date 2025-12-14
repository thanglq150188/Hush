"""Centralized resource registry with lazy loading and pluggable storage."""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from hush.core.utils.yaml_model import YamlModel

from .storage import ConfigStorage, YamlConfigStorage
from .resource_factory import ResourceFactory, get_config_class

logger = logging.getLogger(__name__)


# Global hub instance - lazily initialized
_GLOBAL_HUB: Optional['ResourceHub'] = None


def _get_global_hub() -> Optional['ResourceHub']:
    """Get or create the global ResourceHub instance.

    Tries to load configuration from:
    1. HUSH_CONFIG environment variable
    2. ./resources.yaml (current directory)
    3. ~/.hush/resources.yaml (user home)

    Returns:
        The global ResourceHub instance, or None if initialization fails
    """
    global _GLOBAL_HUB

    if _GLOBAL_HUB is None:
        try:
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

            # Create hub
            if config_path:
                _GLOBAL_HUB = ResourceHub.from_yaml(config_path)
            else:
                # No config file found, create with default path
                _GLOBAL_HUB = ResourceHub.from_yaml(
                    Path.home() / '.hush' / 'resources.yaml'
                )

        except Exception as e:
            logger.error(f"Failed to initialize global hub: {e}")
            return None

    return _GLOBAL_HUB


def get_hub() -> 'ResourceHub':
    """Get the global ResourceHub instance.

    This is the primary way to access the global hub.

    Returns:
        The global ResourceHub instance

    Raises:
        RuntimeError: If hub initialization fails

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

    Use this to override the default global hub.

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
    """Centralized registry for managing application resources.

    Features:
    - Lazy loading: resources are instantiated on first access
    - Pluggable storage: YAML, JSON, or custom backends
    - Extensible: external packages register their configs and factories

    Example:
        # Basic usage
        hub = ResourceHub.from_yaml("configs/resources.yaml")
        llm = hub.llm("gpt-4")
        redis = hub.redis("default")

        # Or use global hub
        from hush.core.registry import RESOURCE_HUB
        llm = RESOURCE_HUB.llm("gpt-4")
    """

    _instance: ClassVar[Optional['ResourceHub']] = None

    def __init__(self, storage: ConfigStorage):
        """Initialize hub with storage backend.

        Args:
            storage: Storage backend for persisting configurations
        """
        self._storage = storage
        self._instances: Dict[str, Any] = {}
        self._configs: Dict[str, YamlModel] = {}

    # ========================================================================
    # Factory Methods
    # ========================================================================

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'ResourceHub':
        """Create hub with YAML file storage.

        Args:
            path: Path to YAML configuration file

        Returns:
            ResourceHub instance
        """
        storage = YamlConfigStorage(Path(path))
        return cls(storage)

    @classmethod
    def from_json(cls, path: str | Path) -> 'ResourceHub':
        """Create hub with JSON file storage.

        Args:
            path: Path to JSON configuration file

        Returns:
            ResourceHub instance
        """
        from .storage import JsonConfigStorage
        storage = JsonConfigStorage(Path(path))
        return cls(storage)

    @classmethod
    def instance(cls) -> 'ResourceHub':
        """Get the singleton instance.

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
    # Config Loading (Lazy)
    # ========================================================================

    def _load_config(self, key: str) -> Optional[YamlModel]:
        """Load a single config from storage (lazy, on-demand)."""
        if key in self._configs:
            return self._configs[key]

        config_data = self._storage.load_one(key)
        if not config_data:
            return None

        config_class_name = config_data.get('_class')
        if not config_class_name:
            logger.warning(f"Missing '_class' field for key: {key}")
            return None

        config_class = get_config_class(config_class_name)
        if not config_class:
            logger.warning(f"Unknown config class: {config_class_name}")
            return None

        try:
            # Parse config (remove _class field)
            data = {k: v for k, v in config_data.items() if k != '_class'}
            config = config_class.model_validate(data)
            self._configs[key] = config
            return config
        except Exception as e:
            logger.error(f"Failed to parse config '{key}': {e}")
            return None

    def _hash_of(self, config: YamlModel) -> str:
        """Generate MD5 hash of config for unique identification."""
        return hashlib.md5(config.model_dump_json().encode()).hexdigest()[:8]

    def _key_of(self, config: YamlModel) -> str:
        """Generate registry key from config type and model/hash."""
        type_name = type(config).__name__.replace('Config', '').lower()

        # Model-based resources use model name
        if hasattr(config, 'model') and config.model:
            return f"{type_name}:{config.model}"

        # Name-based resources
        if hasattr(config, 'name') and config.name:
            return f"{type_name}:{config.name}"

        # Fall back to hash
        return f"{type_name}:{self._hash_of(config)}"

    # ========================================================================
    # Public API
    # ========================================================================

    def keys(self) -> List[str]:
        """Return all registered resource keys (loads all configs from storage)."""
        all_configs = self._storage.load_all()

        for key, config_data in all_configs.items():
            if key not in self._configs:
                config_class_name = config_data.get('_class')
                if config_class_name:
                    config_class = get_config_class(config_class_name)
                    if config_class:
                        try:
                            data = {k: v for k, v in config_data.items() if k != '_class'}
                            self._configs[key] = config_class.model_validate(data)
                        except Exception as e:
                            logger.error(f"Failed to parse config '{key}': {e}")

        return list(self._configs.keys())

    def has(self, key: str) -> bool:
        """Check if resource exists in registry."""
        if key in self._configs:
            return True
        # Try to load from storage
        return self._load_config(key) is not None

    def get(self, key: str) -> Any:
        """Retrieve resource instance by key (lazy loads on first access).

        Args:
            key: Registry key for the resource

        Returns:
            The instantiated resource

        Raises:
            KeyError: If key not found
        """
        # Return cached instance if exists
        if key in self._instances:
            return self._instances[key]

        # Load config from storage
        config = self._load_config(key)
        if not config:
            raise KeyError(f"Resource '{key}' not found in registry")

        # Lazy instantiate the resource
        instance = ResourceFactory.create(config)
        if instance is None:
            raise RuntimeError(f"Failed to create resource for '{key}'")

        self._instances[key] = instance
        logger.info(f"Lazy loaded resource: {key}")

        return self._instances[key]

    def get_config(self, key: str) -> YamlModel:
        """Get the config object for a resource.

        Args:
            key: Registry key

        Returns:
            The resource configuration

        Raises:
            KeyError: If key not found
        """
        config = self._load_config(key)
        if not config:
            raise KeyError(f"Config '{key}' not found in registry")
        return config

    def register(
        self,
        config: YamlModel,
        registry_key: Optional[str] = None
    ) -> str:
        """Register a new resource configuration.

        Args:
            config: Resource configuration object
            registry_key: Optional custom key (auto-generated if not provided)

        Returns:
            The registry key used
        """
        if not registry_key:
            registry_key = self._key_of(config)

        # Store config and create instance
        self._configs[registry_key] = config
        self._instances[registry_key] = ResourceFactory.create(config)

        # Persist to storage
        config_dict = json.loads(config.model_dump_json(exclude_none=True))
        config_dict['_class'] = type(config).__name__
        self._storage.save(registry_key, config_dict)

        logger.info(f"Registered: {registry_key}")
        return registry_key

    def remove(self, key: str) -> bool:
        """Remove a resource from registry.

        Args:
            key: Registry key to remove

        Returns:
            True if removed, False if not found
        """
        if key not in self._configs and key not in self._instances:
            # Try loading first
            if not self._load_config(key):
                return False

        if key in self._instances:
            del self._instances[key]
        if key in self._configs:
            del self._configs[key]

        self._storage.remove(key)
        logger.info(f"Removed: {key}")
        return True

    def clear(self):
        """Remove all resources from registry and storage."""
        keys = list(self._configs.keys())
        self._instances.clear()
        self._configs.clear()
        for key in keys:
            self._storage.remove(key)
        logger.info("Cleared all resources")

    def close(self):
        """Close storage connections and cleanup."""
        if self._storage:
            self._storage.close()

    # ========================================================================
    # Type-specific Accessors
    # ========================================================================

    def _get_with_prefix(self, key: str, prefix: str) -> Any:
        """Helper to get resource with automatic prefix handling."""
        if not key.startswith(f"{prefix}:"):
            key = f"{prefix}:{key}"
        return self.get(key)

    def llm(self, key: str) -> Any:
        """Get LLM instance by key.

        Automatically handles provider prefixes (azure:, openai:, gemini:).

        Args:
            key: LLM identifier (e.g., 'gpt-4', 'azure:gpt-4', 'llm:gpt-4')

        Returns:
            LLM instance
        """
        # Handle provider prefixes
        for prefix in ['azure:', 'openai:', 'gemini:']:
            if key.startswith(prefix):
                return self._get_with_prefix(key, 'llm')
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
        """Get async MongoDB client by key."""
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
