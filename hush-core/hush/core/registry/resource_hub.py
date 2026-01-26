"""ResourceHub - centralized registry with lazy loading and pluggable storage."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, TYPE_CHECKING

from hush.core.loggings import LOGGER
from hush.core.utils.yaml_model import YamlModel

from .config_registry import REGISTRY
from .storage import ConfigStorage, YamlConfigStorage
from .shortcuts.health import HealthCheckResult

# Type hints for IDE support
if TYPE_CHECKING:
    from hush.providers.llms.base import BaseLLM
    from hush.providers.embeddings.base import BaseEmbedding
    from hush.providers.rerankers.base import BaseReranker


@dataclass
class CacheEntry:
    """Cache entry with config and instance (lazy loaded)."""
    config: YamlModel
    instance: Any = None


class ResourceHub:
    """Centralized registry for managing application resources.

    Features:
    - Lazy loading: resources are initialized on first access
    - Pluggable storage: YAML, JSON, or custom backend
    - Extensible: external packages register their configs and factories

    Example:
        # Basic usage
        hub = ResourceHub.from_yaml("configs/resources.yaml")
        llm = hub.llm("gpt-4")
        redis = hub.redis("default")

        # Or use global hub
        from hush.core.registry import get_hub
        llm = get_hub().llm("gpt-4")
    """

    _instance: ClassVar[Optional['ResourceHub']] = None

    def __init__(self, storage: ConfigStorage):
        """Initialize hub with storage backend.

        Args:
            storage: Storage backend for configs
        """
        self._storage = storage
        self._cache: Dict[str, CacheEntry] = {}

    # ========================================================================
    # Factory Methods
    # ========================================================================

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'ResourceHub':
        """Create hub with YAML file storage.

        Args:
            path: Path to YAML config file

        Returns:
            ResourceHub instance
        """
        storage = YamlConfigStorage(Path(path))
        return cls(storage)

    @classmethod
    def from_json(cls, path: str | Path) -> 'ResourceHub':
        """Create hub with JSON file storage.

        Args:
            path: Path to JSON config file

        Returns:
            ResourceHub instance
        """
        from .storage import JsonConfigStorage
        storage = JsonConfigStorage(Path(path))
        return cls(storage)

    @classmethod
    def instance(cls) -> 'ResourceHub':
        """Get singleton instance.

        Returns:
            Global ResourceHub singleton

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
        """Set global singleton instance.

        Args:
            hub: ResourceHub instance to use as singleton
        """
        cls._instance = hub

    # ========================================================================
    # Load Config (Lazy)
    # ========================================================================

    def _load_config(self, key: str) -> Optional[YamlModel]:
        """Load a config from storage (lazy, on demand)."""
        if key in self._cache:
            return self._cache[key].config

        config_data = self._storage.load_one(key)
        if not config_data:
            return None

        # Extract category from key prefix: "llm:gpt-4" -> "llm"
        category = key.split(":")[0] if ":" in key else None

        # Prefer `type` (new), fallback `_class` (old/backward compatible)
        config_type = config_data.get('type') or config_data.get('_class')
        if not config_type:
            LOGGER.warning("Missing 'type' or '_class' field for key: %s", key)
            return None

        # Lookup config class with category namespace
        config_class = REGISTRY.get_class(config_type, category=category)
        if not config_class:
            LOGGER.warning("Unknown config type: %s (category=%s)", config_type, category)
            return None

        try:
            # Parse config (exclude type and _class fields)
            data = {k: v for k, v in config_data.items() if k not in ('type', '_class')}
            config = config_class.model_validate(data)
            self._cache[key] = CacheEntry(config=config)
            return config
        except Exception as e:
            LOGGER.error("Cannot parse config '%s': %s", key, e)
            return None

    def _hash_of(self, config: YamlModel) -> str:
        """Create MD5 hash of config for unique identification."""
        return hashlib.md5(config.model_dump_json().encode()).hexdigest()[:8]

    def _key_of(self, config: YamlModel) -> str:
        """Create registry key from config category and model/name/hash."""
        config_type = type(config)

        # Use _category if available, otherwise derive from class name
        if hasattr(config_type, '_category'):
            category = getattr(config_type, '_category')
        else:
            category = config_type.__name__.replace('Config', '').lower()

        # Resource based on model uses model name
        if hasattr(config, 'model') and config.model:
            return f"{category}:{config.model}"

        # Resource based on name
        if hasattr(config, 'name') and config.name:
            return f"{category}:{config.name}"

        # Fallback to hash
        return f"{category}:{self._hash_of(config)}"

    # ========================================================================
    # Public API
    # ========================================================================

    def keys(self) -> List[str]:
        """Return all registered keys (loads all configs from storage)."""
        all_configs = self._storage.load_all()

        for key, config_data in all_configs.items():
            if key not in self._cache:
                # Extract category from key prefix
                category = key.split(":")[0] if ":" in key else None

                # Prefer `type` (new), fallback `_class` (old)
                config_type = config_data.get('type') or config_data.get('_class')
                if config_type:
                    config_class = REGISTRY.get_class(config_type, category=category)
                    if config_class:
                        try:
                            data = {k: v for k, v in config_data.items() if k not in ('type', '_class')}
                            config = config_class.model_validate(data)
                            self._cache[key] = CacheEntry(config=config)
                        except Exception as e:
                            LOGGER.error("Cannot parse config '%s': %s", key, e)

        return list(self._cache.keys())

    def has(self, key: str) -> bool:
        """Check if resource exists in registry."""
        if key in self._cache:
            return True
        # Try loading from storage
        return self._load_config(key) is not None

    def get(self, key: str) -> Any:
        """Get resource instance by key (lazy load on first access).

        Args:
            key: Registry key of resource

        Returns:
            Initialized resource instance

        Raises:
            KeyError: If key not found
        """
        # Return cached instance if available
        if key in self._cache and self._cache[key].instance is not None:
            return self._cache[key].instance

        # Load config from storage
        config = self._load_config(key)
        if not config:
            raise KeyError(f"Resource '{key}' not found in registry")

        # Lazy initialize resource
        instance = REGISTRY.create(config)
        if instance is None:
            raise RuntimeError(f"Cannot create resource for '{key}'")

        self._cache[key].instance = instance
        LOGGER.debug("Lazy loaded resource: %s", key)

        return self._cache[key].instance

    def get_config(self, key: str) -> YamlModel:
        """Get config object of resource.

        Args:
            key: Registry key

        Returns:
            Resource config

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
        """Register new resource config.

        Args:
            config: Resource config object
            registry_key: Custom key (auto-generated if not provided)

        Returns:
            Registry key used
        """
        if not registry_key:
            registry_key = self._key_of(config)

        # Create instance immediately
        instance = REGISTRY.create(config)
        self._cache[registry_key] = CacheEntry(config=config, instance=instance)

        # Persist to storage
        config_dict = json.loads(config.model_dump_json(exclude_none=True))
        # Use 'type' field if config has _type attribute, otherwise use _class
        if hasattr(type(config), '_type'):
            config_dict['type'] = getattr(type(config), '_type')
        else:
            config_dict['_class'] = type(config).__name__
        self._storage.save(registry_key, config_dict)

        LOGGER.debug("Registered: %s", registry_key)
        return registry_key

    def remove(self, key: str) -> bool:
        """Remove resource from registry.

        Args:
            key: Registry key to remove

        Returns:
            True if removed, False if not found
        """
        if key not in self._cache:
            # Try loading first
            if not self._load_config(key):
                return False

        if key in self._cache:
            del self._cache[key]

        self._storage.remove(key)
        LOGGER.debug("Removed: %s", key)
        return True

    def clear(self):
        """Clear all resources from registry and storage."""
        keys = list(self._cache.keys())
        self._cache.clear()
        for key in keys:
            self._storage.remove(key)
        LOGGER.debug("Cleared all resources")

    def close(self):
        """Close storage connection and cleanup."""
        if self._storage:
            self._storage.close()

    # ========================================================================
    # Type-specific Accessors (with type hints for IDE support)
    # ========================================================================

    def _get_with_prefix(self, key: str, prefix: str) -> Any:
        """Helper to get resource with automatic prefix handling."""
        if not key.startswith(f"{prefix}:"):
            key = f"{prefix}:{key}"
        return self.get(key)

    def llm(self, key: str) -> "BaseLLM":
        """Get LLM instance by key.

        Automatically handles provider prefix (azure:, openai:, gemini:).

        Args:
            key: LLM identifier (e.g., 'gpt-4', 'azure:gpt-4', 'llm:gpt-4')

        Returns:
            BaseLLM instance with chat(), generate() methods
        """
        # Handle provider prefix
        for prefix in ['azure:', 'openai:', 'gemini:']:
            if key.startswith(prefix):
                return self._get_with_prefix(key, 'llm')
        return self._get_with_prefix(key, 'llm')

    def embedding(self, key: str) -> "BaseEmbedding":
        """Get embedding model by key.

        Returns:
            BaseEmbedding instance with embed(), embed_batch() methods
        """
        return self._get_with_prefix(key, 'embedding')

    def reranker(self, key: str) -> "BaseReranker":
        """Get reranker instance by key.

        Returns:
            BaseReranker instance with rerank() method
        """
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

    # ========================================================================
    # Health Check
    # ========================================================================

    def health_check(self, keys: Optional[List[str]] = None) -> HealthCheckResult:
        """Check health of all or specified resources.

        Args:
            keys: Optional list of keys to check. If None, checks all.

        Returns:
            HealthCheckResult with status of each resource

        Example:
            result = hub.health_check()
            if not result.healthy:
                print(f"Unhealthy resources: {result.failed}")
        """
        check_keys = keys if keys else self.keys()
        results: Dict[str, bool] = {}
        errors: Dict[str, str] = {}

        for key in check_keys:
            try:
                # Try to load the resource
                self.get(key)
                results[key] = True
            except Exception as e:
                results[key] = False
                errors[key] = str(e)
                LOGGER.warning("Health check failed for '%s': %s", key, e)

        return HealthCheckResult(
            results=results,
            errors=errors,
        )
