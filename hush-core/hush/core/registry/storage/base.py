"""Abstract base class for configuration storage backends."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ConfigStorage(ABC):
    """Storage backend interface for persisting resource configurations.

    Extend this class to add new storage backends (MongoDB, Redis, etc.)

    Example:
        class MongoConfigStorage(ConfigStorage):
            def __init__(self, uri: str, database: str):
                self._client = MongoClient(uri)
                self._db = self._client[database]

            def load_one(self, key: str) -> Optional[Dict[str, Any]]:
                return self._db.configs.find_one({"_id": key})

            # ... implement other methods
    """

    @abstractmethod
    def load_one(self, key: str) -> Optional[Dict[str, Any]]:
        """Load a single configuration by key.

        Args:
            key: Registry key (e.g., 'llm:gpt-4', 'redis:default')

        Returns:
            Raw config dict with '_class' field, or None if not found
        """
        pass

    @abstractmethod
    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Load all stored configurations.

        Returns:
            Dictionary mapping registry keys to raw config dicts
        """
        pass

    @abstractmethod
    def save(self, key: str, config_dict: Dict[str, Any]) -> bool:
        """Persist a configuration.

        Args:
            key: Registry key
            config_dict: Raw config dictionary (must include '_class' field)

        Returns:
            True if saved successfully
        """
        pass

    @abstractmethod
    def remove(self, key: str) -> bool:
        """Delete a configuration.

        Args:
            key: Registry key to remove

        Returns:
            True if removed, False if not found
        """
        pass

    @abstractmethod
    def close(self):
        """Close connections and cleanup resources."""
        pass
