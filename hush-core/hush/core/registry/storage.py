"""Storage backends for resource configurations."""

import json
import yaml
from pathlib import Path
from typing import Dict, Optional
from abc import ABC, abstractmethod

from .plugin import ResourceConfig


class ConfigStorage(ABC):
    """Storage backend interface for persisting resource configurations."""

    @abstractmethod
    def load_all(self) -> Dict[str, dict]:
        """Load all stored configurations and return as dict of key -> raw dict.

        Returns:
            Dictionary mapping registry keys to raw config dicts (with _class field)
        """
        pass

    @abstractmethod
    def save(self, key: str, config_dict: dict) -> bool:
        """Persist a configuration. Returns True on success.

        Args:
            key: Registry key
            config_dict: Raw config dictionary (with _class field)

        Returns:
            True if saved successfully
        """
        pass

    @abstractmethod
    def remove(self, key: str) -> bool:
        """Delete a configuration. Returns True if removed.

        Args:
            key: Registry key to remove

        Returns:
            True if removed successfully
        """
        pass

    @abstractmethod
    def close(self):
        """Close connections and cleanup resources."""
        pass


class FileConfigStorage(ConfigStorage):
    """File-based storage for resource configurations (YAML or JSON)."""

    def __init__(self, file_path: Path | str, format: str = 'yaml'):
        """Initialize file storage.

        Args:
            file_path: Path to config file
            format: File format ('yaml' or 'json')
        """
        self._file_path = Path(file_path)
        self._format = format.lower()

        if self._format not in ['yaml', 'json']:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")

        self._file_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_file(self) -> Dict:
        """Read file based on format."""
        if not self._file_path.exists():
            return {}

        with open(self._file_path, 'r') as f:
            if self._format == 'yaml':
                return yaml.safe_load(f) or {}
            else:
                return json.load(f)

    def _save_file(self, data: Dict):
        """Write file based on format."""
        with open(self._file_path, 'w') as f:
            if self._format == 'yaml':
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(data, f, indent=2)

    def load_all(self) -> Dict[str, dict]:
        """Load all configurations from file."""
        try:
            return self._load_file()
        except (yaml.YAMLError, json.JSONDecodeError) as e:
            raise ValueError(f"Invalid {self._format.upper()} file: {e}")

    def save(self, key: str, config_dict: dict) -> bool:
        """Save configuration to file."""
        try:
            data = self._load_file()
            data[key] = config_dict
            self._save_file(data)
            return True
        except Exception:
            return False

    def remove(self, key: str) -> bool:
        """Remove configuration from file."""
        try:
            data = self._load_file()
            if key in data:
                del data[key]
                self._save_file(data)
                return True
            return False
        except Exception:
            return False

    def close(self):
        """No-op for file storage."""
        pass


class InMemoryConfigStorage(ConfigStorage):
    """In-memory storage for testing and development."""

    def __init__(self):
        self._data: Dict[str, dict] = {}

    def load_all(self) -> Dict[str, dict]:
        return self._data.copy()

    def save(self, key: str, config_dict: dict) -> bool:
        self._data[key] = config_dict
        return True

    def remove(self, key: str) -> bool:
        if key in self._data:
            del self._data[key]
            return True
        return False

    def close(self):
        pass
