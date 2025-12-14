"""JSON-based configuration storage."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ConfigStorage

logger = logging.getLogger(__name__)


class JsonConfigStorage(ConfigStorage):
    """JSON file storage for resource configurations.

    All configs are stored in a single JSON file.

    Example file structure:
        {
            "llm:gpt-4": {
                "_class": "OpenAIConfig",
                "model": "gpt-4",
                "api_key": "sk-xxx"
            },
            "redis:default": {
                "_class": "RedisConfig",
                "host": "localhost",
                "port": 6379
            }
        }
    """

    def __init__(self, file_path: Path | str):
        """Initialize JSON storage.

        Args:
            file_path: Path to JSON config file
        """
        self._file_path = Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_file(self) -> Dict[str, Any]:
        """Read and parse the JSON file."""
        if not self._file_path.exists():
            return {}

        with open(self._file_path, 'r') as f:
            return json.load(f)

    def _save_file(self, data: Dict[str, Any]):
        """Write data to the JSON file."""
        with open(self._file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_one(self, key: str) -> Optional[Dict[str, Any]]:
        """Load a single configuration by key."""
        try:
            data = self._load_file()
            config_data = data.get(key)

            if not config_data or not isinstance(config_data, dict):
                return None

            if '_class' not in config_data:
                logger.warning(f"Missing '_class' field for key: {key}")
                return None

            return config_data

        except Exception as e:
            logger.error(f"Failed to load config '{key}': {e}")
            return None

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """Load all configurations from file."""
        configs = {}

        try:
            data = self._load_file()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file: {e}")
            return configs

        for key, config_data in data.items():
            if not isinstance(config_data, dict):
                continue

            if '_class' not in config_data:
                logger.warning(f"Missing '_class' field for key: {key}")
                continue

            configs[key] = config_data

        return configs

    def save(self, key: str, config_dict: Dict[str, Any]) -> bool:
        """Save configuration to file."""
        try:
            data = self._load_file()
            data[key] = config_dict
            self._save_file(data)
            logger.info(f"Saved config: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config '{key}': {e}")
            return False

    def remove(self, key: str) -> bool:
        """Remove configuration from file."""
        try:
            data = self._load_file()
            if key in data:
                del data[key]
                self._save_file(data)
                logger.info(f"Removed config: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove config '{key}': {e}")
            return False

    def close(self):
        """No-op for file storage."""
        pass
