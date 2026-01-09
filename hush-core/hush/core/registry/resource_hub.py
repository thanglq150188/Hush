"""Registry resource tập trung với lazy loading và storage có thể thay thế."""

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


# Instance hub global - khởi tạo lazy
_GLOBAL_HUB: Optional['ResourceHub'] = None


def _get_global_hub() -> Optional['ResourceHub']:
    """Lấy hoặc tạo instance ResourceHub global.

    Thử load config từ:
    1. Biến môi trường HUSH_CONFIG
    2. ./resources.yaml (thư mục hiện tại)
    3. ~/.hush/resources.yaml (thư mục home)

    Returns:
        Instance ResourceHub global, hoặc None nếu khởi tạo thất bại
    """
    global _GLOBAL_HUB

    if _GLOBAL_HUB is None:
        try:
            config_path = None

            # 1. Kiểm tra biến môi trường
            env_config = os.getenv('HUSH_CONFIG')
            if env_config and Path(env_config).exists():
                config_path = Path(env_config)

            # 2. Kiểm tra thư mục hiện tại
            elif Path('resources.yaml').exists():
                config_path = Path('resources.yaml')

            # 3. Kiểm tra thư mục home
            elif (Path.home() / '.hush' / 'resources.yaml').exists():
                config_path = Path.home() / '.hush' / 'resources.yaml'

            # Tạo hub
            if config_path:
                _GLOBAL_HUB = ResourceHub.from_yaml(config_path)
            else:
                # Không tìm thấy file config, tạo với đường dẫn mặc định
                _GLOBAL_HUB = ResourceHub.from_yaml(
                    Path.home() / '.hush' / 'resources.yaml'
                )

        except Exception as e:
            logger.error(f"Không thể khởi tạo global hub: {e}")
            return None

    return _GLOBAL_HUB


def get_hub() -> 'ResourceHub':
    """Lấy instance ResourceHub global.

    Đây là cách chính để truy cập global hub.

    Returns:
        Instance ResourceHub global

    Raises:
        RuntimeError: Nếu khởi tạo hub thất bại

    Example:
        from hush.core.registry import get_hub

        hub = get_hub()
        llm = hub.llm("gpt-4")
    """
    hub = _get_global_hub()
    if hub is None:
        raise RuntimeError("Không thể khởi tạo ResourceHub global")
    return hub


def set_global_hub(hub: 'ResourceHub'):
    """Đặt instance ResourceHub global tùy chỉnh.

    Sử dụng để ghi đè global hub mặc định.

    Args:
        hub: Instance ResourceHub dùng làm global

    Example:
        from hush.core.registry import ResourceHub, set_global_hub

        custom_hub = ResourceHub.from_yaml("my_config.yaml")
        set_global_hub(custom_hub)
    """
    global _GLOBAL_HUB
    _GLOBAL_HUB = hub


class ResourceHub:
    """Registry tập trung để quản lý các resource của ứng dụng.

    Tính năng:
    - Lazy loading: resource được khởi tạo khi truy cập lần đầu
    - Storage có thể thay thế: YAML, JSON, hoặc backend tùy chỉnh
    - Mở rộng được: các package bên ngoài đăng ký config và factory của họ

    Example:
        # Sử dụng cơ bản
        hub = ResourceHub.from_yaml("configs/resources.yaml")
        llm = hub.llm("gpt-4")
        redis = hub.redis("default")

        # Hoặc sử dụng global hub
        from hush.core.registry import RESOURCE_HUB
        llm = RESOURCE_HUB.llm("gpt-4")
    """

    _instance: ClassVar[Optional['ResourceHub']] = None

    def __init__(self, storage: ConfigStorage):
        """Khởi tạo hub với storage backend.

        Args:
            storage: Storage backend để lưu trữ config
        """
        self._storage = storage
        self._instances: Dict[str, Any] = {}
        self._configs: Dict[str, YamlModel] = {}

    # ========================================================================
    # Các Factory Method
    # ========================================================================

    @classmethod
    def from_yaml(cls, path: str | Path) -> 'ResourceHub':
        """Tạo hub với storage file YAML.

        Args:
            path: Đường dẫn đến file config YAML

        Returns:
            Instance ResourceHub
        """
        storage = YamlConfigStorage(Path(path))
        return cls(storage)

    @classmethod
    def from_json(cls, path: str | Path) -> 'ResourceHub':
        """Tạo hub với storage file JSON.

        Args:
            path: Đường dẫn đến file config JSON

        Returns:
            Instance ResourceHub
        """
        from .storage import JsonConfigStorage
        storage = JsonConfigStorage(Path(path))
        return cls(storage)

    @classmethod
    def instance(cls) -> 'ResourceHub':
        """Lấy singleton instance.

        Returns:
            Singleton ResourceHub global

        Raises:
            RuntimeError: Nếu singleton chưa được khởi tạo
        """
        if cls._instance is None:
            raise RuntimeError(
                "Singleton ResourceHub chưa được khởi tạo. "
                "Gọi ResourceHub.set_instance() trước."
            )
        return cls._instance

    @classmethod
    def set_instance(cls, hub: 'ResourceHub'):
        """Đặt singleton instance global.

        Args:
            hub: Instance ResourceHub dùng làm singleton
        """
        cls._instance = hub

    # ========================================================================
    # Load Config (Lazy)
    # ========================================================================

    def _load_config(self, key: str) -> Optional[YamlModel]:
        """Load một config từ storage (lazy, theo yêu cầu)."""
        if key in self._configs:
            return self._configs[key]

        config_data = self._storage.load_one(key)
        if not config_data:
            return None

        config_class_name = config_data.get('_class')
        if not config_class_name:
            logger.warning(f"Thiếu field '_class' cho key: {key}")
            return None

        config_class = get_config_class(config_class_name)
        if not config_class:
            logger.warning(f"Config class không xác định: {config_class_name}")
            return None

        try:
            # Parse config (loại bỏ field _class)
            data = {k: v for k, v in config_data.items() if k != '_class'}
            config = config_class.model_validate(data)
            self._configs[key] = config
            return config
        except Exception as e:
            logger.error(f"Không thể parse config '{key}': {e}")
            return None

    def _hash_of(self, config: YamlModel) -> str:
        """Tạo MD5 hash của config để định danh duy nhất."""
        return hashlib.md5(config.model_dump_json().encode()).hexdigest()[:8]

    def _key_of(self, config: YamlModel) -> str:
        """Tạo registry key từ config type và model/hash."""
        type_name = type(config).__name__.replace('Config', '').lower()

        # Resource dựa trên model sử dụng tên model
        if hasattr(config, 'model') and config.model:
            return f"{type_name}:{config.model}"

        # Resource dựa trên name
        if hasattr(config, 'name') and config.name:
            return f"{type_name}:{config.name}"

        # Fallback về hash
        return f"{type_name}:{self._hash_of(config)}"

    # ========================================================================
    # API Public
    # ========================================================================

    def keys(self) -> List[str]:
        """Trả về tất cả registry key đã đăng ký (load tất cả config từ storage)."""
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
                            logger.error(f"Không thể parse config '{key}': {e}")

        return list(self._configs.keys())

    def has(self, key: str) -> bool:
        """Kiểm tra resource có tồn tại trong registry không."""
        if key in self._configs:
            return True
        # Thử load từ storage
        return self._load_config(key) is not None

    def get(self, key: str) -> Any:
        """Lấy resource instance theo key (lazy load khi truy cập lần đầu).

        Args:
            key: Registry key của resource

        Returns:
            Resource instance đã được khởi tạo

        Raises:
            KeyError: Nếu không tìm thấy key
        """
        # Trả về instance đã cache nếu có
        if key in self._instances:
            return self._instances[key]

        # Load config từ storage
        config = self._load_config(key)
        if not config:
            raise KeyError(f"Không tìm thấy resource '{key}' trong registry")

        # Lazy khởi tạo resource
        instance = ResourceFactory.create(config)
        if instance is None:
            raise RuntimeError(f"Không thể tạo resource cho '{key}'")

        self._instances[key] = instance
        logger.info(f"Đã lazy load resource: {key}")

        return self._instances[key]

    def get_config(self, key: str) -> YamlModel:
        """Lấy object config của resource.

        Args:
            key: Registry key

        Returns:
            Config của resource

        Raises:
            KeyError: Nếu không tìm thấy key
        """
        config = self._load_config(key)
        if not config:
            raise KeyError(f"Không tìm thấy config '{key}' trong registry")
        return config

    def register(
        self,
        config: YamlModel,
        registry_key: Optional[str] = None
    ) -> str:
        """Đăng ký config resource mới.

        Args:
            config: Object config của resource
            registry_key: Key tùy chỉnh (tự động tạo nếu không cung cấp)

        Returns:
            Registry key được sử dụng
        """
        if not registry_key:
            registry_key = self._key_of(config)

        # Lưu config và tạo instance
        self._configs[registry_key] = config
        self._instances[registry_key] = ResourceFactory.create(config)

        # Persist vào storage
        config_dict = json.loads(config.model_dump_json(exclude_none=True))
        config_dict['_class'] = type(config).__name__
        self._storage.save(registry_key, config_dict)

        logger.info(f"Đã đăng ký: {registry_key}")
        return registry_key

    def remove(self, key: str) -> bool:
        """Xóa resource khỏi registry.

        Args:
            key: Registry key cần xóa

        Returns:
            True nếu đã xóa, False nếu không tìm thấy
        """
        if key not in self._configs and key not in self._instances:
            # Thử load trước
            if not self._load_config(key):
                return False

        if key in self._instances:
            del self._instances[key]
        if key in self._configs:
            del self._configs[key]

        self._storage.remove(key)
        logger.info(f"Đã xóa: {key}")
        return True

    def clear(self):
        """Xóa tất cả resource khỏi registry và storage."""
        keys = list(self._configs.keys())
        self._instances.clear()
        self._configs.clear()
        for key in keys:
            self._storage.remove(key)
        logger.info("Đã xóa tất cả resource")

    def close(self):
        """Đóng kết nối storage và dọn dẹp."""
        if self._storage:
            self._storage.close()

    # ========================================================================
    # Các Accessor theo Type
    # ========================================================================

    def _get_with_prefix(self, key: str, prefix: str) -> Any:
        """Helper để lấy resource với xử lý prefix tự động."""
        if not key.startswith(f"{prefix}:"):
            key = f"{prefix}:{key}"
        return self.get(key)

    def llm(self, key: str) -> Any:
        """Lấy LLM instance theo key.

        Tự động xử lý provider prefix (azure:, openai:, gemini:).

        Args:
            key: Định danh LLM (ví dụ: 'gpt-4', 'azure:gpt-4', 'llm:gpt-4')

        Returns:
            LLM instance
        """
        # Xử lý provider prefix
        for prefix in ['azure:', 'openai:', 'gemini:']:
            if key.startswith(prefix):
                return self._get_with_prefix(key, 'llm')
        return self._get_with_prefix(key, 'llm')

    def embedding(self, key: str) -> Any:
        """Lấy embedding model theo key."""
        return self._get_with_prefix(key, 'embedding')

    def reranker(self, key: str) -> Any:
        """Lấy reranker instance theo key."""
        return self._get_with_prefix(key, 'reranking')

    def redis(self, key: str) -> Any:
        """Lấy Redis client theo key."""
        return self._get_with_prefix(key, 'redis')

    def mongo(self, key: str) -> Any:
        """Lấy async MongoDB client theo key."""
        return self._get_with_prefix(key, 'mongo')

    def milvus(self, key: str) -> Any:
        """Lấy Milvus client theo key."""
        return self._get_with_prefix(key, 'milvus')

    def s3(self, key: str) -> Any:
        """Lấy S3 client theo key."""
        return self._get_with_prefix(key, 's3')

    def langfuse(self, key: str) -> Any:
        """Lấy Langfuse client theo key."""
        return self._get_with_prefix(key, 'langfuse')

    def mcp(self, key: str) -> Any:
        """Lấy MCP server theo key."""
        return self._get_with_prefix(key, 'mcp')
