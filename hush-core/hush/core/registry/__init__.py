"""Registry resource mở rộng cho các package hush.

Module này cung cấp hệ thống quản lý resource tập trung có thể được
mở rộng bởi các package hush-* khác.

Sử dụng cơ bản:
    from hush.core.registry import RESOURCE_HUB

    llm = RESOURCE_HUB.llm("gpt-4")
    redis = RESOURCE_HUB.redis("default")

Mở rộng từ các package khác:
    from hush.core.registry import register_config_class, register_factory_handler
    from my_package.configs import MyConfig
    from my_package.factory import MyFactory

    # Đăng ký config class để deserialize
    register_config_class(MyConfig)

    # Đăng ký factory handler để khởi tạo instance
    register_factory_handler(MyConfig, MyFactory.create)
"""

from .resource_hub import (
    ResourceHub,
    get_hub,
    set_global_hub,
)
from .resource_factory import (
    ResourceFactory,
    register_config_class,
    register_config_classes,
    register_factory_handler,
    get_config_class,
    CLASS_NAME_MAP,
    FACTORY_HANDLERS,
)
from .storage import (
    ConfigStorage,
    YamlConfigStorage,
    JsonConfigStorage,
)

# RESOURCE_HUB global - khởi tạo lazy khi truy cập lần đầu
RESOURCE_HUB = get_hub()

__all__ = [
    # Hub chính
    "ResourceHub",
    "RESOURCE_HUB",
    "get_hub",
    "set_global_hub",
    # Factory và đăng ký
    "ResourceFactory",
    "register_config_class",
    "register_config_classes",
    "register_factory_handler",
    "get_config_class",
    "CLASS_NAME_MAP",
    "FACTORY_HANDLERS",
    # Các backend storage
    "ConfigStorage",
    "YamlConfigStorage",
    "JsonConfigStorage",
]
