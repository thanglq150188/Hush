"""Extensible resource registry for hush packages.

This module provides a centralized resource management system that can be
extended by other hush-* packages.

Basic usage:
    from hush.core.registry import RESOURCE_HUB

    llm = RESOURCE_HUB.llm("gpt-4")
    redis = RESOURCE_HUB.redis("default")

Extending from other packages:
    from hush.core.registry import register_config_class, register_factory_handler
    from my_package.configs import MyConfig
    from my_package.factory import MyFactory

    # Register config class for deserialization
    register_config_class(MyConfig)

    # Register factory handler for instantiation
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

# Global RESOURCE_HUB - lazily initialized on first access
RESOURCE_HUB = get_hub()

__all__ = [
    # Main hub
    "ResourceHub",
    "RESOURCE_HUB",
    "get_hub",
    "set_global_hub",
    # Factory and registration
    "ResourceFactory",
    "register_config_class",
    "register_config_classes",
    "register_factory_handler",
    "get_config_class",
    "CLASS_NAME_MAP",
    "FACTORY_HANDLERS",
    # Storage backends
    "ConfigStorage",
    "YamlConfigStorage",
    "JsonConfigStorage",
]
