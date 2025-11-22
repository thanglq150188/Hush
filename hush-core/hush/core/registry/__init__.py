"""Resource registry with plugin-based extensibility.

This module provides a centralized resource management system that can be
extended by other hush-* packages through a plugin architecture.
"""

from .resource_hub import ResourceHub
from .plugin import ResourcePlugin, ResourceConfig
from .storage import ConfigStorage, FileConfigStorage

__all__ = [
    "ResourceHub",
    "ResourcePlugin",
    "ResourceConfig",
    "ConfigStorage",
    "FileConfigStorage",
]
