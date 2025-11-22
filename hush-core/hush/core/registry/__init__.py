"""Resource registry with plugin-based extensibility.

This module provides a centralized resource management system that can be
extended by other hush-* packages through a plugin architecture.

The global RESOURCE_HUB is automatically initialized and plugins are
auto-registered when imported.
"""

from .resource_hub import ResourceHub, get_hub, set_global_hub
from .plugin import ResourcePlugin, ResourceConfig
from .storage import ConfigStorage, FileConfigStorage

# Create the global RESOURCE_HUB for convenient access
# This will be lazily initialized on first access
RESOURCE_HUB = get_hub()

__all__ = [
    "ResourceHub",
    "RESOURCE_HUB",
    "get_hub",
    "set_global_hub",
    "ResourcePlugin",
    "ResourceConfig",
    "ConfigStorage",
    "FileConfigStorage",
]
