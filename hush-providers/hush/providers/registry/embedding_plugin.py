"""Embedding resource plugin for ResourceHub."""

from typing import Any, Type

from hush.core.registry import ResourcePlugin, ResourceConfig
from hush.providers.embeddings.config import EmbeddingConfig
from hush.providers.embeddings.factory import EmbeddingFactory


class EmbeddingPlugin(ResourcePlugin):
    """Plugin for embedding resources."""

    @classmethod
    def config_class(cls) -> Type[ResourceConfig]:
        """Return EmbeddingConfig as the config class."""
        return EmbeddingConfig

    @classmethod
    def create(cls, config: ResourceConfig) -> Any:
        """Create embedding instance from config."""
        if not isinstance(config, EmbeddingConfig):
            raise ValueError(f"Expected EmbeddingConfig, got {type(config)}")
        return EmbeddingFactory.create(config)

    @classmethod
    def resource_type(cls) -> str:
        """Return 'embedding' as the resource type."""
        return "embedding"
