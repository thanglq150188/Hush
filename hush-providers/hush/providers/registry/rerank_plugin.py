"""Reranking resource plugin for ResourceHub."""

from typing import Any, Type

from hush.core.registry import ResourcePlugin, ResourceConfig
from hush.providers.rerankers.config import RerankingConfig
from hush.providers.rerankers.factory import RerankingFactory


class RerankPlugin(ResourcePlugin):
    """Plugin for reranking resources."""

    @classmethod
    def config_class(cls) -> Type[ResourceConfig]:
        """Return RerankingConfig as the config class."""
        return RerankingConfig

    @classmethod
    def create(cls, config: ResourceConfig) -> Any:
        """Create reranker instance from config."""
        if not isinstance(config, RerankingConfig):
            raise ValueError(f"Expected RerankingConfig, got {type(config)}")
        return RerankingFactory.create(config)

    @classmethod
    def resource_type(cls) -> str:
        """Return 'reranking' as the resource type."""
        return "reranking"
