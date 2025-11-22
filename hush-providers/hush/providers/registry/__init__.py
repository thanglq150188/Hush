"""Resource registry plugins for hush-providers.

This module provides plugins that extend hush-core's ResourceHub
to support LLM, embedding, and reranking resources.
"""

from .llm_plugin import LLMPlugin
from .embedding_plugin import EmbeddingPlugin
from .rerank_plugin import RerankPlugin

__all__ = [
    "LLMPlugin",
    "EmbeddingPlugin",
    "RerankPlugin",
]
