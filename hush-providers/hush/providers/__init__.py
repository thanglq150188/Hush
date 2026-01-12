"""
Hush Providers - LLM, embedding, and reranking providers for hush workflows.

This package provides AI provider integrations for the Hush workflow engine:
- LLM providers (OpenAI, Azure, Gemini, vLLM)
- Embedding providers (vLLM, TEI, HuggingFace, ONNX)
- Reranking providers (vLLM, TEI, HuggingFace, ONNX, Pinecone)
- Workflow nodes for integrating providers into workflows
"""

# LLM exports
from hush.providers.llms import (
    BaseLLM,
    LLMType,
    LLMConfig,
    OpenAIConfig,
    AzureConfig,
    GeminiConfig,
    LLMFactory,
    LLMGenerator,
    OpenAISDKModel,
    AzureSDKModel,
    # GeminiOpenAISDKModel - lazy loaded, access via hush.providers.llms.GeminiOpenAISDKModel
)

# Embedding exports
from hush.providers.embeddings import (
    BaseEmbedder,
    EmbeddingType,
    EmbeddingConfig,
    EmbeddingFactory,
    VLLMEmbedding,
    TEIEmbedding,
    HFEmbedding,
    ONNXEmbedding,
)

# Reranking exports
from hush.providers.rerankers import (
    BaseReranker,
    RerankingType,
    RerankingConfig,
    RerankingFactory,
    VLLMReranker,
    TEIReranker,
    HFReranker,
    ONNXReranker,
    PineconeReranker,
)

# Node exports
from hush.providers.nodes import (
    LLMNode,
    EmbeddingNode,
    RerankNode,
    PromptNode,
    LLMChainNode,
)

# Registry plugin exports
from hush.providers.registry import (
    LLMPlugin,
    EmbeddingPlugin,
    RerankPlugin,
)

__version__ = "0.1.0"

__all__ = [
    # LLM
    "BaseLLM",
    "LLMType",
    "LLMConfig",
    "OpenAIConfig",
    "AzureConfig",
    "GeminiConfig",
    "LLMFactory",
    "LLMGenerator",
    "OpenAISDKModel",
    "AzureSDKModel",
    # Embedding
    "BaseEmbedder",
    "EmbeddingType",
    "EmbeddingConfig",
    "EmbeddingFactory",
    "VLLMEmbedding",
    "TEIEmbedding",
    "HFEmbedding",
    "ONNXEmbedding",
    # Reranking
    "BaseReranker",
    "RerankingType",
    "RerankingConfig",
    "RerankingFactory",
    "VLLMReranker",
    "TEIReranker",
    "HFReranker",
    "ONNXReranker",
    "PineconeReranker",
    # Nodes
    "LLMNode",
    "EmbeddingNode",
    "RerankNode",
    "PromptNode",
    "LLMChainNode",
    # Registry Plugins
    "LLMPlugin",
    "EmbeddingPlugin",
    "RerankPlugin",
]
