# Hush Providers

LLM, embedding, and reranking providers for the Hush workflow engine.

## Installation

```bash
pip install hush-providers
```

Or with specific provider extras:

```bash
# LLM providers
pip install hush-providers[openai]         # OpenAI LLM support
pip install hush-providers[azure]          # Azure OpenAI support
pip install hush-providers[gemini]         # Google Gemini support

# Local inference (lightweight - recommended)
pip install hush-providers[onnx]           # ONNX Runtime + tokenizers
pip install hush-providers[embeddings]     # Embeddings (ONNX-based)
pip install hush-providers[rerankers]      # Rerankers (ONNX-based)

# Heavy ML frameworks (only if you need HuggingFace models)
pip install hush-providers[huggingface]    # Transformers + PyTorch

# All providers
pip install hush-providers[all-light]      # Everything except heavy ML frameworks
pip install hush-providers[all]            # Everything including transformers/torch
```

## Quick Start

### LLM Node

```python
from hush.core import WorkflowEngine, START, END, INPUT, OUTPUT
from hush.providers import LLMNode, LLMConfig

# Configure LLM
config = LLMConfig.create_config({
    "api_type": "openai",
    "api_key": "your-api-key",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4"
})

with WorkflowEngine(name="chat") as workflow:
    llm = LLMNode(
        name="chat",
        config=config,
        inputs={"messages": INPUT},
        outputs={"content": OUTPUT}
    )
    START >> llm >> END

workflow.compile()
result = await workflow.run(inputs={
    "messages": [{"role": "user", "content": "Hello!"}]
})
```

### Embedding Node

```python
from hush.providers import EmbeddingNode, EmbeddingConfig

config = EmbeddingConfig(
    api_type="vllm",
    base_url="http://localhost:8000/v1/embeddings",
    model="BAAI/bge-m3",
    dimensions=1024
)

with WorkflowEngine(name="embed") as workflow:
    embed = EmbeddingNode(
        name="embed",
        config=config,
        inputs={"texts": INPUT},
        outputs={"embeddings": OUTPUT}
    )
    START >> embed >> END
```

### Reranking Node

```python
from hush.providers import RerankNode, RerankingConfig

config = RerankingConfig(
    api_type="pipecone",
    api_key="your-api-key",
    model="bge-reranker-v2-m3"
)

with WorkflowEngine(name="rerank") as workflow:
    rerank = RerankNode(
        name="rerank",
        config=config,
        inputs={"query": INPUT, "documents": INPUT},
        outputs={"ranked_docs": OUTPUT}
    )
    START >> rerank >> END
```

## Supported Providers

### LLM Providers

- **OpenAI** - OpenAI API and compatible endpoints (vLLM, etc.)
- **Azure** - Azure OpenAI Service
- **Gemini** - Google Gemini via Vertex AI

### Embedding Providers

- **vLLM** - OpenAI-compatible embedding API
- **TEI** - Hugging Face Text Embedding Inference
- **HuggingFace** - Local HuggingFace Transformers models
- **ONNX** - Local ONNX Runtime models

### Reranking Providers

- **vLLM** - OpenAI-compatible reranking API
- **TEI** - Hugging Face Text Embedding Inference
- **HuggingFace** - Local sequence classification models
- **ONNX** - Local ONNX Runtime models
- **Pinecone** - Pinecone reranking API

## Features

### LLM Features

- ✅ Streaming and non-streaming responses
- ✅ Token counting and usage tracking
- ✅ Multimodal input support (images)
- ✅ Tool/function calling
- ✅ Batch processing
- ✅ Automatic image path resolution
- ✅ Chinese character filtering

### Embedding Features

- ✅ Async/sync interface
- ✅ Batch embedding support
- ✅ Multiple backend support (API and local)
- ✅ Configurable dimensions
- ✅ Tokenization handling

### Reranking Features

- ✅ Query-document relevance scoring
- ✅ Top-K filtering
- ✅ Threshold-based filtering
- ✅ Result sorting and ranking

## Configuration

All providers support YAML configuration:

```yaml
# llm_config.yaml
api_type: openai
api_key: your-api-key
base_url: https://api.openai.com/v1
model: gpt-4
```

Load from YAML:

```python
from hush.providers import LLMConfig
config = LLMConfig.from_yaml_file("llm_config.yaml")
```

## License

MIT
