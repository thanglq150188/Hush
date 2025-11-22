# Hush-Providers Quick Start Guide

Get started with hush-providers in 5 minutes!

## Installation

```bash
# Basic installation
pip install hush-providers

# With all provider support
pip install hush-providers[all]
```

## Quick Examples

### 1. LLM Chat Completion

```python
import asyncio
from hush.providers import LLMConfig, LLMFactory

async def main():
    # Configure LLM
    config = LLMConfig.create_config({
        "api_type": "openai",
        "api_key": "your-api-key",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4"
    })

    # Create LLM instance
    llm = LLMFactory.create(config)

    # Generate response
    response = await llm.generate(
        messages=[
            {"role": "user", "content": "What is machine learning?"}
        ],
        temperature=0.7,
        max_tokens=500
    )

    print(response.choices[0].message.content)

asyncio.run(main())
```

### 2. LLM in Workflow

```python
from hush.core import WorkflowEngine, START, END, INPUT, OUTPUT
from hush.providers import LLMNode, LLMConfig

# Configure
config = LLMConfig.create_config({
    "api_type": "openai",
    "api_key": "your-key",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4"
})

# Build workflow
with WorkflowEngine(name="chatbot") as workflow:
    llm = LLMNode(
        name="chat",
        config=config,
        inputs={"messages": INPUT},
        outputs={"content": OUTPUT}
    )
    START >> llm >> END

# Compile and run
workflow.compile()
result = await workflow.run(inputs={
    "messages": [{"role": "user", "content": "Hello!"}]
})

print(result["content"])
```

### 3. Text Embeddings

```python
from hush.providers import EmbeddingConfig, EmbeddingFactory

async def embed_texts():
    # Configure embedder
    config = EmbeddingConfig(
        api_type="vllm",
        base_url="http://localhost:8000/v1/embeddings",
        model="BAAI/bge-m3",
        dimensions=1024
    )

    # Create embedder
    embedder = EmbeddingFactory.create(config)

    # Embed texts
    result = await embedder.run([
        "Machine learning is awesome",
        "AI is transforming industries",
        "Deep learning uses neural networks"
    ])

    embeddings = result["embeddings"]
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Dimension: {len(embeddings[0])}")

asyncio.run(embed_texts())
```

### 4. Document Reranking

```python
from hush.providers import RerankingConfig, RerankingFactory

async def rerank_documents():
    # Configure reranker
    config = RerankingConfig(
        api_type="pinecone",
        api_key="your-pinecone-key",
        model="bge-reranker-v2-m3"
    )

    # Create reranker
    reranker = RerankingFactory.create(config)

    # Rerank documents
    results = await reranker.run(
        query="What is artificial intelligence?",
        texts=[
            "AI is the simulation of human intelligence by machines.",
            "Pizza is a popular Italian dish.",
            "Machine learning is a subset of AI.",
            "The weather today is sunny.",
            "Deep learning uses neural networks."
        ],
        top_k=3,
        threshold=0.5
    )

    for result in results:
        print(f"Score: {result['score']:.4f} - Index: {result['index']}")

asyncio.run(rerank_documents())
```

### 5. Complete RAG Workflow

```python
from hush.core import WorkflowEngine, START, END, INPUT, OUTPUT, GraphNode
from hush.providers import (
    EmbeddingNode,
    RerankNode,
    LLMNode,
    EmbeddingConfig,
    RerankingConfig,
    LLMConfig
)

# Configure providers
embed_config = EmbeddingConfig(
    api_type="vllm",
    base_url="http://localhost:8000/v1/embeddings",
    model="BAAI/bge-m3",
    dimensions=1024
)

rerank_config = RerankingConfig(
    api_type="pinecone",
    api_key="your-key",
    model="bge-reranker-v2-m3"
)

llm_config = LLMConfig.create_config({
    "api_type": "openai",
    "api_key": "your-key",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4"
})

# Build RAG workflow
with WorkflowEngine(name="rag_pipeline") as workflow:
    # Embed query
    embed = EmbeddingNode(
        name="embed_query",
        config=embed_config,
        inputs={"texts": INPUT},
        outputs={"query_embedding": OUTPUT}
    )

    # Rerank retrieved documents
    rerank = RerankNode(
        name="rerank_docs",
        config=rerank_config,
        inputs={"query": INPUT, "documents": INPUT},
        outputs={"ranked_docs": OUTPUT}
    )

    # Generate answer
    llm = LLMNode(
        name="generate_answer",
        config=llm_config,
        inputs={"messages": INPUT},
        outputs={"answer": OUTPUT}
    )

    START >> [embed, rerank] >> llm >> END

# Run the pipeline
workflow.compile()
result = await workflow.run(inputs={
    "texts": ["What is machine learning?"],
    "query": "What is machine learning?",
    "documents": [
        "Machine learning is a subset of AI",
        "AI is transforming industries",
        "Deep learning uses neural networks"
    ],
    "messages": [{"role": "user", "content": "Explain machine learning"}]
})
```

## Configuration from YAML

```yaml
# llm_config.yaml
api_type: openai
api_key: your-api-key
base_url: https://api.openai.com/v1
model: gpt-4
```

```python
from pathlib import Path
from hush.providers import LLMConfig

# Load from YAML
config = LLMConfig.from_yaml_file(Path("llm_config.yaml"))
```

## Supported Provider Types

### LLM Types
- `openai` - OpenAI API
- `azure` - Azure OpenAI
- `gemini` - Google Gemini
- `vllm` - vLLM (OpenAI-compatible)

### Embedding Types
- `vllm` - vLLM OpenAI-compatible
- `tei` - Hugging Face TEI
- `hf` - Local HuggingFace
- `onnx` - Local ONNX Runtime

### Reranking Types
- `vllm` - vLLM OpenAI-compatible
- `tei` - Hugging Face TEI
- `hf` - Local HuggingFace
- `onnx` - Local ONNX Runtime
- `pinecone` - Pinecone API

## Next Steps

- Read the full [README.md](README.md) for more details
- Check [MIGRATION_SUMMARY.md](MIGRATION_SUMMARY.md) for migration guide
- Browse the examples in the `tests/` directory
- Explore provider-specific configuration options

## Need Help?

- GitHub Issues: https://github.com/your-org/hush/issues
- Documentation: https://github.com/your-org/hush#readme
