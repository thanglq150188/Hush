# Embeddings và Reranking

Hướng dẫn sử dụng embedding và reranking cho các ứng dụng RAG (Retrieval-Augmented Generation).

## Embedding Providers

| Provider | Type | Đặc điểm |
|----------|------|----------|
| OpenAI | API-based | Đơn giản, chất lượng tốt |
| Azure OpenAI | API-based | Enterprise, compliance |
| vLLM | Self-hosted | OpenAI-compatible API |
| TEI | Self-hosted | HuggingFace optimized |
| HuggingFace | Local | Chạy local, không cần API |
| ONNX | Local | Fast inference với ONNX Runtime |

## Cấu hình Embedding

### OpenAI Embeddings

```yaml
embedding:openai:
  _class: EmbeddingConfig
  api_type: openai
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
  model: text-embedding-3-small
  dimensions: 1536
```

### vLLM / TEI (Self-hosted)

```yaml
embedding:local:
  _class: EmbeddingConfig
  api_type: vllm
  base_url: http://localhost:8080/v1
  model: BAAI/bge-m3
  dimensions: 1024
  embed_batch_size: 32
```

### HuggingFace (Local)

```yaml
embedding:hf:
  _class: EmbeddingConfig
  api_type: hf
  model: BAAI/bge-small-en-v1.5
  dimensions: 384
```

## EmbeddingNode

### Basic usage

```python
from hush.core import GraphNode, START, END, PARENT
from hush.providers import EmbeddingNode

with GraphNode(name="embed-workflow") as graph:
    embed = EmbeddingNode(
        name="embed",
        resource_key="embedding:openai",
        inputs={"texts": PARENT["documents"]},
        outputs={"embeddings": PARENT["vectors"]}
    )

    START >> embed >> END

# Input: {"documents": ["Hello world", "Goodbye world"]}
# Output: {"vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...]]}
```

### EmbeddingNode outputs

| Output | Type | Mô tả |
|--------|------|-------|
| `embeddings` | list | Vector hoặc list of vectors |
| `model_used` | str | Model đã sử dụng |
| `dimensions` | int | Số dimensions |
| `tokens_used` | int | Tổng số tokens |

## Reranking Providers

| Provider | Type | Đặc điểm |
|----------|------|----------|
| Cohere | API-based | Chất lượng cao |
| Pinecone | API-based | Tích hợp vector DB |
| vLLM | Self-hosted | Cross-encoder models |
| TEI | Self-hosted | HuggingFace optimized |

## Cấu hình Reranking

### Pinecone Reranker

```yaml
rerank:pinecone:
  _class: RerankConfig
  api_type: pinecone
  api_key: ${PINECONE_API_KEY}
  model: bge-reranker-v2-m3
  base_url: https://api.pinecone.io/rerank
```

### Cohere Reranker

```yaml
rerank:cohere:
  _class: RerankConfig
  api_type: cohere
  api_key: ${COHERE_API_KEY}
  model: rerank-english-v3.0
```

## RerankNode

### Basic usage

```python
from hush.providers import RerankNode

with GraphNode(name="rerank-workflow") as graph:
    rerank = RerankNode(
        name="rerank",
        resource_key="rerank:pinecone",
        inputs={
            "query": PARENT["query"],
            "documents": PARENT["documents"],
            "top_k": 5
        },
        outputs={"reranked_documents": PARENT["results"]}
    )

    START >> rerank >> END
```

### RerankNode outputs

| Output | Type | Mô tả |
|--------|------|-------|
| `reranked_documents` | list | Documents sorted by relevance |
| `scores` | list | Relevance scores |
| `indices` | list | Original indices |

## RAG Pipeline: Embed → Retrieve → Rerank

```python
from hush.core import GraphNode, CodeNode, START, END, PARENT
from hush.providers import EmbeddingNode, RerankNode, PromptNode, LLMNode

with GraphNode(name="rag-pipeline") as graph:
    # Step 1: Embed query
    embed_query = EmbeddingNode(
        name="embed_query",
        resource_key="embedding:openai",
        inputs={"texts": PARENT["query"]}
    )

    # Step 2: Vector search (custom logic)
    retrieve = CodeNode(
        name="retrieve",
        code_fn=lambda query_vector, documents, doc_vectors: {
            "retrieved": retrieve_top_k(query_vector, documents, doc_vectors, k=20)
        },
        inputs={
            "query_vector": embed_query["embeddings"],
            "documents": PARENT["knowledge_base"],
            "doc_vectors": PARENT["doc_embeddings"]
        }
    )

    # Step 3: Rerank
    rerank = RerankNode(
        name="rerank",
        resource_key="rerank:pinecone",
        inputs={
            "query": PARENT["query"],
            "documents": retrieve["retrieved"],
            "top_k": 5
        }
    )

    # Step 4: Build prompt
    prompt = PromptNode(
        name="prompt",
        inputs={
            "prompt": {
                "system": "Trả lời dựa trên context sau:\n\n{context}",
                "user": "{query}"
            },
            "context": rerank["reranked_documents"],
            "query": PARENT["query"]
        }
    )

    # Step 5: Generate answer
    llm = LLMNode(
        name="llm",
        resource_key="gpt-4o",
        inputs={"messages": prompt["messages"]},
        outputs={"content": PARENT["answer"]}
    )

    START >> embed_query >> retrieve >> rerank >> prompt >> llm >> END
```

## Batch Embedding

```python
from hush.core.nodes.iteration import MapNode
from hush.core.nodes.iteration.base import Each

with GraphNode(name="batch-embed") as graph:
    # Split documents into batches
    batch = CodeNode(
        name="batch",
        code_fn=lambda docs, batch_size: {
            "batches": [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]
        },
        inputs={"docs": PARENT["documents"], "batch_size": 100}
    )

    # Parallel embedding
    with MapNode(
        name="embed_batches",
        inputs={"batch": Each(batch["batches"])},
        max_concurrency=5
    ) as map_node:
        embed = EmbeddingNode(
            name="embed",
            resource_key="embedding:openai",
            inputs={"texts": PARENT["batch"]},
            outputs={"embeddings": PARENT}
        )
        START >> embed >> END

    # Flatten results
    flatten = CodeNode(
        name="flatten",
        code_fn=lambda batches: {
            "all_embeddings": [e for batch in batches for e in batch]
        },
        inputs={"batches": map_node["embeddings"]},
        outputs={"*": PARENT}
    )

    START >> batch >> map_node >> flatten >> END
```

## Best Practices

### 1. Chọn embedding model phù hợp

| Use Case | Recommended Model |
|----------|-------------------|
| General English | `text-embedding-3-small` |
| Multilingual | `BAAI/bge-m3` |
| Code | `text-embedding-3-large` |
| High accuracy | `text-embedding-3-large` |
| Fast/cheap | `BAAI/bge-small-en-v1.5` |

### 2. Batch để tối ưu throughput

```yaml
embedding:optimized:
  _class: EmbeddingConfig
  api_type: openai
  model: text-embedding-3-small
  embed_batch_size: 100
```

### 3. Cache embeddings

Pre-compute embeddings cho knowledge base và store trong vector database.

### 4. Rerank để improve precision

Two-stage retrieval:
- Stage 1: Fast retrieval (top 50)
- Stage 2: Rerank (top 5)

## Xem thêm

- [RAG Workflow](../examples/rag-workflow.md)
- [Tích hợp LLM](llm-integration.md)
- [Thực thi song song](parallel-execution.md)
