# Embeddings và Reranking

Hướng dẫn này sẽ giúp bạn sử dụng embedding và reranking cho các ứng dụng RAG (Retrieval-Augmented Generation).

## Embedding Providers

Hush hỗ trợ nhiều embedding providers:

| Provider | Type | Đặc điểm |
|----------|------|----------|
| OpenAI | API-based | Đơn giản, chất lượng tốt |
| Azure OpenAI | API-based | Enterprise, compliance |
| vLLM | Self-hosted | OpenAI-compatible API |
| TEI (Text Embedding Inference) | Self-hosted | HuggingFace optimized |
| HuggingFace | Local | Chạy local, không cần API |
| ONNX | Local | Fast inference với ONNX Runtime |

## Cấu hình Embedding

### OpenAI Embeddings

```yaml
# resources.yaml
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
  api_type: vllm  # hoặc "tei"
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

### ONNX Runtime

```yaml
embedding:onnx:
  _class: EmbeddingConfig
  api_type: onnx
  model: path/to/model.onnx
  dimensions: 384
```

## EmbeddingNode - Tạo Embeddings

### Basic usage

```python
from hush.core import GraphNode, START, END, PARENT
from hush.providers import EmbeddingNode

with GraphNode(name="embed-workflow") as graph:
    embed = EmbeddingNode(
        name="embed",
        resource_key="embedding:openai",
        inputs={"texts": PARENT["documents"]}
    )

    embed["embeddings"] >> PARENT["vectors"]

    START >> embed >> END

# Input: {"documents": ["Hello world", "Goodbye world"]}
# Output: {"vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...]]}
```

### Single text vs batch

```python
# Single text
embed = EmbeddingNode(
    name="embed",
    resource_key="embedding:openai",
    inputs={"texts": PARENT["query"]}  # String
)
# Output: {"embeddings": [0.1, 0.2, ...]}

# Batch texts
embed = EmbeddingNode(
    name="embed",
    resource_key="embedding:openai",
    inputs={"texts": PARENT["documents"]}  # List[str]
)
# Output: {"embeddings": [[0.1, ...], [0.2, ...]]}
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
| HuggingFace | Local | Chạy local |
| ONNX | Local | Fast inference |

## Cấu hình Reranking

### Pinecone Reranker

```yaml
# resources.yaml
rerank:pinecone:
  _class: RerankingConfig
  api_type: pinecone
  api_key: ${PINECONE_API_KEY}
  model: bge-reranker-v2-m3
  base_url: https://api.pinecone.io/rerank
```

### Cohere Reranker

```yaml
rerank:cohere:
  _class: RerankingConfig
  api_type: cohere
  api_key: ${COHERE_API_KEY}
  model: rerank-english-v3.0
```

### vLLM / TEI Reranker

```yaml
rerank:local:
  _class: RerankingConfig
  api_type: vllm  # hoặc "tei"
  base_url: http://localhost:8081
  model: BAAI/bge-reranker-v2-m3
```

### HuggingFace (Local)

```yaml
rerank:hf:
  _class: RerankingConfig
  api_type: hf
  model: BAAI/bge-reranker-base
```

## RerankNode - Reranking Documents

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
        }
    )

    rerank["reranked_documents"] >> PARENT["results"]

    START >> rerank >> END
```

### RerankNode inputs

| Input | Type | Mô tả |
|-------|------|-------|
| `query` | str | Query để rank theo |
| `documents` | list | List of documents/texts |
| `top_k` | int | Số kết quả trả về (default: 10) |

### RerankNode outputs

| Output | Type | Mô tả |
|--------|------|-------|
| `reranked_documents` | list | Documents sorted by relevance |
| `scores` | list | Relevance scores |
| `indices` | list | Original indices |

## RAG Pipeline: Embed → Retrieve → Rerank

### Ví dụ hoàn chỉnh

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

    # Step 2: Vector search (CodeNode for custom logic)
    retrieve = CodeNode(
        name="retrieve",
        code_fn=lambda query_vector, documents, doc_vectors: {
            # Compute cosine similarity và return top 20
            "retrieved": retrieve_top_k(query_vector, documents, doc_vectors, k=20)
        },
        inputs={
            "query_vector": embed_query["embeddings"],
            "documents": PARENT["knowledge_base"],
            "doc_vectors": PARENT["doc_embeddings"]
        }
    )

    # Step 3: Rerank to get top 5
    rerank = RerankNode(
        name="rerank",
        resource_key="rerank:pinecone",
        inputs={
            "query": PARENT["query"],
            "documents": retrieve["retrieved"],
            "top_k": 5
        }
    )

    # Step 4: Build prompt with context
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
        resource_key="gpt-4",
        inputs={"messages": prompt["messages"]}
    )

    llm["content"] >> PARENT["answer"]

    START >> embed_query >> retrieve >> rerank >> prompt >> llm >> END
```

## Batch Embedding

Xử lý nhiều documents hiệu quả với MapNode.

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
        inputs={
            "docs": PARENT["documents"],
            "batch_size": 100
        }
    )

    # Parallel embedding with MapNode
    with MapNode(
        name="embed_batches",
        inputs={"batch": Each(batch["batches"])},
        max_concurrency=5
    ) as map_node:
        embed = EmbeddingNode(
            name="embed",
            resource_key="embedding:openai",
            inputs={"texts": PARENT["batch"]}
        )
        embed["embeddings"] >> PARENT["embeddings"]
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

## Hybrid Search

Kết hợp keyword search và vector search.

```python
with GraphNode(name="hybrid-search") as graph:
    # Parallel: keyword search và vector search
    keyword_search = CodeNode(
        name="keyword_search",
        code_fn=lambda query, docs: {
            "results": [d for d in docs if query.lower() in d.lower()][:10]
        },
        inputs={
            "query": PARENT["query"],
            "docs": PARENT["documents"]
        }
    )

    embed_query = EmbeddingNode(
        name="embed_query",
        resource_key="embedding:openai",
        inputs={"texts": PARENT["query"]}
    )

    vector_search = CodeNode(
        name="vector_search",
        code_fn=lambda query_vec, docs, doc_vecs: {
            "results": cosine_similarity_search(query_vec, docs, doc_vecs, top_k=10)
        },
        inputs={
            "query_vec": embed_query["embeddings"],
            "docs": PARENT["documents"],
            "doc_vecs": PARENT["doc_embeddings"]
        }
    )

    # Merge results with RRF (Reciprocal Rank Fusion)
    merge = CodeNode(
        name="merge",
        code_fn=lambda keyword_results, vector_results: {
            "merged": reciprocal_rank_fusion(keyword_results, vector_results, k=60)
        },
        inputs={
            "keyword_results": keyword_search["results"],
            "vector_results": vector_search["results"]
        }
    )

    # Rerank merged results
    rerank = RerankNode(
        name="rerank",
        resource_key="rerank:pinecone",
        inputs={
            "query": PARENT["query"],
            "documents": merge["merged"],
            "top_k": 5
        },
        outputs={"*": PARENT}
    )

    START >> [keyword_search, embed_query]
    embed_query >> vector_search
    [keyword_search, vector_search] >> merge >> rerank >> END
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
  embed_batch_size: 100  # Batch 100 texts per request
```

### 3. Cache embeddings

```python
# Pre-compute embeddings cho knowledge base
doc_embeddings = await embed_node.run({"texts": all_documents})

# Store embeddings trong vector database
vector_db.upsert(doc_embeddings)
```

### 4. Rerank để improve precision

```python
# Two-stage retrieval
# Stage 1: Fast retrieval (top 50)
retrieved = vector_search(query, top_k=50)

# Stage 2: Rerank (top 5)
reranked = rerank(query, retrieved, top_k=5)
```

### 5. Monitor latency và costs

```python
from hush.observability import LangfuseTracer

tracer = LangfuseTracer(resource_key="langfuse:default")
result = await engine.run(inputs={...}, tracer=tracer)

# Langfuse sẽ track embedding và reranking latency
```

## Tiếp theo

- [RAG Workflow](../examples/rag-workflow.md) - Ví dụ RAG hoàn chỉnh
- [Tích hợp LLM](llm-integration.md) - LLM generation sau retrieval
- [Thực thi song song](parallel-execution.md) - Parallel embedding
