# RAG Workflow

Ví dụ này hướng dẫn xây dựng một RAG (Retrieval-Augmented Generation) pipeline hoàn chỉnh với Hush.

## Tổng quan Architecture

```
Query → Embed Query → Vector Search → Rerank → Build Context → Generate Answer
```

## Cấu hình Resources

### resources.yaml

```yaml
# Embedding
embedding:default:
  _class: EmbeddingConfig
  api_type: openai
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
  model: text-embedding-3-small
  dimensions: 1536

# Reranking
rerank:default:
  _class: RerankingConfig
  api_type: pinecone
  api_key: ${PINECONE_API_KEY}
  model: bge-reranker-v2-m3
  base_url: https://api.pinecone.io/rerank

# LLM
llm:default:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
```

## Ví dụ 1: Simple RAG

RAG pipeline cơ bản với in-memory vector search.

```python
import numpy as np
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.providers import EmbeddingNode, PromptNode, LLMNode

# Helper: Cosine similarity search
def cosine_similarity_search(query_vector, documents, doc_vectors, top_k=5):
    """Simple in-memory cosine similarity search."""
    query_norm = query_vector / np.linalg.norm(query_vector)
    similarities = []
    for i, doc_vec in enumerate(doc_vectors):
        doc_norm = doc_vec / np.linalg.norm(doc_vec)
        sim = np.dot(query_norm, doc_norm)
        similarities.append((i, sim, documents[i]))

    # Sort by similarity descending
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [doc for _, _, doc in similarities[:top_k]]


with GraphNode(name="simple-rag") as graph:
    # Step 1: Embed query
    embed_query = EmbeddingNode(
        name="embed_query",
        resource_key="embedding:default",
        inputs={"texts": PARENT["query"]}
    )

    # Step 2: Vector search
    retrieve = CodeNode(
        name="retrieve",
        code_fn=lambda query_vec, documents, doc_vectors: {
            "retrieved": cosine_similarity_search(
                query_vec, documents, doc_vectors, top_k=5
            )
        },
        inputs={
            "query_vec": embed_query["embeddings"],
            "documents": PARENT["documents"],
            "doc_vectors": PARENT["doc_embeddings"]
        }
    )

    # Step 3: Build prompt with context
    prompt = PromptNode(
        name="prompt",
        inputs={
            "prompt": {
                "system": """Bạn là assistant trả lời câu hỏi dựa trên context được cung cấp.
Chỉ sử dụng thông tin từ context để trả lời. Nếu không tìm thấy câu trả lời trong context, hãy nói "Tôi không tìm thấy thông tin về điều này trong tài liệu."

Context:
{context}""",
                "user": "{query}"
            },
            "context": retrieve["retrieved"],
            "query": PARENT["query"]
        }
    )

    # Step 4: Generate answer
    llm = LLMNode(
        name="llm",
        resource_key="llm:default",
        inputs={"messages": prompt["messages"]}
    )

    llm["content"] >> PARENT["answer"]
    retrieve["retrieved"] >> PARENT["sources"]

    START >> embed_query >> retrieve >> prompt >> llm >> END


# Usage
async def main():
    # Sample knowledge base
    documents = [
        "Hà Nội là thủ đô của Việt Nam, nằm ở miền Bắc.",
        "TP.HCM là thành phố lớn nhất Việt Nam về dân số.",
        "Đà Nẵng là thành phố lớn nhất miền Trung Việt Nam.",
        "Việt Nam có 63 tỉnh thành phố.",
        "Phở là món ăn truyền thống nổi tiếng của Việt Nam.",
    ]

    # Pre-compute embeddings (in production, store in vector DB)
    from hush.providers.embeddings import get_embedding_client
    from hush.core.registry import ResourceHub

    hub = ResourceHub.instance()
    embed_config = hub.get("embedding:default")
    embed_client = get_embedding_client(embed_config)
    doc_embeddings = await embed_client.embed(documents)

    # Run RAG
    engine = Hush(graph)
    result = await engine.run(inputs={
        "query": "Thủ đô Việt Nam là gì?",
        "documents": documents,
        "doc_embeddings": doc_embeddings
    })

    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")

import asyncio
asyncio.run(main())
```

## Ví dụ 2: RAG với Reranking

Thêm reranking để cải thiện độ chính xác.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.providers import EmbeddingNode, RerankNode, PromptNode, LLMNode

with GraphNode(name="rag-with-rerank") as graph:
    # Step 1: Embed query
    embed_query = EmbeddingNode(
        name="embed_query",
        resource_key="embedding:default",
        inputs={"texts": PARENT["query"]}
    )

    # Step 2: Initial retrieval (top 20)
    retrieve = CodeNode(
        name="retrieve",
        code_fn=lambda query_vec, documents, doc_vectors: {
            "candidates": cosine_similarity_search(
                query_vec, documents, doc_vectors, top_k=20
            )
        },
        inputs={
            "query_vec": embed_query["embeddings"],
            "documents": PARENT["documents"],
            "doc_vectors": PARENT["doc_embeddings"]
        }
    )

    # Step 3: Rerank to get top 5
    rerank = RerankNode(
        name="rerank",
        resource_key="rerank:default",
        inputs={
            "query": PARENT["query"],
            "documents": retrieve["candidates"],
            "top_k": 5
        }
    )

    # Step 4: Build prompt
    prompt = PromptNode(
        name="prompt",
        inputs={
            "prompt": {
                "system": """Trả lời câu hỏi dựa trên context sau:

{context}

Nếu không có thông tin, trả lời "Không tìm thấy thông tin.""",
                "user": "{query}"
            },
            "context": rerank["reranked_documents"],
            "query": PARENT["query"]
        }
    )

    # Step 5: Generate
    llm = LLMNode(
        name="llm",
        resource_key="llm:default",
        inputs={"messages": prompt["messages"]}
    )

    llm["content"] >> PARENT["answer"]
    rerank["reranked_documents"] >> PARENT["sources"]
    rerank["scores"] >> PARENT["relevance_scores"]

    START >> embed_query >> retrieve >> rerank >> prompt >> llm >> END
```

## Ví dụ 3: Hybrid Search RAG

Kết hợp keyword search và vector search.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.providers import EmbeddingNode, RerankNode, PromptNode, LLMNode

def keyword_search(query, documents, top_k=10):
    """Simple keyword matching."""
    query_terms = set(query.lower().split())
    results = []
    for doc in documents:
        doc_terms = set(doc.lower().split())
        overlap = len(query_terms & doc_terms)
        if overlap > 0:
            results.append((overlap, doc))
    results.sort(reverse=True)
    return [doc for _, doc in results[:top_k]]


def reciprocal_rank_fusion(results_list, k=60):
    """Combine multiple ranked lists using RRF."""
    scores = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            if doc not in scores:
                scores[doc] = 0
            scores[doc] += 1 / (k + rank + 1)

    # Sort by score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs]


with GraphNode(name="hybrid-rag") as graph:
    # Parallel: Keyword search và Vector search
    keyword = CodeNode(
        name="keyword_search",
        code_fn=lambda query, documents: {
            "results": keyword_search(query, documents, top_k=10)
        },
        inputs={
            "query": PARENT["query"],
            "documents": PARENT["documents"]
        }
    )

    embed_query = EmbeddingNode(
        name="embed_query",
        resource_key="embedding:default",
        inputs={"texts": PARENT["query"]}
    )

    vector_search = CodeNode(
        name="vector_search",
        code_fn=lambda query_vec, documents, doc_vectors: {
            "results": cosine_similarity_search(
                query_vec, documents, doc_vectors, top_k=10
            )
        },
        inputs={
            "query_vec": embed_query["embeddings"],
            "documents": PARENT["documents"],
            "doc_vectors": PARENT["doc_embeddings"]
        }
    )

    # Merge with RRF
    merge = CodeNode(
        name="merge",
        code_fn=lambda kw_results, vec_results: {
            "merged": reciprocal_rank_fusion([kw_results, vec_results])[:20]
        },
        inputs={
            "kw_results": keyword["results"],
            "vec_results": vector_search["results"]
        }
    )

    # Rerank
    rerank = RerankNode(
        name="rerank",
        resource_key="rerank:default",
        inputs={
            "query": PARENT["query"],
            "documents": merge["merged"],
            "top_k": 5
        }
    )

    # Generate
    prompt = PromptNode(
        name="prompt",
        inputs={
            "prompt": {
                "system": "Trả lời dựa trên context:\n\n{context}",
                "user": "{query}"
            },
            "context": rerank["reranked_documents"],
            "query": PARENT["query"]
        }
    )

    llm = LLMNode(
        name="llm",
        resource_key="llm:default",
        inputs={"messages": prompt["messages"]}
    )

    llm["content"] >> PARENT["answer"]
    rerank["reranked_documents"] >> PARENT["sources"]

    # Parallel branches
    START >> [keyword, embed_query]
    embed_query >> vector_search
    [keyword, vector_search] >> merge >> rerank >> prompt >> llm >> END
```

## Ví dụ 4: Multi-hop RAG

RAG với nhiều bước retrieval cho câu hỏi phức tạp.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.iteration.while_loop_node import WhileLoopNode
from hush.providers import EmbeddingNode, RerankNode, PromptNode, LLMNode

with GraphNode(name="multi-hop-rag") as graph:
    # Initial query processing
    init = CodeNode(
        name="init",
        code_fn=lambda query: {
            "current_query": query,
            "all_context": [],
            "hop_count": 0,
            "max_hops": 3,
            "done": False
        },
        inputs={"query": PARENT["query"]}
    )

    # Multi-hop loop
    with WhileLoopNode(
        name="retrieval_loop",
        inputs={
            "current_query": init["current_query"],
            "all_context": init["all_context"],
            "hop_count": init["hop_count"],
            "max_hops": init["max_hops"],
            "done": init["done"]
        },
        stop_condition="done == True or hop_count >= max_hops",
        max_iterations=5
    ) as loop:
        # Retrieve for current query
        embed = EmbeddingNode(
            name="embed",
            resource_key="embedding:default",
            inputs={"texts": PARENT["current_query"]}
        )

        retrieve = CodeNode(
            name="retrieve",
            code_fn=lambda query_vec, documents, doc_vectors: {
                "retrieved": cosine_similarity_search(
                    query_vec, documents, doc_vectors, top_k=3
                )
            },
            inputs={
                "query_vec": embed["embeddings"],
                "documents": PARENT["documents"],
                "doc_vectors": PARENT["doc_embeddings"]
            }
        )

        # Check if we need more hops
        check = CodeNode(
            name="check",
            code_fn=lambda retrieved, all_context, hop_count, original_query: {
                "new_context": all_context + retrieved,
                "new_hop": hop_count + 1,
                # Simple heuristic: stop if we have enough context
                "is_done": len(all_context) + len(retrieved) >= 6,
                "next_query": original_query  # Could be refined by LLM
            },
            inputs={
                "retrieved": retrieve["retrieved"],
                "all_context": PARENT["all_context"],
                "hop_count": PARENT["hop_count"],
                "original_query": PARENT["original_query"]
            }
        )

        check["new_context"] >> PARENT["all_context"]
        check["new_hop"] >> PARENT["hop_count"]
        check["is_done"] >> PARENT["done"]
        check["next_query"] >> PARENT["current_query"]

        START >> embed >> retrieve >> check >> END

    # Final generation
    prompt = PromptNode(
        name="final_prompt",
        inputs={
            "prompt": {
                "system": """Trả lời câu hỏi dựa trên tất cả context sau:

{context}""",
                "user": "{query}"
            },
            "context": loop["all_context"],
            "query": PARENT["query"]
        }
    )

    llm = LLMNode(
        name="llm",
        resource_key="llm:default",
        inputs={"messages": prompt["messages"]}
    )

    llm["content"] >> PARENT["answer"]
    loop["all_context"] >> PARENT["all_sources"]
    loop["hop_count"] >> PARENT["hops_used"]

    START >> init >> loop >> prompt >> llm >> END
```

## Ví dụ 5: Streaming RAG

RAG với streaming response.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.providers import EmbeddingNode, RerankNode, PromptNode, LLMNode

with GraphNode(name="streaming-rag") as graph:
    # Retrieval (same as before)
    embed_query = EmbeddingNode(
        name="embed_query",
        resource_key="embedding:default",
        inputs={"texts": PARENT["query"]}
    )

    retrieve = CodeNode(
        name="retrieve",
        code_fn=lambda query_vec, documents, doc_vectors: {
            "retrieved": cosine_similarity_search(
                query_vec, documents, doc_vectors, top_k=5
            )
        },
        inputs={
            "query_vec": embed_query["embeddings"],
            "documents": PARENT["documents"],
            "doc_vectors": PARENT["doc_embeddings"]
        }
    )

    prompt = PromptNode(
        name="prompt",
        inputs={
            "prompt": {
                "system": "Context:\n{context}",
                "user": "{query}"
            },
            "context": retrieve["retrieved"],
            "query": PARENT["query"]
        }
    )

    # Streaming LLM
    llm = LLMNode(
        name="llm",
        resource_key="llm:default",
        inputs={"messages": prompt["messages"]},
        stream=True  # Enable streaming
    )

    llm["content"] >> PARENT["answer"]

    START >> embed_query >> retrieve >> prompt >> llm >> END


# Usage with streaming
async def main():
    engine = Hush(graph)

    async for chunk in engine.stream(inputs={
        "query": "Thủ đô Việt Nam là gì?",
        "documents": documents,
        "doc_embeddings": doc_embeddings
    }):
        if "answer" in chunk:
            print(chunk["answer"], end="", flush=True)

import asyncio
asyncio.run(main())
```

## Batch Embedding cho Knowledge Base

Pre-compute embeddings hiệu quả.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.iteration import MapNode
from hush.core.nodes.iteration.base import Each
from hush.providers import EmbeddingNode

with GraphNode(name="batch-embed-kb") as graph:
    # Split into batches
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

    # Parallel embedding
    with MapNode(
        name="embed_batches",
        inputs={"batch": Each(batch["batches"])},
        max_concurrency=5  # Avoid rate limits
    ) as map_node:
        embed = EmbeddingNode(
            name="embed",
            resource_key="embedding:default",
            inputs={"texts": PARENT["batch"]}
        )
        embed["embeddings"] >> PARENT["embeddings"]
        START >> embed >> END

    # Flatten
    flatten = CodeNode(
        name="flatten",
        code_fn=lambda batches: {
            "all_embeddings": [e for batch in batches for e in batch]
        },
        inputs={"batches": map_node["embeddings"]},
        outputs={"*": PARENT}
    )

    START >> batch >> map_node >> flatten >> END


# Usage
async def main():
    engine = Hush(graph)
    result = await engine.run(inputs={
        "documents": all_documents  # Có thể là hàng ngàn documents
    })

    # Store embeddings
    embeddings = result["all_embeddings"]
    # Save to vector DB or file

import asyncio
asyncio.run(main())
```

## Best Practices

### 1. Retrieval Settings

```python
# Retrieval top_k > Rerank top_k
# Ví dụ: retrieve 20, rerank to 5
retrieve_top_k = 20
rerank_top_k = 5
```

### 2. Chunk Size

```python
# Optimal chunk size: 200-500 tokens
# Overlap: 10-20% of chunk size
chunk_size = 400
chunk_overlap = 50
```

### 3. Cost Optimization

```yaml
# Sử dụng model rẻ cho embedding
embedding:default:
  model: text-embedding-3-small  # Cheaper
  # model: text-embedding-3-large  # More accurate but expensive
```

### 4. Caching

```python
# Cache embeddings để tránh compute lại
import redis.asyncio as redis

redis_client = redis.from_url("redis://localhost")

async def get_or_compute_embedding(text):
    cache_key = f"embed:{hash(text)}"
    cached = await redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    embedding = await compute_embedding(text)
    await redis_client.setex(cache_key, 3600, json.dumps(embedding))
    return embedding
```

### 5. Monitoring

```python
from hush.observability import LangfuseTracer

tracer = LangfuseTracer(
    resource_key="langfuse:default",
    tags=["rag", "production"]
)

result = await engine.run(
    inputs={...},
    tracer=tracer
)
# Track: retrieval latency, rerank scores, generation quality
```

## Tiếp theo

- [Agent Workflow](agent-workflow.md) - Xây dựng AI Agent
- [Multi-Model Workflow](multi-model.md) - Sử dụng nhiều models
- [Embeddings và Reranking](../guides/embeddings-reranking.md) - Chi tiết về embedding/reranking
