# RAG Workflow

Ví dụ xây dựng RAG (Retrieval-Augmented Generation) pipeline hoàn chỉnh với Hush.

## Tổng quan Architecture

```
Query → Embed Query → Vector Search → Rerank → Build Context → Generate Answer
```

## Cấu hình Resources

```yaml
# resources.yaml
embedding:default:
  _class: EmbeddingConfig
  api_type: openai
  api_key: ${OPENAI_API_KEY}
  base_url: https://api.openai.com/v1
  model: text-embedding-3-small
  dimensions: 1536

rerank:default:
  _class: RerankConfig
  api_type: pinecone
  api_key: ${PINECONE_API_KEY}
  model: bge-reranker-v2-m3
  base_url: https://api.pinecone.io/rerank

llm:default:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o-mini
  base_url: https://api.openai.com/v1
```

## Ví dụ 1: Simple RAG

```python
import numpy as np
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.providers import EmbeddingNode, PromptNode, LLMNode

def cosine_similarity_search(query_vector, documents, doc_vectors, top_k=5):
    """Simple in-memory cosine similarity search."""
    query_norm = query_vector / np.linalg.norm(query_vector)
    similarities = []
    for i, doc_vec in enumerate(doc_vectors):
        doc_norm = doc_vec / np.linalg.norm(doc_vec)
        sim = np.dot(query_norm, doc_norm)
        similarities.append((i, sim, documents[i]))
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
            "retrieved": cosine_similarity_search(query_vec, documents, doc_vectors, top_k=5)
        },
        inputs={
            "query_vec": embed_query["embeddings"],
            "documents": PARENT["documents"],
            "doc_vectors": PARENT["doc_embeddings"]
        }
    )

    # Step 3: Build prompt
    prompt = PromptNode(
        name="prompt",
        inputs={
            "prompt": {
                "system": """Trả lời câu hỏi dựa trên context được cung cấp.
Nếu không tìm thấy câu trả lời trong context, hãy nói "Không tìm thấy thông tin."

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
        inputs={"messages": prompt["messages"]},
        outputs={"content": PARENT["answer"]}
    )

    retrieve["retrieved"] >> PARENT["sources"]

    START >> embed_query >> retrieve >> prompt >> llm >> END


# Usage
async def main():
    documents = [
        "Hà Nội là thủ đô của Việt Nam, nằm ở miền Bắc.",
        "TP.HCM là thành phố lớn nhất Việt Nam về dân số.",
        "Đà Nẵng là thành phố lớn nhất miền Trung Việt Nam.",
    ]

    # Pre-compute embeddings (in production, store in vector DB)
    # doc_embeddings = await embed_documents(documents)

    engine = Hush(graph)
    result = await engine.run(inputs={
        "query": "Thủ đô Việt Nam là gì?",
        "documents": documents,
        "doc_embeddings": doc_embeddings
    })

    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
```

## Ví dụ 2: RAG với Reranking

```python
from hush.providers import RerankNode

with GraphNode(name="rag-with-rerank") as graph:
    embed_query = EmbeddingNode(
        name="embed_query",
        resource_key="embedding:default",
        inputs={"texts": PARENT["query"]}
    )

    # Initial retrieval (top 20)
    retrieve = CodeNode(
        name="retrieve",
        code_fn=lambda query_vec, documents, doc_vectors: {
            "candidates": cosine_similarity_search(query_vec, documents, doc_vectors, top_k=20)
        },
        inputs={
            "query_vec": embed_query["embeddings"],
            "documents": PARENT["documents"],
            "doc_vectors": PARENT["doc_embeddings"]
        }
    )

    # Rerank to get top 5
    rerank = RerankNode(
        name="rerank",
        resource_key="rerank:default",
        inputs={
            "query": PARENT["query"],
            "documents": retrieve["candidates"],
            "top_k": 5
        }
    )

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
        inputs={"messages": prompt["messages"]},
        outputs={"content": PARENT["answer"]}
    )

    rerank["reranked_documents"] >> PARENT["sources"]
    rerank["scores"] >> PARENT["relevance_scores"]

    START >> embed_query >> retrieve >> rerank >> prompt >> llm >> END
```

## Ví dụ 3: Hybrid Search RAG

Kết hợp keyword search và vector search.

```python
def keyword_search(query, documents, top_k=10):
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
    scores = {}
    for results in results_list:
        for rank, doc in enumerate(results):
            if doc not in scores:
                scores[doc] = 0
            scores[doc] += 1 / (k + rank + 1)
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs]


with GraphNode(name="hybrid-rag") as graph:
    # Parallel: Keyword search và Vector search
    keyword = CodeNode(
        name="keyword_search",
        code_fn=lambda query, documents: {
            "results": keyword_search(query, documents, top_k=10)
        },
        inputs={"query": PARENT["query"], "documents": PARENT["documents"]}
    )

    embed_query = EmbeddingNode(
        name="embed_query",
        resource_key="embedding:default",
        inputs={"texts": PARENT["query"]}
    )

    vector_search = CodeNode(
        name="vector_search",
        code_fn=lambda query_vec, documents, doc_vectors: {
            "results": cosine_similarity_search(query_vec, documents, doc_vectors, top_k=10)
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

    rerank = RerankNode(
        name="rerank",
        resource_key="rerank:default",
        inputs={
            "query": PARENT["query"],
            "documents": merge["merged"],
            "top_k": 5
        }
    )

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
        inputs={"messages": prompt["messages"]},
        outputs={"content": PARENT["answer"]}
    )

    # Parallel branches
    START >> [keyword, embed_query]
    embed_query >> vector_search
    [keyword, vector_search] >> merge >> rerank >> prompt >> llm >> END
```

## Best Practices

1. **Retrieval top_k > Rerank top_k** - Retrieve 20, rerank to 5
2. **Chunk size**: 200-500 tokens với 10-20% overlap
3. **Cache embeddings** để tránh compute lại
4. **Monitor với tracing** để track retrieval quality

## Xem thêm

- [Agent Workflow](agent-workflow.md)
- [Embeddings & Reranking](../guides/embeddings-reranking.md)
