# Reranker Providers

<!--
MỤC ĐÍCH: API reference cho reranker classes
NỘI DUNG SẼ VIẾT:
- BaseReranker (abstract):
  - rerank(query: str, documents: List[str], top_k: int) -> List[RerankResult]
- RerankingConfig:
  - Fields: api_key, base_url, model, api_type, api_version
  - api_type options: pinecone, hf, onnx
- Implementations:
  - PineconeReranker (API-based)
  - HuggingFaceReranker (local)
  - ONNXReranker (local, optimized)
- RerankResult structure
-->
