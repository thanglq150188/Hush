# Embedding Providers

<!--
MỤC ĐÍCH: API reference cho embedding classes
NỘI DUNG SẼ VIẾT:
- BaseEmbedding (abstract):
  - embed(text: str) -> List[float]
  - embed_batch(texts: List[str]) -> List[List[float]]
- EmbeddingConfig:
  - Fields: api_key, base_url, model, api_type, dimensions
  - api_type options: vllm, hf, onnx
- Implementations:
  - VLLMEmbedding (API-based)
  - HuggingFaceEmbedding (local)
  - ONNXEmbedding (local, optimized)
-->
