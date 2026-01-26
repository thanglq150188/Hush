# Hush Providers

LLM, embedding và reranking providers cho Hush workflow engine.

## Cài đặt

```bash
pip install hush-providers

# Với provider cụ thể
pip install hush-providers[openai]      # OpenAI
pip install hush-providers[azure]       # Azure OpenAI
pip install hush-providers[gemini]      # Google Gemini

# Local inference (nhẹ)
pip install hush-providers[onnx]        # ONNX Runtime
pip install hush-providers[embeddings]  # Embeddings (ONNX)
pip install hush-providers[rerankers]   # Rerankers (ONNX)

# Tất cả
pip install hush-providers[all-light]   # Không có ML frameworks nặng
pip install hush-providers[all]         # Bao gồm transformers/torch
```

## Quick Start

### LLM Node

```python
from hush.core import Hush, GraphNode, START, END, PARENT
from hush.providers import PromptNode, LLMNode

async def main():
    with GraphNode(name="chat") as graph:
        prompt = PromptNode(
            name="prompt",
            inputs={
                "prompt": {"system": "Bạn là trợ lý AI.", "user": "{question}"},
                "question": PARENT["question"]
            },
            outputs={"messages": PARENT}
        )
        llm = LLMNode(
            name="llm",
            resource_key="gpt-4o",
            inputs={"messages": PARENT["messages"]},
            outputs={"content": PARENT["answer"]}
        )
        START >> prompt >> llm >> END

    engine = Hush(graph)
    result = await engine.run(inputs={"question": "Hello!"})
```

### Embedding Node

```python
from hush.providers import EmbeddingNode

embed = EmbeddingNode(
    name="embed",
    resource_key="bge-m3",
    inputs={"texts": PARENT["documents"]},
    outputs={"embeddings": PARENT}
)
```

### Rerank Node

```python
from hush.providers import RerankNode

rerank = RerankNode(
    name="rerank",
    resource_key="bge-reranker",
    inputs={"query": PARENT["query"], "documents": PARENT["docs"]},
    outputs={"ranked_docs": PARENT}
)
```

## Supported Providers

### LLM

| Provider | API Type | Mô tả |
|----------|----------|-------|
| OpenAI | `openai` | OpenAI API và compatible endpoints (vLLM) |
| Azure | `azure` | Azure OpenAI Service |
| Gemini | `gemini` | Google Gemini via Vertex AI |

### Embedding

| Provider | API Type | Mô tả |
|----------|----------|-------|
| vLLM | `vllm` | OpenAI-compatible embedding API |
| TEI | `tei` | HuggingFace Text Embedding Inference |
| ONNX | `onnx` | Local ONNX Runtime |

### Reranking

| Provider | API Type | Mô tả |
|----------|----------|-------|
| vLLM | `vllm` | OpenAI-compatible reranking API |
| Pinecone | `pinecone` | Pinecone reranking API |
| ONNX | `onnx` | Local ONNX Runtime |

## Cấu hình với ResourceHub

```yaml
# resources.yaml
llm:gpt-4o:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  api_type: openai
  base_url: https://api.openai.com/v1
  model: gpt-4o

embedding:bge-m3:
  _class: EmbeddingConfig
  api_type: vllm
  base_url: http://localhost:8000/v1
  model: BAAI/bge-m3
```

## Tính năng

- Streaming và non-streaming responses
- Token counting và usage tracking
- Multimodal input (images)
- Tool/function calling
- Batch processing
- YAML configuration

## License

MIT
