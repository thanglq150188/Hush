# Cài đặt

## Yêu cầu hệ thống

- Python >= 3.10
- pip hoặc uv (khuyến nghị)

## Cài đặt nhanh

```bash
# Cài đặt với pip
pip install hush-ai[standard]

# Hoặc với uv (nhanh hơn)
uv add hush-ai[standard]
```

## Các tier cài đặt

Hush cung cấp nhiều tier cài đặt tùy theo nhu cầu:

| Tier | Mô tả | Use case |
|------|-------|----------|
| `hush-ai[core]` | Workflow engine + local tracing | Development, testing |
| `hush-ai[standard]` | Core + LLM providers (OpenAI) | Production cơ bản |
| `hush-ai[all]` | Standard + tất cả providers + observability | Production đầy đủ |
| `hush-ai[full]` | All + ML frameworks (torch, transformers) | Local inference |

### Core - Chỉ workflow engine

```bash
pip install hush-ai[core]
```

Bao gồm:
- Hush workflow engine
- **Graph nodes**: GraphNode
- **Transform nodes**: CodeNode, ParserNode
- **Flow control**: BranchNode
- **Iteration**: ForLoopNode, MapNode, WhileLoopNode, AsyncIterNode
- LocalTracer với SQLite storage và Web UI
- ❌ Không có LLM/AI nodes (PromptNode, LLMNode, EmbeddingNode...)

### Standard - Production cơ bản

```bash
pip install hush-ai[standard]
```

Bao gồm tất cả của `core` + :
- **AI nodes**: PromptNode, LLMNode, EmbeddingNode, RerankNode, LLMChainNode
- OpenAI SDK (GPT-4, GPT-4o, etc.)
- Azure OpenAI support
- Embedding và reranking providers

### All - Production đầy đủ

```bash
pip install hush-ai[all]
```

Bao gồm tất cả của `standard` + :
- Google Gemini
- AWS Bedrock
- ONNX Runtime (local inference nhẹ)
- Langfuse tracing
- OpenTelemetry

### Full - Với ML frameworks

```bash
pip install hush-ai[full]
```

Bao gồm tất cả của `all` + :
- PyTorch
- Transformers (HuggingFace)
- Sentence-transformers

⚠️ **Lưu ý**: Tier `full` có dung lượng lớn (~2GB+), chỉ cài khi cần local inference với HuggingFace models.

## Cài đặt từng provider

Nếu chỉ cần một số providers cụ thể:

```bash
# LLM providers
pip install hush-ai[openai]       # OpenAI
pip install hush-ai[azure]        # Azure OpenAI
pip install hush-ai[gemini]       # Google Gemini
pip install hush-ai[bedrock]      # AWS Bedrock

# Local inference
pip install hush-ai[onnx]         # ONNX Runtime (nhẹ, nhanh)
pip install hush-ai[huggingface]  # HuggingFace (nặng, nhiều models)

# Observability
pip install hush-ai[langfuse]     # Langfuse tracing
pip install hush-ai[otel]         # OpenTelemetry

# Kết hợp nhiều extras
pip install hush-ai[openai,gemini,langfuse]
```

## Cài đặt từ source (Development)

```bash
# Clone repo
git clone https://github.com/your-org/hush.git
cd hush

# Cài đặt với uv (khuyến nghị)
uv sync

# Hoặc cài từng package
pip install -e hush-core/
pip install -e hush-providers/
pip install -e hush-observability/
```

## Cấu hình môi trường

### Biến môi trường HUSH_CONFIG

Hush sử dụng file `resources.yaml` để cấu hình các resources (LLM, database, etc.). Đặt biến môi trường `HUSH_CONFIG` để chỉ định đường dẫn:

```bash
# Linux/macOS
export HUSH_CONFIG=/path/to/your/resources.yaml

# Windows
set HUSH_CONFIG=C:\path\to\your\resources.yaml
```

Nếu không đặt `HUSH_CONFIG`, Hush sẽ tìm config theo thứ tự:
1. `./resources.yaml` (thư mục hiện tại)
2. `~/.hush/resources.yaml` (thư mục home)

### Tạo file resources.yaml

Tạo file `resources.yaml` với cấu trúc sau:

```yaml
# LLM Configuration
llm:gpt-4o:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  api_type: openai
  base_url: https://api.openai.com/v1
  model: gpt-4o

# Embedding Configuration
embedding:bge-m3:
  _class: EmbeddingConfig
  api_key: ${EMBEDDING_API_KEY}
  api_type: vllm
  base_url: https://api.deepinfra.com/v1/openai/embeddings
  model: BAAI/bge-m3
  dimensions: 1024

# Observability
langfuse:default:
  _class: LangfuseConfig
  secret_key: ${LANGFUSE_SECRET_KEY}
  public_key: ${LANGFUSE_PUBLIC_KEY}
  host: https://cloud.langfuse.com
  enabled: true
```

**Lưu ý**: Sử dụng syntax `${VAR}` hoặc `${VAR:default}` để interpolate biến môi trường:
- `${OPENAI_API_KEY}` - bắt buộc, warning nếu không set
- `${REDIS_HOST:localhost}` - optional, dùng `localhost` nếu không set

## Xác nhận cài đặt

Chạy script sau để xác nhận cài đặt thành công:

```python
import asyncio
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT

async def main():
    # Tạo workflow đơn giản
    with GraphNode(name="hello-hush") as graph:
        hello = CodeNode(
            name="hello",
            code_fn=lambda: {"message": "Hello from Hush!"},
            outputs={"message": PARENT}
        )
        START >> hello >> END

    # Chạy workflow
    engine = Hush(graph)
    result = await engine.run(inputs={})

    print(f"✓ Hush đã cài đặt thành công!")
    print(f"  Kết quả: {result['message']}")

asyncio.run(main())
```

Output mong đợi:
```
✓ Hush đã cài đặt thành công!
  Kết quả: Hello from Hush!
```

## Tiếp theo

- [Quickstart](quickstart.md) - Bắt đầu trong 5 phút
- [Workflow đầu tiên](first-workflow.md) - Tutorial chi tiết
