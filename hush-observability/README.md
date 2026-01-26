# Hush Observability

Observability framework cho Hush workflows - hỗ trợ nhiều backend providers.

## Cài đặt

```bash
# Với Langfuse
pip install hush-observability[langfuse]

# Với tất cả backends
pip install hush-observability[all]

# Backends cụ thể
pip install hush-observability[phoenix,opik]
```

## Quick Start

```python
from hush.core import RESOURCE_HUB
from hush.observability import LangfuseConfig

# Đăng ký tracer config
config = LangfuseConfig(
    public_key="pk-...",
    secret_key="sk-...",
    host="https://cloud.langfuse.com"
)
RESOURCE_HUB.register(config, registry_key="tracer:langfuse")

# Sử dụng với workflow
from hush.core import Hush, GraphNode, START, END

with GraphNode(name="demo", tracer_key="langfuse") as graph:
    # ... định nghĩa nodes
    pass

engine = Hush(graph)
await engine.run()  # Traces tự động gửi đến Langfuse
```

## Supported Backends

| Backend | Config Class | Mô tả |
|---------|--------------|-------|
| Langfuse | `LangfuseConfig` | LLM observability platform |
| Phoenix | `PhoenixConfig` | Arize Phoenix (self-hosted) |
| Opik | `OpikConfig` | Comet Opik |
| LangSmith | `LangSmithConfig` | LangChain tracing |

## Cấu hình YAML

```yaml
# resources.yaml
tracer:langfuse:
  _class: LangfuseConfig
  public_key: ${LANGFUSE_PUBLIC_KEY}
  secret_key: ${LANGFUSE_SECRET_KEY}
  host: https://cloud.langfuse.com

tracer:phoenix:
  _class: PhoenixConfig
  endpoint: http://localhost:6006
```

## Tính năng

- **Unified interface**: Một API cho tất cả backends
- **Hierarchical tracing**: Traces với parent-child relationships
- **Token tracking**: Tự động track token usage và cost
- **Async-first**: Buffering và batching cho performance
- **ResourceHub integration**: Quản lý như resource bình thường

## Kiến trúc

```
┌─────────────────────────────────────┐
│         Your Application            │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│      BaseTracer Interface           │
│  (add_span, add_generation, flush)  │
└──────────────┬──────────────────────┘
               │
       ┌───────┴────────┬─────────────┐
       │                │             │
┌──────▼──────┐  ┌──────▼──────┐  ┌──▼───┐
│  Langfuse   │  │   Phoenix   │  │ Opik │
└─────────────┘  └─────────────┘  └──────┘
```

## Advanced Usage

### Hierarchical Tracing

```python
tracer.add_span(request_id="req-1", name="root", parent=None)
tracer.add_generation(request_id="req-1", name="llm", parent="root")
tracer.add_span(request_id="req-1", name="postprocess", parent="llm")

await tracer.flush("req-1")  # Flush theo đúng hierarchy
```

### Custom Metadata

```python
tracer.add_span(
    request_id="req-1",
    name="process",
    input={"data": "..."},
    user_id="user-123",
    session_id="session-abc",
    tags=["production", "v2"],
    metadata={"environment": "prod"}
)
```

## License

MIT
