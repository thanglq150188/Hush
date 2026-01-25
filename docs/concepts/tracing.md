# Tracing và Observability

## Tại sao cần Tracing?

Khi workflow trở nên phức tạp với nhiều nodes, iterations, và LLM calls, việc debug và optimize trở nên khó khăn:

- **Debug**: Xác định node nào gây lỗi? Input/output của mỗi node là gì?
- **Monitor**: Workflow chạy bao lâu? Node nào là bottleneck?
- **Cost tracking**: Tốn bao nhiêu tokens? Chi phí LLM calls?

Hush cung cấp hệ thống tracing để giải quyết các vấn đề này.

## Kiến trúc Tracing

```
┌─────────────────────────────────────────────────────────┐
│                      Hush Engine                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │                   Tracer                        │   │
│  │  • LocalTracer (built-in, SQLite)               │   │
│  │  • LangfuseTracer (cloud)                       │   │
│  │  • OTelTracer (OpenTelemetry)                   │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                          ↓
              ┌───────────────────────┐
              │    Trace Storage      │
              │  • SQLite (local)     │
              │  • Langfuse (cloud)   │
              │  • OTEL backends      │
              └───────────────────────┘
```

## LocalTracer (Built-in)

**LocalTracer** lưu traces vào SQLite - zero dependencies, chạy offline.

### Đặc điểm

- **SQLite storage**: Traces lưu tại `~/.hush/traces.db` (hoặc `HUSH_TRACES_DB` env var)
- **Zero dependencies**: Không cần external service
- **Web UI**: Xem traces qua built-in viewer
- **SQL queries**: Truy vấn traces trực tiếp bằng SQL

### Sử dụng

```python
from hush.core import Hush, GraphNode
from hush.core.tracers import LocalTracer

# Tạo tracer
tracer = LocalTracer(
    name="my-app",
    tags=["dev", "testing"]
)

# Sử dụng với workflow
engine = Hush(graph)
result = await engine.run(inputs={"query": "Hello"}, tracer=tracer)
```

### Xem Traces

```python
# Traces được lưu tại ~/.hush/traces.db
# Query trực tiếp bằng SQL:
import sqlite3

conn = sqlite3.connect("~/.hush/traces.db")
cursor = conn.execute("""
    SELECT * FROM traces
    WHERE request_id = ?
""", [result["$state"].request_id])

for row in cursor:
    print(row)
```

### Web UI

```bash
# Khởi động trace viewer
python -m hush.core.tracers.viewer

# Mở browser tại http://localhost:8000
```

## LangfuseTracer (Cloud)

**LangfuseTracer** gửi traces đến [Langfuse](https://langfuse.com) - cloud-based platform.

### Đặc điểm

- **Cloud-based**: Truy cập từ bất cứ đâu
- **Team collaboration**: Chia sẻ traces với team
- **Cost tracking**: Tự động tính chi phí LLM calls
- **Analytics**: Dashboard và metrics
- **Media support**: Upload images, files

### Cài đặt

```bash
pip install hush-ai[langfuse]
```

### Cấu hình resources.yaml

```yaml
langfuse:default:
  _class: LangfuseConfig
  secret_key: ${LANGFUSE_SECRET_KEY}
  public_key: ${LANGFUSE_PUBLIC_KEY}
  host: https://cloud.langfuse.com
  enabled: true
```

### Sử dụng

```python
from hush.core import Hush
from hush.observability import LangfuseTracer

# Cách 1: Sử dụng ResourceHub (recommended)
tracer = LangfuseTracer(
    resource_key="langfuse:default",
    tags=["production", "v1.0"]
)

# Cách 2: Direct config (không cần ResourceHub)
from hush.observability import LangfuseConfig

tracer = LangfuseTracer(
    config=LangfuseConfig(
        secret_key="sk-...",
        public_key="pk-...",
        host="https://cloud.langfuse.com"
    ),
    tags=["dev"]
)

# Sử dụng với workflow
engine = Hush(graph)
result = await engine.run(inputs={"query": "Hello"}, tracer=tracer)

# Trace URL được log tự động
# INFO: Langfuse trace created. View: https://cloud.langfuse.com/trace/xxx
```

## OTelTracer (OpenTelemetry)

**OTelTracer** export traces theo chuẩn OpenTelemetry - tích hợp với nhiều backends.

### Cài đặt

```bash
pip install hush-ai[otel]
```

### Cấu hình

```yaml
otel:jaeger:
  _class: OTELConfig
  endpoint: http://localhost:4317
  service_name: my-hush-app
  insecure: true
```

### Sử dụng

```python
from hush.observability import OTelTracer

tracer = OTelTracer(resource_key="otel:jaeger")
result = await engine.run(inputs={...}, tracer=tracer)
```

## Trace Data

Mỗi node execution được ghi lại với các thông tin:

| Field | Mô tả |
|-------|-------|
| `node_name` | Tên đầy đủ của node |
| `start_time` | Thời gian bắt đầu |
| `end_time` | Thời gian kết thúc |
| `duration_ms` | Thời gian chạy (ms) |
| `input_data` | Input của node |
| `output_data` | Output của node |
| `model` | Tên model (cho LLM nodes) |
| `usage` | Token usage (prompt_tokens, completion_tokens) |
| `cost` | Chi phí USD (cho LLM nodes) |
| `error` | Lỗi nếu có |

### Ví dụ Trace Output

```json
{
  "request_id": "abc-123",
  "workflow_name": "summarize-pipeline",
  "execution_order": [
    {"node": "summarize-pipeline", "parent": null},
    {"node": "summarize-pipeline.prompt", "parent": "summarize-pipeline"},
    {"node": "summarize-pipeline.llm", "parent": "summarize-pipeline", "contain_generation": true}
  ],
  "nodes_trace_data": {
    "summarize-pipeline.llm": {
      "name": "llm",
      "start_time": "2024-01-15T10:30:00Z",
      "end_time": "2024-01-15T10:30:02Z",
      "duration_ms": 2000,
      "input": {"messages": [...]},
      "output": {"content": "Tóm tắt..."},
      "model": "gpt-4o",
      "usage": {"prompt_tokens": 150, "completion_tokens": 50},
      "cost": 0.003
    }
  }
}
```

## Tags

Tags giúp phân loại và filter traces:

```python
# Static tags - set khi tạo tracer
tracer = LocalTracer(tags=["production", "v2.0", "customer-a"])

# Dynamic tags - thêm trong runtime
state.add_tag("cache-hit")
state.add_tags(["error", "retry"])
```

Filter trong Langfuse:
```
tags CONTAINS "production" AND tags CONTAINS "error"
```

## Media Attachments

Gửi images, files cùng với traces (Langfuse):

```python
from hush.core.tracers import MediaAttachment

# Trong node code
def process_image(image_bytes: bytes) -> dict:
    # Process image...

    # Attach image to trace
    attachment = MediaAttachment.from_bytes(
        content=image_bytes,
        content_type="image/png",
        attach_to="input"  # hoặc "output", "metadata"
    )

    return {
        "result": "...",
        "$media": attachment  # Special key cho media
    }
```

## Best Practices

### 1. Local dev với LocalTracer

```python
# Development - traces lưu local
from hush.core.tracers import LocalTracer

tracer = LocalTracer(tags=["dev", os.getenv("USER")])
```

### 2. Production với LangfuseTracer

```python
# Production - traces gửi cloud
from hush.observability import LangfuseTracer

tracer = LangfuseTracer(
    resource_key="langfuse:production",
    tags=["prod", VERSION]
)
```

### 3. Conditional tracing

```python
import os

if os.getenv("HUSH_TRACING") == "true":
    tracer = LangfuseTracer(resource_key="langfuse:default")
else:
    tracer = None

result = await engine.run(inputs={...}, tracer=tracer)
```

### 4. Request correlation

```python
# Truyền IDs từ request context
result = await engine.run(
    inputs={...},
    tracer=tracer,
    user_id=request.user.id,
    session_id=request.session.id
)

# Traces sẽ có user_id và session_id để filter
```

### 5. Trace sampling (high-traffic)

```python
import random

# Sample 10% of requests
if random.random() < 0.1:
    tracer = LangfuseTracer(resource_key="langfuse:default")
else:
    tracer = None
```

## Tạo Custom Tracer

Kế thừa `BaseTracer` để tạo tracer riêng:

```python
from hush.core.tracers import BaseTracer, register_tracer


@register_tracer
class MyCustomTracer(BaseTracer):
    """Custom tracer gửi traces đến service riêng."""

    def __init__(self, endpoint: str, tags: List[str] = None):
        super().__init__(tags=tags)
        self.endpoint = endpoint

    def _get_tracer_config(self) -> Dict[str, Any]:
        """Config được truyền cho subprocess."""
        return {"endpoint": self.endpoint}

    @staticmethod
    def flush(flush_data: Dict[str, Any]) -> None:
        """Chạy trong subprocess - gửi traces đến service."""
        import requests

        endpoint = flush_data["tracer_config"]["endpoint"]
        requests.post(endpoint, json=flush_data)
```

## Tiếp theo

- [LocalTracer Reference](../reference/observability/local-tracer.md) - API chi tiết
- [Langfuse Integration](../reference/observability/langfuse.md) - Cấu hình nâng cao
