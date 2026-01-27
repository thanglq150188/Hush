# Tracing và Observability

Hướng dẫn sử dụng hệ thống tracing để debug và monitor workflows.

## Tại sao cần Tracing?

Khi workflow trở nên phức tạp với nhiều nodes, iterations, và LLM calls:

- **Debug**: Xác định node nào gây lỗi? Input/output của mỗi node là gì?
- **Monitor**: Workflow chạy bao lâu? Node nào là bottleneck?
- **Cost tracking**: Tốn bao nhiêu tokens? Chi phí LLM calls?

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
```

## LocalTracer (Built-in)

Lưu traces vào SQLite - zero dependencies, chạy offline.

### Đặc điểm

- **SQLite storage**: Traces lưu tại `~/.hush/traces.db`
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
# Query trực tiếp bằng SQL
import sqlite3

conn = sqlite3.connect("~/.hush/traces.db")
cursor = conn.execute("""
    SELECT * FROM traces
    WHERE request_id = ?
""", [result["$state"].request_id])

for row in cursor:
    print(row)
```

## LangfuseTracer (Cloud)

Gửi traces đến [Langfuse](https://langfuse.com) - cloud-based platform.

### Đặc điểm

- **Cloud-based**: Truy cập từ bất cứ đâu
- **Team collaboration**: Chia sẻ traces với team
- **Cost tracking**: Tự động tính chi phí LLM calls
- **Analytics**: Dashboard và metrics

### Cấu hình

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
from hush.observability import LangfuseTracer

tracer = LangfuseTracer(
    resource_key="langfuse:default",
    tags=["production", "v1.0"]
)

result = await engine.run(inputs={"query": "Hello"}, tracer=tracer)
# Trace URL được log tự động
```

## OTelTracer (OpenTelemetry)

Export traces theo chuẩn OpenTelemetry.

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

Mỗi node execution được ghi lại với:

| Field | Mô tả |
|-------|-------|
| `node_name` | Tên đầy đủ của node |
| `start_time` | Thời gian bắt đầu |
| `end_time` | Thời gian kết thúc |
| `duration_ms` | Thời gian chạy (ms) |
| `input_data` | Input của node |
| `output_data` | Output của node |
| `model` | Tên model (cho LLM nodes) |
| `usage` | Token usage |
| `cost` | Chi phí USD |
| `error` | Lỗi nếu có |

## Tags

Tags giúp phân loại và filter traces:

```python
# Static tags - set khi tạo tracer
tracer = LocalTracer(tags=["production", "v2.0", "customer-a"])

# Dynamic tags - thêm trong runtime
state.add_tag("cache-hit")
state.add_tags(["error", "retry"])
```

Hoặc trong CodeNode:

```python
def process(data):
    result = process_data(data)
    if result.get("from_cache"):
        return {"result": result, "$tags": ["cache-hit"]}
    return {"result": result}
```

## Request Correlation

Truyền IDs từ request context để dễ filter:

```python
result = await engine.run(
    inputs={...},
    tracer=tracer,
    user_id=request.user.id,
    session_id=request.session.id
)
```

## Conditional Tracing

### Environment-based

```python
import os

if os.getenv("HUSH_TRACING") == "true":
    tracer = LangfuseTracer(resource_key="langfuse:default")
else:
    tracer = None

result = await engine.run(inputs={...}, tracer=tracer)
```

### Sampling

```python
import random

# Sample 10% of requests
if random.random() < 0.1:
    tracer = LangfuseTracer(resource_key="langfuse:default")
else:
    tracer = None
```

## Best Practices

1. **Local dev với LocalTracer**
2. **Production với LangfuseTracer**
3. **Conditional tracing** để control costs
4. **Request correlation** với user_id, session_id
5. **Sampling** cho high-traffic services

## Xem thêm

- [Tutorial: Production](../tutorials/04-production.md)
- [Xử lý lỗi](error-handling.md)
