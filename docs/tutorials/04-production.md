# Tutorial 4: Production

Tutorial này hướng dẫn cách đưa Hush workflows vào production: tracing, error handling, và deployment.

## Tracing và Observability

### LocalTracer - Development

`LocalTracer` lưu traces vào SQLite - zero dependencies, chạy offline.

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
result = await engine.run(
    inputs={"query": "Hello"},
    tracer=tracer,
    user_id="user-123",
    session_id="session-456"
)

# Traces lưu tại ~/.hush/traces.db
```

### LangfuseTracer - Production

`LangfuseTracer` gửi traces đến [Langfuse](https://langfuse.com) - cloud-based platform.

#### Cấu hình

```yaml
# resources.yaml
langfuse:default:
  _class: LangfuseConfig
  secret_key: ${LANGFUSE_SECRET_KEY}
  public_key: ${LANGFUSE_PUBLIC_KEY}
  host: https://cloud.langfuse.com
  enabled: true
```

#### Sử dụng

```python
from hush.observability import LangfuseTracer

# Sử dụng ResourceHub
tracer = LangfuseTracer(
    resource_key="langfuse:default",
    tags=["production", "v1.0"]
)

result = await engine.run(inputs={"query": "Hello"}, tracer=tracer)
# Trace URL được log tự động
```

### Dynamic Tags

Tags giúp phân loại và filter traces:

```python
# Static tags - set khi tạo tracer
tracer = LocalTracer(tags=["production", "v2.0"])

# Dynamic tags - thêm trong runtime (trong CodeNode)
def process(data: dict) -> dict:
    result = process_data(data)

    # Thêm tag dựa trên kết quả
    if result.get("from_cache"):
        return {"result": result, "$tags": ["cache-hit"]}
    return {"result": result}
```

### Cost Tracking

```yaml
# resources.yaml
llm:gpt-4o:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  cost_per_input_token: 0.000005   # $5 per 1M input
  cost_per_output_token: 0.000015  # $15 per 1M output
```

```python
# Cost được track tự động trong traces
result = await engine.run(inputs={...}, tracer=tracer)

# Với LangfuseTracer: cost hiển thị trên dashboard
# Với LocalTracer: query trực tiếp từ SQLite
```

## Error Handling

### Try-Except trong CodeNode

```python
def safe_process(data: dict) -> dict:
    try:
        result = risky_operation(data)
        return {"result": result, "error": None}
    except Exception as e:
        return {"result": None, "error": str(e)}

process = CodeNode(
    name="safe_process",
    code_fn=safe_process,
    inputs={"data": PARENT["data"]},
    outputs={"result": PARENT, "error": PARENT}
)
```

### Branch-based Error Handling

```python
with GraphNode(name="with-error-handling") as graph:
    # Main processing
    process = CodeNode(
        name="process",
        code_fn=lambda data: {"result": process_data(data), "success": True},
        inputs={"data": PARENT["data"]},
        outputs={"result": PARENT, "success": PARENT}
    )

    # Check success
    branch = BranchNode(
        name="check_error",
        cases={"success == True": "success_path"},
        default="error_path",
        inputs={"success": PARENT["success"]}
    )

    # Success path
    finalize = CodeNode(
        name="finalize",
        code_fn=lambda result: {"output": f"Success: {result}"},
        inputs={"result": PARENT["result"]},
        outputs={"output": PARENT}
    )

    # Error path
    handle_error = CodeNode(
        name="handle_error",
        code_fn=lambda: {"output": "Error occurred, using fallback"},
        outputs={"output": PARENT}
    )

    START >> process >> branch
    branch >> [finalize, handle_error]
    [finalize, handle_error] >> ~END
```

### LLM Fallback

```python
from hush.providers import LLMNode

llm = LLMNode(
    name="llm",
    resource_key="gpt-4o",
    fallback=["azure-gpt4", "gemini"],  # Fallback chain
    inputs={"messages": PARENT["messages"]}
)
# Nếu gpt-4o fails → try azure-gpt4 → try gemini
```

## Retry Pattern

```python
import asyncio

async def fetch_with_retry(url: str, max_retries: int = 3) -> dict:
    """Fetch với retry logic."""
    last_error = None

    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status == 200:
                        return {"data": await resp.json(), "success": True}
                    last_error = f"HTTP {resp.status}"
        except Exception as e:
            last_error = str(e)

        # Exponential backoff
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)

    return {"data": None, "success": False, "error": last_error}

fetch = CodeNode(
    name="fetch",
    code_fn=fetch_with_retry,
    inputs={"url": PARENT["url"]},
    outputs={"*": PARENT}
)
```

## Cấu hình Production

### Environment Variables

```bash
# Required
export OPENAI_API_KEY=sk-...
export HUSH_CONFIG=/path/to/resources.yaml

# Tracing
export LANGFUSE_SECRET_KEY=sk-...
export LANGFUSE_PUBLIC_KEY=pk-...

# Optional
export HUSH_LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
export HUSH_TRACES_DB=/path/to/traces.db
```

### resources.yaml cho Production

```yaml
# LLM - Primary
llm:primary:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  cost_per_input_token: 0.000005
  cost_per_output_token: 0.000015

# LLM - Fallback
llm:fallback:
  _class: OpenAIConfig
  api_key: ${AZURE_API_KEY}
  api_type: azure
  azure_endpoint: ${AZURE_ENDPOINT}
  model: gpt-4-deployment

# Observability
langfuse:production:
  _class: LangfuseConfig
  secret_key: ${LANGFUSE_SECRET_KEY}
  public_key: ${LANGFUSE_PUBLIC_KEY}
  host: https://cloud.langfuse.com
  enabled: true
```

## Conditional Tracing

```python
import os
import random

# Enable tracing based on environment
if os.getenv("ENABLE_TRACING") == "true":
    tracer = LangfuseTracer(resource_key="langfuse:production")
else:
    tracer = None

# Sampling - trace 10% of requests
if random.random() < 0.1:
    tracer = LangfuseTracer(resource_key="langfuse:production")
else:
    tracer = None

result = await engine.run(inputs={...}, tracer=tracer)
```

## Ví dụ Production-Ready Workflow

```python
import asyncio
import os
from hush.core import Hush, GraphNode, CodeNode, BranchNode, START, END, PARENT
from hush.core.tracers import LocalTracer
from hush.providers import PromptNode, LLMNode

# Chọn tracer dựa trên environment
def get_tracer():
    env = os.getenv("ENV", "development")
    if env == "production":
        from hush.observability import LangfuseTracer
        return LangfuseTracer(
            resource_key="langfuse:production",
            tags=["production", os.getenv("VERSION", "unknown")]
        )
    return LocalTracer(name="dev", tags=["development"])


async def main():
    with GraphNode(name="production-chat") as graph:
        # Validate input
        validate = CodeNode(
            name="validate",
            code_fn=lambda query: {
                "valid": len(query.strip()) > 0,
                "query": query.strip()
            },
            inputs={"query": PARENT["query"]},
            outputs={"valid": PARENT, "query": PARENT}
        )

        # Branch: valid input?
        check_valid = BranchNode(
            name="check_valid",
            cases={"valid == True": "process"},
            default="reject",
            inputs={"valid": PARENT["valid"]}
        )

        # Process path
        prompt = PromptNode(
            name="prompt",
            inputs={
                "prompt": {
                    "system": "Bạn là assistant hữu ích. Trả lời ngắn gọn.",
                    "user": "{query}"
                },
                "query": PARENT["query"]
            }
        )

        llm = LLMNode(
            name="llm",
            resource_key="primary",
            fallback=["fallback"],  # Auto fallback
            inputs={
                "messages": prompt["messages"],
                "temperature": 0.7,
                "max_tokens": 500
            },
            outputs={"content": PARENT["response"]}
        )

        # Reject path
        reject = CodeNode(
            name="reject",
            code_fn=lambda: {"response": "Invalid input. Please provide a valid query."},
            outputs={"response": PARENT}
        )

        # Kết nối
        START >> validate >> check_valid
        check_valid >> prompt >> llm
        check_valid >> reject
        [llm, reject] >> ~END

    # Create engine
    engine = Hush(graph)

    # Get tracer
    tracer = get_tracer()

    # Run with tracing
    result = await engine.run(
        inputs={"query": "Python là gì?"},
        tracer=tracer,
        user_id="user-123",
        session_id="session-456"
    )

    print(f"Response: {result['response']}")


if __name__ == "__main__":
    asyncio.run(main())
```

## Deployment Checklist

- [ ] Cấu hình `resources.yaml` với tất cả providers
- [ ] Set environment variables cho API keys
- [ ] Enable tracing (Langfuse hoặc OTEL)
- [ ] Cấu hình fallback cho LLM nodes
- [ ] Implement error handling
- [ ] Set up cost tracking
- [ ] Configure logging level
- [ ] Test với các edge cases

## Tổng kết

| Aspect | Development | Production |
|--------|-------------|------------|
| Tracing | LocalTracer | LangfuseTracer / OTelTracer |
| Logging | DEBUG | INFO / WARNING |
| Error handling | Basic | Comprehensive với fallback |
| Cost tracking | Optional | Required |
| Sampling | 100% | 10-100% tùy traffic |

## Tiếp theo

- [Guide: Tracing](../guides/tracing.md) - Chi tiết về observability
- [Guide: Error Handling](../guides/error-handling.md) - Patterns nâng cao
