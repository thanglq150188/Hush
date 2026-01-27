# Xử lý lỗi

Hướng dẫn xử lý lỗi và implement retry strategies trong Hush workflows.

## Error Handling Basics

### Errors được capture trong State

Khi một node gặp lỗi, error được lưu vào state thay vì crash workflow.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END

with GraphNode(name="error-demo") as graph:
    failing_node = CodeNode(
        name="failing",
        code_fn=lambda: 1/0,  # Division by zero
        inputs={}
    )
    START >> failing_node >> END

engine = Hush(graph)
result = await engine.run(inputs={})

# Access error from state
state = result["$state"]
error = state["error-demo.failing", "error", None]
print(error)  # Traceback ...
```

## Try/Catch trong CodeNode

### Pattern 1: Internal try/catch

```python
def safe_operation(data):
    try:
        result = process(data)
        return {"success": True, "result": result}
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error: {e}"}

safe_node = CodeNode(
    name="safe",
    code_fn=safe_operation,
    inputs={"data": PARENT["data"]},
    outputs={"*": PARENT}
)
```

### Pattern 2: Error output key

```python
def operation_with_error(x):
    if x < 0:
        return {"result": None, "error": "x must be non-negative"}
    return {"result": x ** 0.5, "error": None}
```

## LLMNode Fallback

LLMNode có built-in fallback support.

```python
from hush.providers import LLMNode

llm = LLMNode(
    name="llm",
    resource_key="gpt-4o",
    fallback=["azure-gpt4", "gemini"],
    inputs={"messages": prompt["messages"]}
)
# Nếu gpt-4o fails → try azure-gpt4 → try gemini
```

### Cấu hình fallback

```yaml
llm:primary:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o

llm:backup-azure:
  _class: OpenAIConfig
  api_key: ${AZURE_API_KEY}
  api_type: azure
  azure_endpoint: https://backup.openai.azure.com
  model: gpt-4

llm:backup-gemini:
  _class: GeminiConfig
  project_id: ${GCP_PROJECT}
  model: gemini-2.0-flash-001
```

## Retry với Exponential Backoff

```python
import asyncio

async def operation_with_backoff(data, max_attempts=3, base_delay=1.0):
    """Retry với exponential backoff."""
    last_error = None

    for attempt in range(max_attempts):
        try:
            result = await risky_api_call(data)
            return {"success": True, "result": result}
        except Exception as e:
            last_error = str(e)
            if attempt < max_attempts - 1:
                delay = base_delay * (2 ** attempt)
                await asyncio.sleep(delay)

    return {"success": False, "error": last_error}
```

## Conditional Error Handling với BranchNode

```python
from hush.core import BranchNode

with GraphNode(name="error-routing") as graph:
    operation = CodeNode(
        name="operation",
        code_fn=lambda data: try_process(data),
        inputs={"data": PARENT["data"]}
    )

    # Route based on success
    router = BranchNode(
        name="router",
        cases={"success == True": "handle_success"},
        default="handle_error",
        inputs={"success": operation["success"]}
    )

    handle_success = CodeNode(
        name="handle_success",
        code_fn=lambda result: {"output": result},
        inputs={"result": operation["result"]},
        outputs={"output": PARENT}
    )

    handle_error = CodeNode(
        name="handle_error",
        code_fn=lambda error: {"output": f"Error: {error}"},
        inputs={"error": operation["error"]},
        outputs={"output": PARENT}
    )

    START >> operation >> router
    router >> [handle_success, handle_error]
    [handle_success, handle_error] >> ~END
```

## Graceful Degradation

Trả về kết quả mặc định khi operation fails.

```python
def with_fallback(primary_fn, fallback_value):
    """Wrapper để add fallback value."""
    async def wrapped(**kwargs):
        try:
            return await primary_fn(**kwargs)
        except Exception as e:
            return {"result": fallback_value, "used_fallback": True, "error": str(e)}
    return wrapped
```

## Error Logging và Tracing

### Với LocalTracer

```python
from hush.core.tracers import LocalTracer

tracer = LocalTracer(tags=["production", "error-monitoring"])
result = await engine.run(inputs={...}, tracer=tracer)

# Errors được log trong traces.db
```

### Với LangfuseTracer

```python
from hush.observability import LangfuseTracer

tracer = LangfuseTracer(
    resource_key="langfuse:default",
    tags=["production"]
)
result = await engine.run(inputs={...}, tracer=tracer)

# Errors hiển thị trong Langfuse dashboard
```

## Best Practices

1. **Fail fast, recover gracefully** - Check inputs early
2. **Specific error handling** - Handle different error types differently
3. **Always have fallback for critical paths**
4. **Log context với errors**
5. **Set timeouts** cho external calls

```python
import asyncio

async def with_timeout(coro, timeout_seconds=30):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return {"error": "Operation timed out", "result": None}
```

## Xem thêm

- [Tích hợp LLM](llm-integration.md) - LLM fallback
- [Thực thi song song](parallel-execution.md) - Error handling trong parallel
- [Tutorial: Production](../tutorials/04-production.md)
