# Xử lý lỗi

Hướng dẫn này sẽ giúp bạn xử lý lỗi và implement retry strategies trong Hush workflows.

## Error Handling Basics

### Errors được capture trong State

Khi một node gặp lỗi, error được lưu vào state thay vì crash workflow.

```python
from hush.core import GraphNode, CodeNode, START, END, PARENT

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
print(error)  # "division by zero"
```

### Kiểm tra lỗi trong workflow

```python
with GraphNode(name="check-error") as graph:
    risky = CodeNode(
        name="risky",
        code_fn=lambda: risky_operation(),
        inputs={}
    )

    # Check if error occurred
    check = CodeNode(
        name="check",
        code_fn=lambda error=None: {
            "has_error": error is not None,
            "error_msg": str(error) if error else None
        },
        inputs={"error": risky["error"]},  # Access error output
        outputs={"*": PARENT}
    )

    START >> risky >> check >> END
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

node = CodeNode(
    name="sqrt",
    code_fn=operation_with_error,
    inputs={"x": PARENT["x"]},
    outputs={"*": PARENT}
)
```

## LLMNode Fallback

LLMNode có built-in fallback support.

```python
from hush.providers import LLMNode

llm = LLMNode(
    name="llm",
    resource_key="gpt-4",
    fallback=["azure-gpt4", "gemini"],  # Fallback chain
    inputs={"messages": prompt["messages"]}
)
# Nếu gpt-4 fails → try azure-gpt4 → try gemini
```

### Fallback với different configs

```yaml
# resources.yaml
llm:primary:
  _class: OpenAIConfig
  api_key: ${OPENAI_API_KEY}
  model: gpt-4o
  base_url: https://api.openai.com/v1

llm:backup-azure:
  _class: AzureConfig
  api_key: ${AZURE_API_KEY}
  azure_endpoint: https://backup.openai.azure.com
  model: gpt-4

llm:backup-gemini:
  _class: GeminiConfig
  project_id: ${GCP_PROJECT}
  model: gemini-2.0-flash-001
```

```python
llm = LLMNode(
    name="llm",
    resource_key="primary",
    fallback=["backup-azure", "backup-gemini"],
    inputs={"messages": prompt["messages"]}
)
```

## Retry Pattern

### Manual retry với WhileLoopNode

```python
from hush.core.nodes.iteration.while_loop_node import WhileLoopNode

with GraphNode(name="retry-pattern") as graph:
    with WhileLoopNode(
        name="retry_loop",
        inputs={
            "attempt": 0,
            "max_attempts": 3,
            "success": False,
            "result": None
        },
        stop_condition="success == True or attempt >= max_attempts",
        max_iterations=5
    ) as loop:
        # Attempt operation
        attempt = CodeNode(
            name="attempt",
            code_fn=lambda attempt, data: {
                "new_attempt": attempt + 1,
                **try_operation(data)
            },
            inputs={
                "attempt": PARENT["attempt"],
                "data": PARENT["data"]
            }
        )

        attempt["new_attempt"] >> PARENT["attempt"]
        attempt["success"] >> PARENT["success"]
        attempt["result"] >> PARENT["result"]

        START >> attempt >> END

    loop["result"] >> PARENT["result"]
    loop["success"] >> PARENT["success"]
    loop["attempt"] >> PARENT["attempts_made"]

    START >> loop >> END
```

### Retry với exponential backoff

```python
import asyncio

async def operation_with_backoff(attempt, max_attempts, base_delay=1.0):
    """Retry với exponential backoff."""
    try:
        result = await risky_api_call()
        return {
            "success": True,
            "result": result,
            "new_attempt": attempt + 1
        }
    except Exception as e:
        if attempt < max_attempts - 1:
            # Exponential backoff: 1s, 2s, 4s, ...
            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)
        return {
            "success": False,
            "error": str(e),
            "new_attempt": attempt + 1
        }

retry_node = CodeNode(
    name="retry",
    code_fn=operation_with_backoff,
    inputs={
        "attempt": PARENT["attempt"],
        "max_attempts": PARENT["max_attempts"]
    }
)
```

## Conditional Error Handling với BranchNode

```python
from hush.core.nodes.flow.branch_node import Branch

with GraphNode(name="error-routing") as graph:
    operation = CodeNode(
        name="operation",
        code_fn=lambda data: try_process(data),
        inputs={"data": PARENT["data"]}
    )

    # Route based on error type
    router = (Branch("error_router")
        .if_(operation["error_type"] == "validation", "handle_validation")
        .if_(operation["error_type"] == "network", "handle_network")
        .if_(operation["error_type"] == "timeout", "handle_timeout")
        .otherwise("handle_success"))

    handle_validation = CodeNode(
        name="handle_validation",
        code_fn=lambda: {"status": "validation_error", "action": "fix_data"},
        inputs={}
    )

    handle_network = CodeNode(
        name="handle_network",
        code_fn=lambda: {"status": "network_error", "action": "retry"},
        inputs={}
    )

    handle_timeout = CodeNode(
        name="handle_timeout",
        code_fn=lambda: {"status": "timeout", "action": "retry_with_longer_timeout"},
        inputs={}
    )

    handle_success = CodeNode(
        name="handle_success",
        code_fn=lambda result: {"status": "success", "result": result},
        inputs={"result": operation["result"]}
    )

    # Merge results
    merge = CodeNode(
        name="merge",
        code_fn=lambda **kwargs: {"final": next(v for v in kwargs.values() if v)},
        inputs={
            "v1": handle_validation["status"],
            "v2": handle_network["status"],
            "v3": handle_timeout["status"],
            "v4": handle_success["status"]
        },
        outputs={"*": PARENT}
    )

    START >> operation >> router >> [handle_validation, handle_network, handle_timeout, handle_success]
    [handle_validation, handle_network, handle_timeout, handle_success] >> ~merge
    merge >> END
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

# Usage
node = CodeNode(
    name="with_fallback",
    code_fn=with_fallback(fetch_data, default_data),
    inputs={"query": PARENT["query"]}
)
```

### Graceful degradation pattern

```python
with GraphNode(name="graceful-degradation") as graph:
    # Primary operation
    primary = CodeNode(
        name="primary",
        code_fn=lambda query: fetch_from_api(query),
        inputs={"query": PARENT["query"]}
    )

    # Check if primary failed
    check = CodeNode(
        name="check",
        code_fn=lambda result, error: {
            "use_fallback": error is not None or result is None
        },
        inputs={
            "result": primary["result"],
            "error": primary["error"]
        }
    )

    # Branch based on success
    router = (Branch("fallback_router")
        .if_(check["use_fallback"] == True, "fallback")
        .otherwise("use_primary"))

    # Fallback: use cached/default data
    fallback = CodeNode(
        name="fallback",
        code_fn=lambda query: {"result": get_cached_data(query)},
        inputs={"query": PARENT["query"]}
    )

    # Use primary result
    use_primary = CodeNode(
        name="use_primary",
        code_fn=lambda result: {"result": result},
        inputs={"result": primary["result"]}
    )

    # Merge
    merge = CodeNode(
        name="merge",
        code_fn=lambda r1=None, r2=None: {"final": r1 if r1 else r2},
        inputs={
            "r1": fallback["result"],
            "r2": use_primary["result"]
        },
        outputs={"*": PARENT}
    )

    START >> primary >> check >> router >> [fallback, use_primary]
    [fallback, use_primary] >> ~merge
    merge >> END
```

## Error Logging và Tracing

### Với LocalTracer

```python
from hush.core.tracers import LocalTracer

tracer = LocalTracer(tags=["production", "error-monitoring"])
result = await engine.run(inputs={...}, tracer=tracer)

# Errors được log trong traces.db
# Query: SELECT * FROM traces WHERE error IS NOT NULL
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

### Custom error logging

```python
from hush.core import LOGGER

def logged_operation(data):
    try:
        result = process(data)
        LOGGER.info(f"Operation succeeded: {result}")
        return {"success": True, "result": result}
    except Exception as e:
        LOGGER.error(f"Operation failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
```

## Best Practices

### 1. Fail fast, recover gracefully

```python
# Check inputs early
def validate_and_process(data):
    if not data:
        return {"error": "Empty data", "result": None}
    if not isinstance(data, dict):
        return {"error": "Invalid data type", "result": None}

    # Process valid data
    return {"error": None, "result": process(data)}
```

### 2. Specific error handling

```python
def handle_api_error(response):
    if response.status_code == 429:
        return {"action": "retry_after_delay", "delay": response.headers.get("Retry-After", 60)}
    elif response.status_code == 503:
        return {"action": "use_fallback"}
    elif response.status_code >= 500:
        return {"action": "retry"}
    else:
        return {"action": "fail", "error": response.text}
```

### 3. Always have fallback for critical paths

```python
llm = LLMNode(
    resource_key="gpt-4",
    fallback=["azure-gpt4", "gemini"],  # Multiple fallbacks
    inputs={...}
)
```

### 4. Log context với errors

```python
def operation_with_context(data, request_id):
    try:
        return {"result": process(data)}
    except Exception as e:
        LOGGER.error(
            f"Operation failed",
            extra={
                "request_id": request_id,
                "data_size": len(str(data)),
                "error_type": type(e).__name__
            }
        )
        raise
```

### 5. Set timeouts

```python
import asyncio

async def with_timeout(coro, timeout_seconds=30):
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return {"error": "Operation timed out", "result": None}
```

## Tiếp theo

- [Tích hợp LLM](llm-integration.md) - LLM fallback và retry
- [Thực thi song song](parallel-execution.md) - Error handling trong parallel execution
- [Deploy Production](production-deployment.md) - Production error monitoring
