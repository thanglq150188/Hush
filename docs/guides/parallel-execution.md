# Thực thi song song

Hướng dẫn này sẽ giúp bạn tận dụng parallel execution để tăng performance của workflows.

## Khi nào dùng Parallel Execution?

Parallel execution phù hợp khi:

1. **Independent operations**: Các tasks không phụ thuộc nhau
2. **I/O-bound operations**: API calls, database queries, file reads
3. **Batch processing**: Xử lý nhiều items cùng lúc
4. **Multiple LLM calls**: Gọi nhiều models hoặc nhiều prompts

**Không nên dùng** khi:
- Tasks có dependency tuần tự
- CPU-bound operations (Python GIL limitation)
- Rate-limited APIs (cần control concurrency)

## Syntax Parallel Branches

### Basic: List syntax

```python
from hush.core import GraphNode, CodeNode, START, END, PARENT

with GraphNode(name="parallel-demo") as graph:
    prepare = CodeNode(
        name="prepare",
        code_fn=lambda x: {"value": x},
        inputs={"x": PARENT["x"]}
    )

    # Parallel branches using [list] syntax
    branch_a = CodeNode(
        name="branch_a",
        code_fn=lambda x: {"result": x * 2},
        inputs={"x": prepare["value"]}
    )
    branch_b = CodeNode(
        name="branch_b",
        code_fn=lambda x: {"result": x * 3},
        inputs={"x": prepare["value"]}
    )

    merge = CodeNode(
        name="merge",
        code_fn=lambda a, b: {"total": a + b},
        inputs={
            "a": branch_a["result"],
            "b": branch_b["result"]
        },
        outputs={"*": PARENT}
    )

    # [branch_a, branch_b] runs in parallel
    START >> prepare >> [branch_a, branch_b] >> merge >> END
```

### Multiple parallel groups

```python
with GraphNode(name="multi-parallel") as graph:
    # First parallel group
    START >> [task_a1, task_a2, task_a3]

    # Sequential step
    [task_a1, task_a2, task_a3] >> middle_step

    # Second parallel group
    middle_step >> [task_b1, task_b2]

    [task_b1, task_b2] >> END
```

## Fan-out / Fan-in Pattern

### Classic fan-out/fan-in

```python
with GraphNode(name="fan-out-fan-in") as graph:
    # Fan-out: one source to multiple targets
    source = CodeNode(
        name="source",
        code_fn=lambda data: {"items": data},
        inputs={"data": PARENT["data"]}
    )

    # Multiple parallel processors
    process_1 = CodeNode(
        name="process_1",
        code_fn=lambda items: {"result": transform_1(items)},
        inputs={"items": source["items"]}
    )
    process_2 = CodeNode(
        name="process_2",
        code_fn=lambda items: {"result": transform_2(items)},
        inputs={"items": source["items"]}
    )
    process_3 = CodeNode(
        name="process_3",
        code_fn=lambda items: {"result": transform_3(items)},
        inputs={"items": source["items"]}
    )

    # Fan-in: aggregate all results
    aggregate = CodeNode(
        name="aggregate",
        code_fn=lambda r1, r2, r3: {
            "combined": combine_results(r1, r2, r3)
        },
        inputs={
            "r1": process_1["result"],
            "r2": process_2["result"],
            "r3": process_3["result"]
        },
        outputs={"*": PARENT}
    )

    START >> source >> [process_1, process_2, process_3] >> aggregate >> END
```

### Dynamic fan-out với MapNode

```python
from hush.core.nodes.iteration import MapNode
from hush.core.nodes.iteration.base import Each

with GraphNode(name="dynamic-fanout") as graph:
    # MapNode tự động fan-out theo items
    with MapNode(
        name="parallel_process",
        inputs={"item": Each(PARENT["items"])},
        max_concurrency=10  # Limit concurrent executions
    ) as map_node:
        process = CodeNode(
            name="process",
            code_fn=lambda item: {"result": transform(item)},
            inputs={"item": PARENT["item"]}
        )
        process["result"] >> PARENT["result"]
        START >> process >> END

    map_node["result"] >> PARENT["results"]
    START >> map_node >> END

# Input: {"items": [1, 2, 3, 4, 5]}
# Output: {"results": [transformed_1, transformed_2, ...]}
```

## Parallel LLM Calls

### Multiple prompts in parallel

```python
from hush.providers import PromptNode, LLMNode

with GraphNode(name="parallel-llm") as graph:
    # Same input, different prompts
    prompt_summary = PromptNode(
        name="prompt_summary",
        inputs={
            "prompt": {"system": "Tóm tắt văn bản sau.", "user": "{text}"},
            "text": PARENT["text"]
        }
    )

    prompt_keywords = PromptNode(
        name="prompt_keywords",
        inputs={
            "prompt": {"system": "Liệt kê keywords.", "user": "{text}"},
            "text": PARENT["text"]
        }
    )

    prompt_sentiment = PromptNode(
        name="prompt_sentiment",
        inputs={
            "prompt": {"system": "Phân tích sentiment.", "user": "{text}"},
            "text": PARENT["text"]
        }
    )

    # Parallel LLM calls
    llm_summary = LLMNode(
        name="llm_summary",
        resource_key="gpt-4",
        inputs={"messages": prompt_summary["messages"]}
    )

    llm_keywords = LLMNode(
        name="llm_keywords",
        resource_key="gpt-4-mini",  # Cheaper model for simple task
        inputs={"messages": prompt_keywords["messages"]}
    )

    llm_sentiment = LLMNode(
        name="llm_sentiment",
        resource_key="gpt-4-mini",
        inputs={"messages": prompt_sentiment["messages"]}
    )

    # Merge results
    merge = CodeNode(
        name="merge",
        code_fn=lambda summary, keywords, sentiment: {
            "summary": summary,
            "keywords": keywords,
            "sentiment": sentiment
        },
        inputs={
            "summary": llm_summary["content"],
            "keywords": llm_keywords["content"],
            "sentiment": llm_sentiment["content"]
        },
        outputs={"*": PARENT}
    )

    # Parallel execution
    START >> [prompt_summary, prompt_keywords, prompt_sentiment]
    prompt_summary >> llm_summary
    prompt_keywords >> llm_keywords
    prompt_sentiment >> llm_sentiment
    [llm_summary, llm_keywords, llm_sentiment] >> merge >> END
```

### Batch LLM với MapNode

```python
with GraphNode(name="batch-llm") as graph:
    with MapNode(
        name="llm_batch",
        inputs={"query": Each(PARENT["queries"])},
        max_concurrency=5  # Avoid rate limiting
    ) as map_node:
        prompt = PromptNode(
            name="prompt",
            inputs={
                "prompt": {"system": "Answer briefly.", "user": "{query}"},
                "query": PARENT["query"]
            }
        )
        llm = LLMNode(
            name="llm",
            resource_key="gpt-4-mini",
            inputs={"messages": prompt["messages"]}
        )
        llm["content"] >> PARENT["answer"]
        START >> prompt >> llm >> END

    map_node["answer"] >> PARENT["answers"]
    START >> map_node >> END
```

## Concurrency Control

### MapNode max_concurrency

```python
with MapNode(
    name="controlled_parallel",
    inputs={"item": Each(PARENT["items"])},
    max_concurrency=5  # Max 5 concurrent executions
) as map_node:
    # ...
```

### Rate limiting với semaphore

```python
import asyncio

# Global semaphore for rate limiting
_semaphore = asyncio.Semaphore(10)

async def rate_limited_call(data):
    async with _semaphore:
        return await api_call(data)

node = CodeNode(
    name="rate_limited",
    code_fn=rate_limited_call,
    inputs={"data": PARENT["data"]}
)
```

## Error Handling trong Parallel Execution

### Partial failure handling

```python
with GraphNode(name="partial-failure") as graph:
    with MapNode(
        name="parallel_process",
        inputs={"item": Each(PARENT["items"])},
        max_concurrency=10
    ) as map_node:
        # Each item processed independently
        # Failed items won't affect others
        process = CodeNode(
            name="process",
            code_fn=lambda item: safe_process(item),  # Returns error if fails
            inputs={"item": PARENT["item"]}
        )
        process["result"] >> PARENT["result"]
        process["error"] >> PARENT["error"]
        START >> process >> END

    # Filter successful results
    filter_results = CodeNode(
        name="filter",
        code_fn=lambda results, errors: {
            "successful": [r for r, e in zip(results, errors) if e is None],
            "failed": [e for e in errors if e is not None]
        },
        inputs={
            "results": map_node["result"],
            "errors": map_node["error"]
        },
        outputs={"*": PARENT}
    )

    START >> map_node >> filter_results >> END
```

### Fail-fast pattern

```python
def process_with_early_exit(items):
    """Process items, stop on first critical error."""
    results = []
    for item in items:
        result = process_single(item)
        if result.get("critical_error"):
            return {"results": results, "error": result["error"], "stopped_early": True}
        results.append(result)
    return {"results": results, "error": None, "stopped_early": False}
```

## Performance Considerations

### 1. Overhead của parallel execution

```python
# Bad: Too many small parallel tasks
# Overhead > benefit
START >> [task1, task2, task3, ..., task100] >> END

# Good: Batch small tasks
with MapNode(
    name="batched",
    inputs={"batch": Each(batches)},  # Each batch has multiple items
    max_concurrency=10
) as map_node:
    process_batch = CodeNode(...)  # Process entire batch
```

### 2. Memory considerations

```python
# Bad: Load all data into memory
items = load_all_items()  # 1M items
results = await parallel_process(items)

# Good: Process in chunks
async def process_in_chunks(items, chunk_size=1000):
    results = []
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i+chunk_size]
        chunk_results = await parallel_process(chunk)
        results.extend(chunk_results)
    return results
```

### 3. Connection pooling

```python
# Configure connection pool for API clients
import httpx

client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_keepalive_connections=20,
        max_connections=100
    )
)
```

## Ví dụ: Parallel Data Processing

### ETL Pipeline

```python
with GraphNode(name="etl-pipeline") as graph:
    # Extract from multiple sources in parallel
    extract_db = CodeNode(
        name="extract_db",
        code_fn=lambda: {"data": fetch_from_db()},
        inputs={}
    )
    extract_api = CodeNode(
        name="extract_api",
        code_fn=lambda: {"data": fetch_from_api()},
        inputs={}
    )
    extract_file = CodeNode(
        name="extract_file",
        code_fn=lambda: {"data": read_from_file()},
        inputs={}
    )

    # Combine extracted data
    combine = CodeNode(
        name="combine",
        code_fn=lambda db, api, file: {
            "combined": merge_data(db, api, file)
        },
        inputs={
            "db": extract_db["data"],
            "api": extract_api["data"],
            "file": extract_file["data"]
        }
    )

    # Transform
    transform = CodeNode(
        name="transform",
        code_fn=lambda data: {"transformed": transform_data(data)},
        inputs={"data": combine["combined"]}
    )

    # Load to multiple destinations in parallel
    load_warehouse = CodeNode(
        name="load_warehouse",
        code_fn=lambda data: {"status": load_to_warehouse(data)},
        inputs={"data": transform["transformed"]}
    )
    load_cache = CodeNode(
        name="load_cache",
        code_fn=lambda data: {"status": update_cache(data)},
        inputs={"data": transform["transformed"]}
    )

    # Parallel extract
    START >> [extract_db, extract_api, extract_file]
    [extract_db, extract_api, extract_file] >> combine

    # Sequential transform
    combine >> transform

    # Parallel load
    transform >> [load_warehouse, load_cache]
    [load_warehouse, load_cache] >> END
```

## Best Practices

1. **Limit concurrency cho rate-limited APIs**
   ```python
   MapNode(..., max_concurrency=5)
   ```

2. **Batch small tasks** thay vì parallelize từng task nhỏ

3. **Handle partial failures** - don't let one failure crash all

4. **Monitor parallel execution** với tracing
   ```python
   tracer = LangfuseTracer(...)
   result = await engine.run(inputs={...}, tracer=tracer)
   ```

5. **Test với small data first** trước khi scale up

## Tiếp theo

- [Xử lý lỗi](error-handling.md) - Error handling trong parallel execution
- [Deploy Production](production-deployment.md) - Scaling parallel workflows
- [Multi-Model Workflow](../examples/multi-model.md) - Parallel multi-model example
