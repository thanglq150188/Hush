# Thực thi song song

Hướng dẫn tận dụng parallel execution để tăng performance của workflows.

## Khi nào dùng Parallel Execution?

**Phù hợp khi:**
- Independent operations - các tasks không phụ thuộc nhau
- I/O-bound operations - API calls, database queries
- Batch processing - xử lý nhiều items cùng lúc
- Multiple LLM calls

**Không nên dùng khi:**
- Tasks có dependency tuần tự
- CPU-bound operations (Python GIL limitation)
- Rate-limited APIs (cần control concurrency)

## Syntax Parallel Branches

### List syntax

```python
from hush.core import GraphNode, CodeNode, START, END, PARENT

with GraphNode(name="parallel-demo") as graph:
    prepare = CodeNode(
        name="prepare",
        code_fn=lambda x: {"value": x},
        inputs={"x": PARENT["x"]}
    )

    # Parallel branches
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

## Fan-out / Fan-in Pattern

```python
with GraphNode(name="fan-out-fan-in") as graph:
    source = CodeNode(
        name="source",
        code_fn=lambda data: {"items": data},
        inputs={"data": PARENT["data"]}
    )

    # Multiple parallel processors
    process_1 = CodeNode(name="process_1", code_fn=lambda items: {"result": transform_1(items)}, inputs={"items": source["items"]})
    process_2 = CodeNode(name="process_2", code_fn=lambda items: {"result": transform_2(items)}, inputs={"items": source["items"]})
    process_3 = CodeNode(name="process_3", code_fn=lambda items: {"result": transform_3(items)}, inputs={"items": source["items"]})

    # Fan-in: aggregate
    aggregate = CodeNode(
        name="aggregate",
        code_fn=lambda r1, r2, r3: {"combined": combine_results(r1, r2, r3)},
        inputs={
            "r1": process_1["result"],
            "r2": process_2["result"],
            "r3": process_3["result"]
        },
        outputs={"*": PARENT}
    )

    START >> source >> [process_1, process_2, process_3] >> aggregate >> END
```

## Dynamic fan-out với MapNode

```python
from hush.core.nodes.iteration import MapNode
from hush.core.nodes.iteration.base import Each

with GraphNode(name="dynamic-fanout") as graph:
    with MapNode(
        name="parallel_process",
        inputs={"item": Each(PARENT["items"])},
        max_concurrency=10
    ) as map_node:
        process = CodeNode(
            name="process",
            code_fn=lambda item: {"result": transform(item)},
            inputs={"item": PARENT["item"]},
            outputs={"result": PARENT}
        )
        START >> process >> END

    map_node["result"] >> PARENT["results"]
    START >> map_node >> END
```

## Parallel LLM Calls

### Multiple prompts in parallel

```python
from hush.providers import PromptNode, LLMNode

with GraphNode(name="parallel-llm") as graph:
    # Different prompts for same input
    prompt_summary = PromptNode(name="prompt_summary", inputs={
        "prompt": {"system": "Tóm tắt văn bản sau.", "user": "{text}"},
        "text": PARENT["text"]
    })

    prompt_keywords = PromptNode(name="prompt_keywords", inputs={
        "prompt": {"system": "Liệt kê keywords.", "user": "{text}"},
        "text": PARENT["text"]
    })

    # Parallel LLM calls
    llm_summary = LLMNode(name="llm_summary", resource_key="gpt-4o", inputs={"messages": prompt_summary["messages"]})
    llm_keywords = LLMNode(name="llm_keywords", resource_key="gpt-4o-mini", inputs={"messages": prompt_keywords["messages"]})

    # Merge results
    merge = CodeNode(
        name="merge",
        code_fn=lambda summary, keywords: {"summary": summary, "keywords": keywords},
        inputs={
            "summary": llm_summary["content"],
            "keywords": llm_keywords["content"]
        },
        outputs={"*": PARENT}
    )

    START >> [prompt_summary, prompt_keywords]
    prompt_summary >> llm_summary
    prompt_keywords >> llm_keywords
    [llm_summary, llm_keywords] >> merge >> END
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
            resource_key="gpt-4o-mini",
            inputs={"messages": prompt["messages"]},
            outputs={"content": PARENT["answer"]}
        )
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
    ...
```

### Rate limiting với semaphore

```python
import asyncio

_semaphore = asyncio.Semaphore(10)

async def rate_limited_call(data):
    async with _semaphore:
        return await api_call(data)
```

## Error Handling trong Parallel

### Partial failure handling

```python
with MapNode(
    name="parallel_process",
    inputs={"item": Each(PARENT["items"])},
    max_concurrency=10
) as map_node:
    # Each item processed independently
    # Failed items won't affect others
    process = CodeNode(
        name="process",
        code_fn=lambda item: safe_process(item),
        inputs={"item": PARENT["item"]},
        outputs={"result": PARENT, "error": PARENT}
    )
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
```

## Performance Considerations

1. **Avoid too many small parallel tasks** - overhead > benefit
2. **Batch small tasks** thay vì parallelize từng task nhỏ
3. **Process in chunks** để manage memory
4. **Configure connection pooling** cho API clients

## Best Practices

1. **Limit concurrency** cho rate-limited APIs
2. **Batch small tasks**
3. **Handle partial failures**
4. **Monitor với tracing**
5. **Test với small data first**

## Xem thêm

- [Tutorial: Loops và Branches](../tutorials/03-loops-branches.md)
- [Xử lý lỗi](error-handling.md)
- [Multi-Model Workflow](../examples/multi-model.md)
