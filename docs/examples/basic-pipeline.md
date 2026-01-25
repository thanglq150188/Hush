# Pipeline cơ bản

Ví dụ này hướng dẫn xây dựng một pipeline cơ bản với Hush, từ đơn giản đến phức tạp.

## Ví dụ 1: Hello World

Pipeline đơn giản nhất với một node.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT

# Define workflow
with GraphNode(name="hello-world") as graph:
    greet = CodeNode(
        name="greet",
        code_fn=lambda name: {"message": f"Hello, {name}!"},
        inputs={"name": PARENT["name"]},
        outputs={"*": PARENT}
    )
    START >> greet >> END

# Run workflow
async def main():
    engine = Hush(graph)
    result = await engine.run(inputs={"name": "World"})
    print(result["message"])  # "Hello, World!"

import asyncio
asyncio.run(main())
```

## Ví dụ 2: Sequential Pipeline

Pipeline với nhiều bước xử lý tuần tự.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT

with GraphNode(name="text-processor") as graph:
    # Step 1: Clean text
    clean = CodeNode(
        name="clean",
        code_fn=lambda text: {"cleaned": text.strip().lower()},
        inputs={"text": PARENT["text"]}
    )

    # Step 2: Tokenize
    tokenize = CodeNode(
        name="tokenize",
        code_fn=lambda text: {"tokens": text.split()},
        inputs={"text": clean["cleaned"]}
    )

    # Step 3: Count words
    count = CodeNode(
        name="count",
        code_fn=lambda tokens: {
            "word_count": len(tokens),
            "unique_count": len(set(tokens))
        },
        inputs={"tokens": tokenize["tokens"]},
        outputs={"*": PARENT}
    )

    START >> clean >> tokenize >> count >> END

# Run
async def main():
    engine = Hush(graph)
    result = await engine.run(inputs={"text": "  Hello World Hello  "})
    print(result)
    # {"word_count": 3, "unique_count": 2}

import asyncio
asyncio.run(main())
```

## Ví dụ 3: Parallel Branches

Xử lý song song và merge kết quả.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT

with GraphNode(name="parallel-analysis") as graph:
    # Prepare data
    prepare = CodeNode(
        name="prepare",
        code_fn=lambda text: {"text": text.strip()},
        inputs={"text": PARENT["text"]}
    )

    # Parallel analysis branches
    count_words = CodeNode(
        name="count_words",
        code_fn=lambda text: {"word_count": len(text.split())},
        inputs={"text": prepare["text"]}
    )

    count_chars = CodeNode(
        name="count_chars",
        code_fn=lambda text: {"char_count": len(text)},
        inputs={"text": prepare["text"]}
    )

    find_longest = CodeNode(
        name="find_longest",
        code_fn=lambda text: {
            "longest_word": max(text.split(), key=len) if text.split() else ""
        },
        inputs={"text": prepare["text"]}
    )

    # Merge all results
    merge = CodeNode(
        name="merge",
        code_fn=lambda words, chars, longest: {
            "word_count": words,
            "char_count": chars,
            "longest_word": longest
        },
        inputs={
            "words": count_words["word_count"],
            "chars": count_chars["char_count"],
            "longest": find_longest["longest_word"]
        },
        outputs={"*": PARENT}
    )

    # Parallel execution with [list] syntax
    START >> prepare >> [count_words, count_chars, find_longest] >> merge >> END

# Run
async def main():
    engine = Hush(graph)
    result = await engine.run(inputs={"text": "Hello wonderful world"})
    print(result)
    # {"word_count": 3, "char_count": 21, "longest_word": "wonderful"}

import asyncio
asyncio.run(main())
```

## Ví dụ 4: Conditional Logic

Routing dựa trên conditions.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.flow.branch_node import Branch

with GraphNode(name="score-grader") as graph:
    # Evaluate score
    evaluate = CodeNode(
        name="evaluate",
        code_fn=lambda score: {"score": score},
        inputs={"score": PARENT["score"]}
    )

    # Conditional routing
    router = (Branch("grade_router")
        .if_(evaluate["score"] >= 90, "grade_a")
        .if_(evaluate["score"] >= 80, "grade_b")
        .if_(evaluate["score"] >= 70, "grade_c")
        .if_(evaluate["score"] >= 60, "grade_d")
        .otherwise("grade_f"))

    # Grade handlers
    grade_a = CodeNode(
        name="grade_a",
        code_fn=lambda: {"grade": "A", "status": "Excellent"},
        inputs={}
    )
    grade_b = CodeNode(
        name="grade_b",
        code_fn=lambda: {"grade": "B", "status": "Good"},
        inputs={}
    )
    grade_c = CodeNode(
        name="grade_c",
        code_fn=lambda: {"grade": "C", "status": "Average"},
        inputs={}
    )
    grade_d = CodeNode(
        name="grade_d",
        code_fn=lambda: {"grade": "D", "status": "Below Average"},
        inputs={}
    )
    grade_f = CodeNode(
        name="grade_f",
        code_fn=lambda: {"grade": "F", "status": "Fail"},
        inputs={}
    )

    # Merge - sử dụng soft edges vì chỉ MỘT branch chạy
    merge = CodeNode(
        name="merge",
        code_fn=lambda grade=None, status=None, **kwargs: {
            "grade": grade,
            "status": status
        },
        inputs={
            "grade": grade_a["grade"],
            "status": grade_a["status"]
        },
        outputs={"*": PARENT}
    )

    # Flow
    START >> evaluate >> router >> [grade_a, grade_b, grade_c, grade_d, grade_f]
    [grade_a, grade_b, grade_c, grade_d, grade_f] >> ~merge  # Soft edges!
    merge >> END

# Run
async def main():
    engine = Hush(graph)

    result = await engine.run(inputs={"score": 85})
    print(result)  # {"grade": "B", "status": "Good"}

    result = await engine.run(inputs={"score": 55})
    print(result)  # {"grade": "F", "status": "Fail"}

import asyncio
asyncio.run(main())
```

## Ví dụ 5: Loop Processing

Xử lý collection với ForLoopNode và MapNode.

### ForLoopNode (Sequential)

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.iteration.for_loop_node import ForLoopNode
from hush.core.nodes.iteration.base import Each

with GraphNode(name="sequential-loop") as graph:
    with ForLoopNode(
        name="process_items",
        inputs={"item": Each(PARENT["items"])}
    ) as loop:
        process = CodeNode(
            name="process",
            code_fn=lambda item: {"doubled": item * 2},
            inputs={"item": PARENT["item"]}
        )
        process["doubled"] >> PARENT["doubled"]
        START >> process >> END

    loop["doubled"] >> PARENT["results"]
    START >> loop >> END

# Run
async def main():
    engine = Hush(graph)
    result = await engine.run(inputs={"items": [1, 2, 3, 4, 5]})
    print(result["results"])  # [2, 4, 6, 8, 10]

import asyncio
asyncio.run(main())
```

### MapNode (Parallel)

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.iteration import MapNode
from hush.core.nodes.iteration.base import Each

with GraphNode(name="parallel-loop") as graph:
    with MapNode(
        name="process_items",
        inputs={"item": Each(PARENT["items"])},
        max_concurrency=5  # Giới hạn concurrent tasks
    ) as map_node:
        process = CodeNode(
            name="process",
            code_fn=lambda item: {"squared": item ** 2},
            inputs={"item": PARENT["item"]}
        )
        process["squared"] >> PARENT["squared"]
        START >> process >> END

    map_node["squared"] >> PARENT["results"]
    START >> map_node >> END

# Run
async def main():
    engine = Hush(graph)
    result = await engine.run(inputs={"items": [1, 2, 3, 4, 5]})
    print(result["results"])  # [1, 4, 9, 16, 25]

import asyncio
asyncio.run(main())
```

## Ví dụ 6: LLM Integration

Pipeline với LLM call.

```python
from hush.core import Hush, GraphNode, START, END, PARENT
from hush.providers import PromptNode, LLMNode

with GraphNode(name="chat-pipeline") as graph:
    # Build prompt
    prompt = PromptNode(
        name="prompt",
        inputs={
            "prompt": {
                "system": "Bạn là assistant thân thiện, trả lời ngắn gọn.",
                "user": "{query}"
            },
            "query": PARENT["query"]
        }
    )

    # Call LLM
    llm = LLMNode(
        name="llm",
        resource_key="llm:default",
        inputs={"messages": prompt["messages"]}
    )

    llm["content"] >> PARENT["response"]

    START >> prompt >> llm >> END

# Run
async def main():
    engine = Hush(graph)
    result = await engine.run(inputs={"query": "Thủ đô Việt Nam là gì?"})
    print(result["response"])  # "Thủ đô Việt Nam là Hà Nội."

import asyncio
asyncio.run(main())
```

## Ví dụ 7: Error Handling

Pipeline với error handling.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT

def safe_divide(a, b):
    try:
        return {"result": a / b, "error": None}
    except ZeroDivisionError:
        return {"result": None, "error": "Cannot divide by zero"}
    except Exception as e:
        return {"result": None, "error": str(e)}

with GraphNode(name="safe-pipeline") as graph:
    divide = CodeNode(
        name="divide",
        code_fn=safe_divide,
        inputs={"a": PARENT["a"], "b": PARENT["b"]}
    )

    # Handle result
    handle = CodeNode(
        name="handle",
        code_fn=lambda result, error: {
            "output": result if error is None else f"Error: {error}",
            "success": error is None
        },
        inputs={
            "result": divide["result"],
            "error": divide["error"]
        },
        outputs={"*": PARENT}
    )

    START >> divide >> handle >> END

# Run
async def main():
    engine = Hush(graph)

    # Success case
    result = await engine.run(inputs={"a": 10, "b": 2})
    print(result)  # {"output": 5.0, "success": True}

    # Error case
    result = await engine.run(inputs={"a": 10, "b": 0})
    print(result)  # {"output": "Error: Cannot divide by zero", "success": False}

import asyncio
asyncio.run(main())
```

## Ví dụ 8: Complete Data Pipeline

Pipeline hoàn chỉnh xử lý data.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.iteration import MapNode
from hush.core.nodes.iteration.base import Each

with GraphNode(name="data-pipeline") as graph:
    # Step 1: Validate input
    validate = CodeNode(
        name="validate",
        code_fn=lambda data: {
            "valid_data": [d for d in data if d.get("id") and d.get("value")],
            "invalid_count": len([d for d in data if not (d.get("id") and d.get("value"))])
        },
        inputs={"data": PARENT["data"]}
    )

    # Step 2: Transform each item (parallel)
    with MapNode(
        name="transform",
        inputs={"item": Each(validate["valid_data"])},
        max_concurrency=10
    ) as transform_map:
        transform_item = CodeNode(
            name="transform_item",
            code_fn=lambda item: {
                "transformed": {
                    "id": item["id"],
                    "value": item["value"] * 2,
                    "processed": True
                }
            },
            inputs={"item": PARENT["item"]}
        )
        transform_item["transformed"] >> PARENT["transformed"]
        START >> transform_item >> END

    # Step 3: Aggregate results
    aggregate = CodeNode(
        name="aggregate",
        code_fn=lambda items, invalid_count: {
            "results": items,
            "total_processed": len(items),
            "total_invalid": invalid_count,
            "total_value": sum(item["value"] for item in items)
        },
        inputs={
            "items": transform_map["transformed"],
            "invalid_count": validate["invalid_count"]
        },
        outputs={"*": PARENT}
    )

    START >> validate >> transform_map >> aggregate >> END

# Run
async def main():
    engine = Hush(graph)

    data = [
        {"id": "1", "value": 10},
        {"id": "2", "value": 20},
        {"value": 30},  # Invalid - no id
        {"id": "3", "value": 15},
    ]

    result = await engine.run(inputs={"data": data})
    print(f"Processed: {result['total_processed']}")  # 3
    print(f"Invalid: {result['total_invalid']}")      # 1
    print(f"Total value: {result['total_value']}")    # 90 (20+40+30)

import asyncio
asyncio.run(main())
```

## Sử dụng Tracer

Thêm tracing để debug và monitor.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.tracers import LocalTracer

with GraphNode(name="traced-pipeline") as graph:
    step1 = CodeNode(
        name="step1",
        code_fn=lambda x: {"result": x + 10},
        inputs={"x": PARENT["x"]}
    )
    step2 = CodeNode(
        name="step2",
        code_fn=lambda x: {"result": x * 2},
        inputs={"x": step1["result"]},
        outputs={"*": PARENT}
    )
    START >> step1 >> step2 >> END

async def main():
    engine = Hush(graph)

    # Add tracer
    tracer = LocalTracer(tags=["example", "basic"])

    result = await engine.run(
        inputs={"x": 5},
        tracer=tracer
    )

    print(result["result"])  # 30

    # Traces saved to ~/.hush/traces.db

import asyncio
asyncio.run(main())
```

## Tiếp theo

- [RAG Workflow](rag-workflow.md) - Xây dựng RAG pipeline
- [Agent Workflow](agent-workflow.md) - Xây dựng AI Agent
- [Multi-Model Workflow](multi-model.md) - Sử dụng nhiều LLM models
