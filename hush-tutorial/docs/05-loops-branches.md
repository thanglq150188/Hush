# Loops và Branches

Sử dụng các node điều khiển luồng: ForLoopNode, MapNode, WhileLoopNode và BranchNode.

> **Ví dụ chạy được**: `examples/05_loops_and_branches.py`

## ForLoopNode — Iterate tuần tự

Xử lý từng item một cách tuần tự. Dùng khi items có thể phụ thuộc vào nhau.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.iteration import ForLoopNode
from hush.core.nodes.iteration.base import Each

with GraphNode(name="sequential-process") as graph:
    with ForLoopNode(
        name="process_items",
        inputs={
            "item": Each(PARENT["items"]),  # Iterate qua mỗi item
            "prefix": PARENT["prefix"]       # Broadcast cho tất cả iterations
        },
        outputs={"results": PARENT}
    ) as loop:
        process = CodeNode(
            name="process",
            code_fn=lambda item, prefix: {"result": f"{prefix}: {item}"},
            inputs={"item": PARENT["item"], "prefix": PARENT["prefix"]},
            outputs={"result": PARENT}
        )
        START >> process >> END

    START >> loop >> END

engine = Hush(graph)
result = await engine.run(inputs={"items": ["a", "b", "c"], "prefix": "Item"})
# result["results"] = ["Item: a", "Item: b", "Item: c"]
```

### Giải thích

- `Each(PARENT["items"])`: Đánh dấu biến sẽ được iterate
- Các biến không có `Each()` sẽ được broadcast cho tất cả iterations
- Output là list kết quả theo thứ tự

## MapNode — Iterate song song

Xử lý nhiều items cùng lúc (parallel). Dùng cho I/O bound tasks hoặc items độc lập.

```python
from hush.core.nodes.iteration import MapNode

with GraphNode(name="parallel-fetch") as graph:
    with MapNode(
        name="fetch_all",
        inputs={"url": Each(PARENT["urls"]), "timeout": 30},
        max_concurrency=10,  # Giới hạn concurrent tasks
        outputs={"results": PARENT}
    ) as map_node:
        fetch = CodeNode(
            name="fetch",
            code_fn=lambda url, timeout: {"data": f"Content from {url}"},
            inputs={"url": PARENT["url"], "timeout": PARENT["timeout"]},
            outputs={"data": PARENT}
        )
        START >> fetch >> END

    START >> map_node >> END
```

### So sánh ForLoopNode vs MapNode

| Tiêu chí | ForLoopNode | MapNode |
|----------|-------------|---------|
| Execution | Tuần tự (sequential) | Song song (parallel) |
| Dependencies | Items có thể phụ thuộc nhau | Items độc lập |
| Memory | Thấp hơn | Cao hơn |
| Use case | Chain processing, stateful | I/O bound, batch processing |

## WhileLoopNode — Loop với điều kiện

Chạy cho đến khi điều kiện trả về False.

```python
from hush.core.nodes.iteration import WhileLoopNode

with GraphNode(name="countdown") as graph:
    with WhileLoopNode(
        name="countdown_loop",
        condition=lambda count: count > 0,
        inputs={"count": PARENT["start"]},
        outputs={"final_count": PARENT}
    ) as loop:
        decrement = CodeNode(
            name="decrement",
            code_fn=lambda count: {"count": count - 1, "message": f"Count: {count}"},
            inputs={"count": PARENT["count"]},
            outputs={"count": PARENT, "message": PARENT}
        )
        START >> decrement >> END

    START >> loop >> END

result = await engine.run(inputs={"start": 5})
# result["final_count"] = 0
```

## BranchNode — Conditional Routing

Định tuyến workflow theo điều kiện. Chỉ một nhánh được thực thi.

```python
from hush.core.nodes.flow.branch_node import BranchNode

with GraphNode(name="grade-workflow") as graph:
    branch = BranchNode(
        name="grade_router",
        cases={
            "score >= 90": "excellent",
            "score >= 70": "good",
            "score >= 50": "average",
        },
        default="fail",
        inputs={"score": PARENT["score"]}
    )

    excellent = CodeNode(name="excellent", code_fn=lambda: {"grade": "A"}, outputs={"grade": PARENT})
    good = CodeNode(name="good", code_fn=lambda: {"grade": "B"}, outputs={"grade": PARENT})
    average = CodeNode(name="average", code_fn=lambda: {"grade": "C"}, outputs={"grade": PARENT})
    fail = CodeNode(name="fail", code_fn=lambda: {"grade": "F"}, outputs={"grade": PARENT})

    START >> branch
    branch >> [excellent, good, average, fail]
    [excellent, good, average, fail] >> ~END  # Soft edge — chỉ 1 nhánh chạy

result = await engine.run(inputs={"score": 85})
# result["grade"] = "B"
```

### Hard Edge vs Soft Edge

- `>>` (Hard Edge): Node đích chờ **tất cả** predecessors hoàn thành
- `~` (Soft Edge): Node đích chờ **bất kỳ một** soft predecessor hoàn thành

```python
# Sau branch, dùng soft edge vì chỉ 1 nhánh chạy
[path_a, path_b, path_c] >> ~merge_node
```

## Nested Loops

Loops có thể nest bên trong nhau:

```python
with GraphNode(name="nested") as graph:
    with ForLoopNode(
        name="outer",
        inputs={"category": Each(PARENT["categories"])},
        outputs={"all_results": PARENT}
    ) as outer:
        with MapNode(
            name="inner",
            inputs={"item": Each(PARENT["category"]["items"])},
            max_concurrency=5,
            outputs={"category_results": PARENT}
        ) as inner:
            process = CodeNode(...)
            START >> process >> END
        START >> inner >> END
    START >> outer >> END
```

## Tổng kết

| Node | Execution | Use case |
|------|-----------|----------|
| `ForLoopNode` | Sequential | Items phụ thuộc nhau |
| `MapNode` | Parallel | I/O bound, independent items |
| `WhileLoopNode` | Conditional | Loop đến khi điều kiện False |
| `BranchNode` | Conditional | Route dựa trên điều kiện |

| Syntax | Mô tả |
|--------|-------|
| `Each(PARENT["items"])` | Đánh dấu biến để iterate |
| `>>` | Hard edge — chờ tất cả |
| `~` | Soft edge — chờ bất kỳ một |

## Tiếp theo

- [Embeddings & RAG](06-embeddings-rag.md) — Vector search và reranking
- [Error Handling](07-error-handling.md) — Xử lý lỗi
- [Parallel Execution](08-parallel-execution.md) — Chi tiết về parallel patterns
