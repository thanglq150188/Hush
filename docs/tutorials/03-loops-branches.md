# Tutorial 3: Loops và Branches

Tutorial này hướng dẫn cách sử dụng các node điều khiển luồng trong Hush: ForLoopNode, MapNode, WhileLoopNode và BranchNode.

## ForLoopNode - Iterate tuần tự

`ForLoopNode` xử lý từng item một cách tuần tự. Dùng khi items có thể phụ thuộc vào nhau.

### Basic usage

```python
import asyncio
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.iteration import ForLoopNode
from hush.core.nodes.iteration.base import Each

async def main():
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
    result = await engine.run(inputs={
        "items": ["a", "b", "c"],
        "prefix": "Item"
    })

    print(result["results"])
    # Output: ["Item: a", "Item: b", "Item: c"]

asyncio.run(main())
```

### Giải thích

- `Each(PARENT["items"])`: Đánh dấu biến này sẽ được iterate
- Các biến không có `Each()` sẽ được broadcast cho tất cả iterations
- Output là list các kết quả theo thứ tự

## MapNode - Iterate song song

`MapNode` xử lý nhiều items cùng lúc (parallel). Dùng cho I/O bound tasks hoặc khi items độc lập.

### Basic usage

```python
import asyncio
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.iteration import MapNode
from hush.core.nodes.iteration.base import Each

async def fetch_url(url: str, timeout: int) -> dict:
    """Simulate fetching URL."""
    await asyncio.sleep(0.1)  # Simulate network delay
    return {"data": f"Content from {url}"}

async def main():
    with GraphNode(name="parallel-fetch") as graph:
        with MapNode(
            name="fetch_all",
            inputs={
                "url": Each(PARENT["urls"]),  # Iterate song song
                "timeout": 30                  # Broadcast
            },
            max_concurrency=10,  # Giới hạn concurrent tasks
            outputs={"results": PARENT}
        ) as map_node:
            fetch = CodeNode(
                name="fetch",
                code_fn=fetch_url,
                inputs={"url": PARENT["url"], "timeout": PARENT["timeout"]},
                outputs={"data": PARENT}
            )
            START >> fetch >> END

        START >> map_node >> END

    engine = Hush(graph)
    result = await engine.run(inputs={
        "urls": ["https://a.com", "https://b.com", "https://c.com"]
    })

    print(result["results"])

asyncio.run(main())
```

### So sánh ForLoopNode vs MapNode

| Tiêu chí | ForLoopNode | MapNode |
|----------|-------------|---------|
| Execution | Tuần tự (sequential) | Song song (parallel) |
| Dependencies | Items có thể phụ thuộc nhau | Items độc lập |
| Memory | Thấp hơn | Cao hơn (nhiều tasks cùng lúc) |
| Use case | Chain processing, stateful | I/O bound, batch processing |

## WhileLoopNode - Loop với điều kiện

`WhileLoopNode` chạy cho đến khi điều kiện trả về False.

```python
import asyncio
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.core.nodes.iteration import WhileLoopNode

async def main():
    with GraphNode(name="countdown") as graph:
        with WhileLoopNode(
            name="countdown_loop",
            condition=lambda count: count > 0,  # Tiếp tục khi count > 0
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

    engine = Hush(graph)
    result = await engine.run(inputs={"start": 5})

    print(f"Final count: {result['final_count']}")

asyncio.run(main())
```

## BranchNode - Conditional Routing

`BranchNode` định tuyến workflow theo điều kiện. Chỉ một nhánh được thực thi.

### Basic usage với string conditions

```python
import asyncio
from hush.core import Hush, GraphNode, CodeNode, BranchNode, START, END, PARENT

async def main():
    with GraphNode(name="grade-workflow") as graph:
        # Branch node với string conditions
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

        # Các target nodes
        excellent = CodeNode(
            name="excellent",
            code_fn=lambda: {"grade": "A", "message": "Xuất sắc!"},
            outputs={"grade": PARENT, "message": PARENT}
        )
        good = CodeNode(
            name="good",
            code_fn=lambda: {"grade": "B", "message": "Tốt!"},
            outputs={"grade": PARENT, "message": PARENT}
        )
        average = CodeNode(
            name="average",
            code_fn=lambda: {"grade": "C", "message": "Trung bình"},
            outputs={"grade": PARENT, "message": PARENT}
        )
        fail = CodeNode(
            name="fail",
            code_fn=lambda: {"grade": "F", "message": "Cần cải thiện"},
            outputs={"grade": PARENT, "message": PARENT}
        )

        # Kết nối
        START >> branch
        branch >> [excellent, good, average, fail]

        # Merge với soft edges (chỉ 1 nhánh chạy)
        [excellent, good, average, fail] >> ~END

    engine = Hush(graph)

    # Test với các scores khác nhau
    for score in [95, 75, 55, 30]:
        result = await engine.run(inputs={"score": score})
        print(f"Score {score}: {result['grade']} - {result['message']}")

asyncio.run(main())
```

### Fluent builder với Ref

```python
from hush.core.nodes.flow.branch_node import Branch

branch = (Branch("router")
    .if_(PARENT["score"] >= 90, "excellent")
    .if_(PARENT["score"] >= 70, "good")
    .if_(PARENT["score"] >= 50, "average")
    .otherwise("fail"))
```

### Hard Edge vs Soft Edge

- `>>` (Hard Edge): Node đích chờ TẤT CẢ predecessors hoàn thành
- `~` (Soft Edge): Node đích chờ BẤT KỲ MỘT soft predecessor hoàn thành

```python
# Sau branch, dùng soft edge vì chỉ 1 nhánh chạy
[path_a, path_b, path_c] >> ~merge_node
# hoặc
path_a >> ~merge_node
path_b >> ~merge_node
path_c >> ~merge_node
```

## Ví dụ thực tế: Batch Processing với Retry

```python
import asyncio
from hush.core import Hush, GraphNode, CodeNode, BranchNode, START, END, PARENT
from hush.core.nodes.iteration import MapNode
from hush.core.nodes.iteration.base import Each

async def process_item(item: dict) -> dict:
    """Process một item, có thể fail."""
    import random
    if random.random() < 0.3:  # 30% chance to fail
        return {"success": False, "error": "Random failure", "item": item}
    return {"success": True, "result": item["value"] * 2, "item": item}

async def main():
    with GraphNode(name="batch-with-retry") as graph:
        # Bước 1: Parallel processing
        with MapNode(
            name="process_batch",
            inputs={"item": Each(PARENT["items"])},
            max_concurrency=5,
            outputs={"results": PARENT}
        ) as batch:
            process = CodeNode(
                name="process",
                code_fn=process_item,
                inputs={"item": PARENT["item"]},
                outputs={"*": PARENT}
            )
            START >> process >> END

        # Bước 2: Separate thành công và thất bại
        separate = CodeNode(
            name="separate",
            code_fn=lambda results: {
                "succeeded": [r for r in results if r.get("success")],
                "failed": [r for r in results if not r.get("success")]
            },
            inputs={"results": PARENT["results"]},
            outputs={"succeeded": PARENT, "failed": PARENT}
        )

        # Bước 3: Check nếu có failures
        branch = BranchNode(
            name="check_failures",
            cases={"len(failed) > 0": "has_failures"},
            default="all_success",
            inputs={"failed": PARENT["failed"]}
        )

        # Nếu có failures, tổng hợp report
        report_failures = CodeNode(
            name="report_failures",
            code_fn=lambda succeeded, failed: {
                "total": len(succeeded) + len(failed),
                "success_count": len(succeeded),
                "failure_count": len(failed),
                "status": "partial"
            },
            inputs={"succeeded": PARENT["succeeded"], "failed": PARENT["failed"]},
            outputs={"*": PARENT}
        )

        # Nếu tất cả thành công
        report_success = CodeNode(
            name="report_success",
            code_fn=lambda succeeded: {
                "total": len(succeeded),
                "success_count": len(succeeded),
                "failure_count": 0,
                "status": "complete"
            },
            inputs={"succeeded": PARENT["succeeded"]},
            outputs={"*": PARENT}
        )

        # Kết nối
        START >> batch >> separate >> branch
        branch >> [report_failures, report_success]
        [report_failures, report_success] >> ~END

    engine = Hush(graph)
    result = await engine.run(inputs={
        "items": [{"id": i, "value": i * 10} for i in range(10)]
    })

    print(f"Status: {result['status']}")
    print(f"Total: {result['total']}, Success: {result['success_count']}, Failed: {result['failure_count']}")

asyncio.run(main())
```

## Nested Loops

Loops có thể được nest bên trong nhau:

```python
with GraphNode(name="nested-loops") as graph:
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
| `>>` | Hard edge - chờ tất cả |
| `~` | Soft edge - chờ bất kỳ một |

## Tiếp theo

- [Tutorial 4: Production](04-production.md) - Tracing, error handling, deployment
- [Guide: Parallel Execution](../guides/parallel-execution.md) - Chi tiết về parallel
