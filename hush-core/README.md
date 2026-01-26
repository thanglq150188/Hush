# Hush Core

Workflow engine cốt lõi cho Hush - async orchestration với built-in tracing.

## Cài đặt

```bash
pip install hush-core
```

## Quick Start

```python
import asyncio
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT

async def main():
    with GraphNode(name="my-workflow") as graph:
        step1 = CodeNode(
            name="fetch",
            code_fn=lambda: {"data": [1, 2, 3, 4, 5]},
            outputs={"data": PARENT}
        )
        step2 = CodeNode(
            name="transform",
            code_fn=lambda data: {"result": sum(data)},
            inputs={"data": PARENT["data"]},
            outputs={"result": PARENT}
        )
        START >> step1 >> step2 >> END

    engine = Hush(graph)
    result = await engine.run()
    print(result["result"])  # 15

asyncio.run(main())
```

## Thành phần chính

### Nodes

| Node | Mô tả |
|------|-------|
| `GraphNode` | Container chứa subgraph |
| `CodeNode` | Chạy Python function |
| `BranchNode` | Conditional routing |
| `ForLoopNode` | Iterate qua collection |
| `MapNode` | Parallel map qua collection |
| `WhileLoopNode` | Loop với điều kiện |

### Flow Control

```python
# Sequential
START >> node1 >> node2 >> END

# Fork (parallel)
START >> node1 >> [node2a, node2b] >> node3 >> END

# Branch (conditional)
START >> branch_node >> {
    "case_a": node_a,
    "case_b": node_b
} >> END
```

### State Management

```python
# Đọc từ parent scope
inputs={"data": PARENT["input_data"]}

# Ghi ra parent scope
outputs={"result": PARENT}

# Đọc từ node khác
inputs={"value": other_node["output_key"]}
```

## Local Tracing

hush-core có built-in LocalTracer lưu traces vào SQLite:

```python
from hush.core import Hush, GraphNode
from hush.core.tracers import LocalTracer

tracer = LocalTracer()  # Mặc định: ~/.hush/traces.db

with GraphNode(name="demo") as graph:
    # ... định nghĩa nodes
    pass

engine = Hush(graph, tracer=tracer)
await engine.run()

# Xem traces
# python -m hush.core.ui.server
```

## Packages liên quan

- [hush-providers](../hush-providers/) - LLM, embedding, reranking nodes
- [hush-observability](../hush-observability/) - Langfuse, OpenTelemetry integration
- [hush-vscode-traceview](../hush-vscode-traceview/) - VS Code extension

## License

MIT
