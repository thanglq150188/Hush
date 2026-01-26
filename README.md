# Hush

> Async workflow orchestration engine cho GenAI applications.

Hush là một **workflow engine** nhẹ, được thiết kế để xây dựng các pipeline AI/LLM phức tạp một cách đơn giản và hiệu quả.

## Cài đặt

```bash
# Standard - workflow engine + LLM providers (OpenAI)
pip install hush-ai[standard]

# Core only - workflow engine với local tracing
pip install hush-ai[core]

# Full - tất cả providers + observability
pip install hush-ai[all]
```

## Quick Start

```python
import asyncio
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT

async def main():
    with GraphNode(name="hello-world") as graph:
        greet = CodeNode(
            name="greet",
            code_fn=lambda name: {"greeting": f"Xin chào, {name}!"},
            inputs={"name": PARENT["name"]},
            outputs={"greeting": PARENT}
        )
        START >> greet >> END

    engine = Hush(graph)
    result = await engine.run(inputs={"name": "Hush"})
    print(result["greeting"])  # Xin chào, Hush!

asyncio.run(main())
```

## Sử dụng với LLM

```python
from hush.core import Hush, GraphNode, START, END, PARENT
from hush.providers import PromptNode, LLMNode

async def main():
    with GraphNode(name="chat") as graph:
        prompt = PromptNode(
            name="prompt",
            inputs={
                "prompt": {"system": "Bạn là trợ lý AI.", "user": "{question}"},
                "question": PARENT["question"]
            },
            outputs={"messages": PARENT}
        )
        llm = LLMNode(
            name="llm",
            resource_key="gpt-4o",  # Cấu hình trong resources.yaml
            inputs={"messages": PARENT["messages"]},
            outputs={"content": PARENT["answer"]}
        )
        START >> prompt >> llm >> END

    engine = Hush(graph)
    result = await engine.run(inputs={"question": "Python là gì?"})
    print(result["answer"])

asyncio.run(main())
```

## Kiến trúc

Hush được tổ chức thành 3 package độc lập:

```
┌─────────────────────────────────────────────────────────┐
│                    hush-observability                   │
│         (LocalTracer, Langfuse, OpenTelemetry)          │
├─────────────────────────────────────────────────────────┤
│                     hush-providers                      │
│    (LLMNode, PromptNode, EmbeddingNode, RerankNode)     │
├─────────────────────────────────────────────────────────┤
│                       hush-core                         │
│  (GraphNode, CodeNode, BranchNode, State, ResourceHub)  │
└─────────────────────────────────────────────────────────┘
```

| Package | Mô tả |
|---------|-------|
| [hush-core](hush-core/) | Workflow engine cốt lõi, state management, local tracing |
| [hush-providers](hush-providers/) | LLM, embedding, reranking providers |
| [hush-observability](hush-observability/) | Tích hợp Langfuse, OpenTelemetry |
| [hush-vscode-traceview](hush-vscode-traceview/) | VS Code extension xem traces |

## Tính năng chính

- **Async-first**: Hỗ trợ parallel execution tự động
- **Declarative data flow**: Syntax `PARENT["key"]` rõ ràng, dễ hiểu
- **Built-in tracing**: Local SQLite tracer với Web UI và VS Code extension
- **Modular**: Cài đặt theo nhu cầu, không bloat
- **Type-safe**: Schema validation tại build time và runtime

## Local Trace Viewer

Traces được tự động lưu vào `~/.hush/traces.db`. Xem traces bằng:

```bash
# Web UI
python -m hush.core.ui.server
# Mở http://localhost:8765
```

Hoặc cài VS Code extension: [hush-vscode-traceview](hush-vscode-traceview/)

## Tài liệu

- [Bắt đầu nhanh](docs/getting-started/quickstart.md)
- [Khái niệm cốt lõi](docs/concepts/overview.md)
- [Hướng dẫn xây dựng workflow](docs/guides/building-workflows.md)
- [Ví dụ](docs/examples/)
- [API Reference](docs/reference/)

## License

MIT

## Liên hệ

thanglq150188@gmail.com
