# Hush Documentation

> Async workflow orchestration engine cho GenAI applications.

## Hush là gì?

**Hush** là workflow engine cho các ứng dụng AI/LLM, được thiết kế để xây dựng các pipeline phức tạp một cách đơn giản và hiệu quả. Hush tập trung vào việc điều phối (orchestration) và thực thi (execution) các node trong một graph.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END, PARENT
from hush.providers import PromptNode, LLMNode

with GraphNode(name="chat-workflow") as graph:
    prompt = PromptNode(
        name="prompt",
        inputs={
            "prompt": {"system": "Bạn là trợ lý AI.", "user": "{question}"},
            "question": PARENT["question"]
        }
    )
    llm = LLMNode(
        name="llm",
        resource_key="gpt-4o",
        inputs={"messages": prompt["messages"]},
        outputs={"content": PARENT["answer"]}
    )
    START >> prompt >> llm >> END

engine = Hush(graph)
result = await engine.run(inputs={"question": "Python là gì?"})
print(result["answer"])
```

## Bắt đầu từ đây

| Bước | Nội dung | Link |
|------|----------|------|
| 1 | Cài đặt Hush | [Installation](installation.md) |
| 2 | Hello World | [Quickstart](quickstart.md) |
| 3 | Workflow đầu tiên | [Tutorial](tutorials/01-first-workflow.md) |

## Tutorials (theo thứ tự)

Học Hush từ cơ bản đến nâng cao:

1. [Workflow đầu tiên](tutorials/01-first-workflow.md) - Cơ bản về nodes, edges và data flow
2. [Sử dụng LLM](tutorials/02-llm-basics.md) - PromptNode, LLMNode và ResourceHub
3. [Loops và Branches](tutorials/03-loops-branches.md) - ForLoop, MapNode, BranchNode
4. [Production](tutorials/04-production.md) - Tracing, error handling, deployment

## Guides (đọc khi cần)

Hướng dẫn theo tác vụ cụ thể:

- [Tích hợp LLM](guides/llm-integration.md) - Cấu hình và sử dụng LLM providers
- [Embeddings & Reranking](guides/embeddings-reranking.md) - Vector search và semantic ranking
- [Xử lý lỗi](guides/error-handling.md) - Error handling, retry và fallback
- [Thực thi song song](guides/parallel-execution.md) - Parallel và concurrent execution
- [Tracing & Debug](guides/tracing.md) - LocalTracer, Langfuse, observability

## Examples

Các ví dụ hoàn chỉnh, copy-paste được:

- [RAG Pipeline](examples/rag-workflow.md) - Retrieval-Augmented Generation
- [AI Agent](examples/agent-workflow.md) - Agent với tools
- [Multi-model](examples/multi-model.md) - Nhiều LLM providers

## Kiến trúc 3 lớp

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
| `hush-core` | Workflow engine cốt lõi - không có dependency nặng |
| `hush-providers` | AI/LLM providers (OpenAI, Azure, Gemini, etc.) |
| `hush-observability` | Tracing backends (LocalTracer, Langfuse, OTEL) |

## Cài đặt nhanh

```bash
# Standard - Core + LLM providers
pip install hush-ai[standard]

# Hoặc với uv (nhanh hơn)
uv add hush-ai[standard]
```

Xem thêm [Installation](installation.md) cho các tier cài đặt khác.

## Cho Developers

Nếu bạn muốn hiểu cách Hush hoạt động bên trong hoặc contribute:

→ [Architecture Documentation](../architecture/index.md)
