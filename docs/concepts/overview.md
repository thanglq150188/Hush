# Kiến trúc tổng quan

## Hush là gì?

**Hush** là một **workflow engine** cho các ứng dụng AI/LLM, được thiết kế để xây dựng các pipeline phức tạp một cách đơn giản và hiệu quả.

**Workflow engine, không phải framework**: Hush tập trung vào việc điều phối (orchestration) và thực thi (execution) các node trong một graph. Bạn không cần học một framework mới - chỉ cần định nghĩa các node và kết nối chúng lại với nhau.

```python
from hush.core import Hush, GraphNode, CodeNode, START, END

with GraphNode(name="my-workflow") as graph:
    step1 = CodeNode(name="step1", code_fn=lambda: {"data": "hello"})
    step2 = CodeNode(name="step2", code_fn=lambda data: {"result": data.upper()})

    START >> step1 >> step2 >> END

engine = Hush(graph)
result = await engine.run()  # {"result": "HELLO"}
```

## Kiến trúc 3 lớp

Hush được tổ chức thành 3 package độc lập, cho phép cài đặt theo nhu cầu:

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

### Layer 1: hush-core

**Workflow engine cốt lõi** - không có dependency nặng.

Bao gồm:
- **Execution engine**: Class `Hush` để chạy workflow
- **Graph nodes**: `GraphNode` - container quản lý subgraph
- **Transform nodes**: `CodeNode`, `ParserNode`
- **Flow control**: `BranchNode` - conditional routing
- **Iteration nodes**: `ForLoopNode`, `MapNode`, `WhileLoopNode`, `AsyncIterNode`
- **State management**: `StateSchema`, `MemoryState`, `Ref`, `Cell`
- **Resource hub**: Registry quản lý configs (LLM, database, etc.)

```python
# Chỉ cần core cho workflow logic
from hush.core import (
    Hush, GraphNode, CodeNode, BranchNode,
    ForLoopNode, MapNode, START, END, PARENT
)
```

### Layer 2: hush-providers

**AI/LLM providers** - cung cấp các node tích hợp với LLM services.

Bao gồm:
- `PromptNode`: Build messages từ templates
- `LLMNode`: Gọi LLM APIs (OpenAI, Azure, Gemini, Bedrock)
- `EmbeddingNode`: Tạo vector embeddings
- `RerankNode`: Reranking cho search results
- `LLMChainNode`: Kết hợp PromptNode + LLMNode

```python
# Thêm providers cho AI features
from hush.providers.nodes import PromptNode, LLMNode, EmbeddingNode
```

### Layer 3: hush-observability

**Monitoring và debugging** - tích hợp các observability platforms.

Bao gồm:
- `LocalTracer`: SQLite-based tracing với Web UI
- `LangfuseTracer`: Tích hợp Langfuse
- `OTelTracer`: OpenTelemetry support

```python
# Thêm observability cho production
from hush.observability import LocalTracer, LangfuseTracer
```

## Design Principles

### 1. Async-first (Transparent)

Hush engine chạy async internally để hỗ trợ parallel execution, nhưng **bạn không cần viết async code**. Hush tự động wrap sync functions:

```python
# Viết function bình thường - KHÔNG cần async
def process_data(text: str) -> dict:
    result = text.upper()
    return {"output": result}

node = CodeNode(name="processor", code_fn=process_data)

# Hush tự động:
# 1. Wrap sync function thành async
# 2. Chạy parallel khi không có dependency
# 3. Handle concurrent execution
```

Nếu bạn CÓ async function (ví dụ: gọi API), Hush cũng hỗ trợ:

```python
# Async function cũng được hỗ trợ
async def fetch_data(url: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return {"data": await resp.json()}

node = CodeNode(name="fetcher", code_fn=fetch_data)
```

### 2. Composition over Inheritance

Thay vì tạo class hierarchy phức tạp, Hush sử dụng composition:

```python
# Graph có thể chứa graph khác (nested)
with GraphNode(name="outer") as outer:
    with GraphNode(name="inner") as inner:
        node1 = CodeNode(...)
        START >> node1 >> END

    START >> inner >> END

# Nodes kết hợp thông qua edges, không qua inheritance
START >> [node_a, node_b] >> merge_node >> END
```

### 3. Lazy Loading

Resources chỉ được load khi cần thiết:

```python
from hush.core.resources import ResourceHub

# Config được đọc nhưng chưa khởi tạo client
hub = ResourceHub.from_yaml("resources.yaml")

# Client chỉ được tạo khi gọi get()
llm_config = hub.get("llm:gpt-4o")  # Khởi tạo tại đây
```

### 4. Type-safe với Runtime Validation

Mỗi node có `INPUT_SCHEMA` và `OUTPUT_SCHEMA`:

```python
class LLMNode(BaseNode):
    INPUT_SCHEMA = {
        'messages': Param(type=list, required=True),
        'model': Param(type=str, required=False),
        'temperature': Param(type=float, required=False, default=0.7),
    }

    OUTPUT_SCHEMA = {
        'content': Param(type=str, required=True),
        'usage': Param(type=dict, required=False),
    }
```

Schema được validate tại:
- Build time: Khi gọi `graph.build()`
- Runtime: Trước và sau khi node execute

### 5. Declarative Data Flow

Data flow được khai báo rõ ràng qua syntax `PARENT`:

```python
# Đọc từ parent scope
node = CodeNode(
    name="processor",
    inputs={"data": PARENT["input_data"]},  # Lấy data từ parent
    outputs={"result": PARENT}               # Ghi result ra parent
)

# Đọc output của node khác
START >> node_a >> node_b
node_b = CodeNode(
    inputs={"value": node_a["output_key"]},  # Lấy từ node_a
    ...
)
```

## Package Structure

```
hush/
├── hush-core/                    # Core workflow engine
│   └── hush/core/
│       ├── __init__.py           # Public exports
│       ├── engine.py             # Hush execution engine
│       ├── nodes/                # Node implementations
│       │   ├── base.py           # BaseNode, START, END, PARENT
│       │   ├── graph/            # GraphNode
│       │   ├── flow/             # BranchNode
│       │   ├── iteration/        # ForLoop, Map, While, AsyncIter
│       │   └── transform/        # CodeNode, ParserNode
│       ├── states/               # State management
│       │   ├── schema.py         # StateSchema
│       │   └── memory_state.py   # MemoryState implementation
│       ├── resources/            # ResourceHub
│       └── configs/              # Configuration classes
│
├── hush-providers/               # AI/LLM providers
│   └── hush/providers/
│       ├── nodes/                # Provider nodes
│       │   ├── llm.py            # LLMNode
│       │   ├── prompt.py         # PromptNode
│       │   ├── embedding.py      # EmbeddingNode
│       │   └── rerank.py         # RerankNode
│       └── configs/              # Provider configs
│
├── hush-observability/           # Monitoring & tracing
│   └── hush/observability/
│       ├── local_tracer.py       # SQLite + Web UI
│       ├── langfuse_tracer.py    # Langfuse integration
│       └── otel_tracer.py        # OpenTelemetry
│
└── docs/                         # Documentation
```

## So sánh với các công cụ khác

| Feature | Hush | LangGraph | Prefect |
|---------|------|-----------|---------|
| **Focus** | AI/LLM workflows | Agent graphs | General workflows |
| **Execution** | Async-first | Sync by default | Async support |
| **State** | Built-in StateSchema | External state | Task-level state |
| **Tracing** | Built-in LocalTracer | LangSmith | Prefect Cloud |
| **Package size** | Lightweight (core ~50KB) | Medium | Heavy |
| **Learning curve** | Low (Python-native syntax) | Medium | Medium |

**Khi nào chọn Hush?**
- Cần workflow engine nhẹ, dễ tích hợp
- Ưu tiên async/parallel execution
- Muốn control chi tiết data flow
- Cần built-in tracing mà không phụ thuộc external service

## Tiếp theo

- [Graph và Nodes](graph-and-nodes.md) - Chi tiết về nodes và cách kết nối
- [State Management](state-management.md) - Quản lý state trong workflow
- [ResourceHub](resource-hub.md) - Cấu hình và quản lý resources
