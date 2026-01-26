# State Architecture

Tài liệu này mô tả kiến trúc nội bộ của State system trong Hush.

## Overview

State system quản lý data flow giữa các nodes trong workflow:

```
┌─────────────────────────────────────────────────────────────────┐
│                         StateSchema                             │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Pydantic model định nghĩa structure của state             │  │
│  │                                                           │  │
│  │  class MyState(StateSchema):                              │  │
│  │      query: str                                           │  │
│  │      results: list = []                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      State Implementation                       │
│  ┌─────────────────────┐    ┌─────────────────────┐             │
│  │    MemoryState      │    │     RedisState      │             │
│  │   (In-memory)       │    │   (Distributed)     │             │
│  └─────────────────────┘    └─────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### StateSchema

Base class cho state definition, kế thừa từ Pydantic BaseModel:

```python
from hush.core import StateSchema

class WorkflowState(StateSchema):
    query: str
    context: list[str] = []
    response: str = ""
```

### MemoryState

Default state implementation, lưu data trong memory:

```python
class MemoryState:
    def __init__(self, schema: Type[StateSchema]):
        self._schema = schema
        self._data: Dict[str, Any] = {}

    def get(self, key: str) -> Any: ...
    def set(self, key: str, value: Any): ...
    def update(self, data: Dict[str, Any]): ...
```

### RedisState

Distributed state cho multi-process/multi-node deployments:

```python
class RedisState:
    def __init__(self, schema: Type[StateSchema], redis_client):
        self._schema = schema
        self._redis = redis_client
        self._prefix = f"hush:state:{uuid4()}"

    async def get(self, key: str) -> Any: ...
    async def set(self, key: str, value: Any): ...
```

## Reactive References

### Ref

Reference đến một field trong state:

```python
from hush.core import Ref

# Tạo reference
query_ref = Ref("query")

# Trong node definition
inputs={"user_query": Ref("query")}  # Lấy giá trị từ state.query
```

### Cell

Computed reference với transformation:

```python
from hush.core import Cell

# Transform value khi đọc
upper_query = Cell("query", transform=lambda x: x.upper())
```

## Data Flow

### Input Resolution

```
Node Execution
     │
     ├──▶ Resolve inputs
     │    ├── Ref("query") → state.get("query")
     │    ├── PARENT["field"] → parent_state.get("field")
     │    └── literal value → as-is
     │
     ├──▶ Execute node logic
     │
     └──▶ Write outputs
          ├── outputs={"result": PARENT} → parent_state.set("result", ...)
          └── outputs={"*": PARENT} → parent_state.update(all_outputs)
```

### State Propagation

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   GraphNode  │     │   GraphNode  │     │   GraphNode  │
│   (Parent)   │     │   (Child)    │     │   (Child)    │
│              │     │              │     │              │
│  ┌────────┐  │     │  ┌────────┐  │     │  ┌────────┐  │
│  │ State  │──┼────>│  │ State  │  │     │  │ State  │  │
│  └────────┘  │     │  └────────┘  │     │  └────────┘  │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
       │     PARENT["x"]    │                    │
       │<───────────────────┤                    │
       │                    │                    │
       │              Ref("y")                   │
       │                    │<───────────────────┤
```

## Special Markers

### PARENT

Reference đến parent graph's state:

```python
# Đọc từ parent
inputs={"data": PARENT["input_data"]}

# Ghi vào parent
outputs={"result": PARENT["output_result"]}
outputs={"*": PARENT}  # Ghi tất cả outputs
```

### START / END

Graph boundary markers:

```python
with GraphNode(name="workflow") as graph:
    node_a = CodeNode(...)
    node_b = CodeNode(...)

    START >> node_a >> node_b >> END
```

## State Isolation

Mỗi GraphNode có state riêng, isolated từ siblings:

```
┌─────────────────────────────────────────────────────────┐
│                    Root GraphNode                       │
│  State: {query: "...", final_result: "..."}             │
│                                                         │
│  ┌─────────────────┐    ┌─────────────────┐             │
│  │  SubGraph A     │    │  SubGraph B     │             │
│  │  State: {...}   │    │  State: {...}   │             │
│  │  (isolated)     │    │  (isolated)     │             │
│  └─────────────────┘    └─────────────────┘             │
└─────────────────────────────────────────────────────────┘
```

## Folder Structure

```
hush-core/hush/core/states/
├── __init__.py          # Public exports
├── schema.py            # StateSchema base class
├── memory.py            # MemoryState implementation
├── redis.py             # RedisState implementation
├── refs.py              # Ref, Cell classes
└── markers.py           # START, END, PARENT markers
```

## Design Decisions

1. **Pydantic-based Schema**: Sử dụng Pydantic cho validation và serialization tự động.

2. **Lazy Resolution**: References (Ref, Cell) chỉ được resolve tại execution time, không phải definition time.

3. **Hierarchical State**: Mỗi GraphNode có state riêng, communicate với parent qua PARENT marker.

4. **Pluggable Backends**: State interface cho phép swap giữa MemoryState và RedisState dễ dàng.

## Tiếp theo

- [Architecture Overview](overview.md) - Tổng quan kiến trúc
- [Registry Architecture](registry.md) - ResourceHub và ConfigRegistry
