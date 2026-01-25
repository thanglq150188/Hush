# State Management

## State là gì?

**State** trong Hush là nơi lưu trữ tất cả các biến được sử dụng trong quá trình vận hành workflow. Mỗi lần chạy workflow tạo ra một state mới, chứa:

- **Workflow inputs/outputs**: Dữ liệu đầu vào và kết quả cuối cùng của workflow
- **Node inputs/outputs**: Giá trị trung gian giữa các nodes trong quá trình thực thi
- **Metadata**: Timestamps, errors, execution traces, etc.

### Cách truy cập State

State được truy cập thông qua 3 thành phần:

| Thành phần | Mô tả | Ví dụ |
|------------|-------|-------|
| **node** | Tên đầy đủ của node | `"my-workflow.processor"` |
| **var** | Tên biến | `"result"` |
| **context_id** | ID context (cho iteration) | `None`, `"iter_0"`, `"iter_1"` |

```python
# Truy cập giá trị: state[node, var, context_id]
value = state["my-workflow.processor", "result", None]

# Ghi giá trị
state["my-workflow.processor", "result", None] = {"data": "hello"}
```

Giá trị được lưu trong **Cell** - một container hỗ trợ nhiều context (quan trọng cho iteration nodes).

## Kiến trúc State

Hush sử dụng hệ thống state management với 4 thành phần chính:

| Component | Mô tả |
|-----------|-------|
| **StateSchema** | Định nghĩa cấu trúc state với O(1) lookup |
| **MemoryState** | Lưu trữ giá trị trong bộ nhớ |
| **Ref** | Tham chiếu đến biến khác với khả năng chain operations |
| **Cell** | Lưu trữ giá trị đa context (cho iteration) |

```
┌─────────────────────────────────────────┐
│              MemoryState                │
│  ┌───────────────────────────────────┐  │
│  │           StateSchema             │  │
│  │  (node, var) → index mapping      │  │
│  │  pull_refs / push_refs            │  │
│  └───────────────────────────────────┘  │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │Cell │ │Cell │ │Cell │ │Cell │ ...   │
│  │ [0] │ │ [1] │ │ [2] │ │ [3] │       │
│  └─────┘ └─────┘ └─────┘ └─────┘       │
└─────────────────────────────────────────┘
```

## Luồng dữ liệu cơ bản

```
         inputs                      outputs
Workflow ────────> Node ────────> Workflow
 state              ↓               state
                  core()
```

1. Node đọc inputs từ state (pull)
2. Node thực thi `core()` function
3. Node ghi outputs ra state (push)

## Ref - Tham chiếu biến

`Ref` là cách khai báo kết nối dữ liệu giữa các nodes:

```python
from hush.core import PARENT, CodeNode

# Đọc biến "input_data" từ PARENT (graph cha)
node = CodeNode(
    name="processor",
    inputs={"data": PARENT["input_data"]}
)

# Đọc biến từ node khác
node_b = CodeNode(
    name="node_b",
    inputs={"value": node_a["output_key"]}
)
```

### Ref với Operations

Ref hỗ trợ chain các operations để transform dữ liệu:

```python
# Getitem - truy cập phần tử
PARENT["data"]["key"]       # data["key"]
PARENT["items"][0]          # items[0]

# Getattr - truy cập attribute
PARENT["text"].upper        # text.upper (method reference)
PARENT["text"].upper()      # text.upper() (method call)

# Arithmetic
PARENT["x"] + 10            # x + 10
PARENT["x"] * 2 - 5         # (x * 2) - 5

# Comparison (cho BranchNode)
PARENT["score"] >= 90       # score >= 90
PARENT["status"] == "active"  # status == "active"

# Apply custom function
PARENT["data"].apply(len)   # len(data)
PARENT["x"].apply(lambda v: v * 2)  # v * 2
PARENT["text"].apply(str.split, ",")  # text.split(",")
```

## Inputs Mapping

### Cú pháp cơ bản

```python
# Đọc từ PARENT
inputs={"x": PARENT["input_x"]}

# Đọc từ node khác
inputs={"data": other_node["output_data"]}

# Giá trị literal (static)
inputs={"limit": 100}

# Kết hợp
inputs={
    "query": PARENT["user_query"],
    "config": config_node["settings"],
    "max_results": 10
}
```

### Wildcard Forward (*)

Forward tất cả keys chưa được định nghĩa explicitly:

```python
# Forward tất cả từ PARENT
inputs={"*": PARENT}

# Kết hợp: explicit + wildcard
inputs={
    "query": PARENT["q"],  # Rename: PARENT["q"] → node["query"]
    "*": PARENT            # Forward remaining keys
}
```

### Tuple Keys

Map nhiều keys cùng nguồn:

```python
# Map cả "a" và "b" từ PARENT
inputs={("a", "b"): PARENT}
# Equivalent to:
# inputs={"a": PARENT["a"], "b": PARENT["b"]}
```

## Outputs Mapping

### Cú pháp cơ bản

```python
# Ghi ra PARENT với cùng tên
outputs={"result": PARENT}  # node.result → PARENT.result

# Ghi ra PARENT với tên khác (via Ref syntax)
node["internal_name"] >> PARENT["external_name"]

# Wildcard - forward tất cả outputs
outputs={"*": PARENT}
```

### Output không có mapping

Nếu không khai báo outputs, giá trị vẫn được lưu trong state và có thể truy cập qua:

```python
# Sau khi run
state["graph_name.node_name", "output_var", None]
```

## StateSchema

`StateSchema` định nghĩa cấu trúc state của workflow với O(1) lookup:

```python
from hush.core.states import StateSchema

# Tự động tạo từ graph
schema = StateSchema(graph)

# Debug cấu trúc
schema.show()
```

Output của `schema.show()`:

```
=== StateSchema: my-workflow ===
my-workflow.input_x [0] = None
my-workflow.processor.data [1] <- pull my-workflow[0]
my-workflow.processor.result [2] -> push my-workflow.output[3]
my-workflow.output [3] = None
Tổng: 4 biến
```

### Pull vs Push Refs

- **Pull ref**: Node đọc (pull) giá trị từ source khi cần
- **Push ref**: Node ghi (push) giá trị đến target sau khi execute

```python
# Pull ref (trong inputs)
inputs={"data": PARENT["input"]}  # Khi node đọc "data", pull từ PARENT["input"]

# Push ref (trong outputs)
outputs={"result": PARENT}  # Khi node ghi "result", push đến PARENT["result"]
```

## MemoryState

`MemoryState` lưu trữ giá trị thực tế trong bộ nhớ:

```python
from hush.core.states import StateSchema, MemoryState

# Tạo state từ schema
schema = StateSchema(graph)
state = schema.create_state(inputs={"x": 5})

# Hoặc
state = MemoryState(schema, inputs={"x": 5})

# Truy cập giá trị: state[node, var, context_id]
state["my-workflow", "x", None]  # → 5

# Debug state
state.show()
```

### Truy cập State

```python
# Get value
value = state["node_name", "var_name", context_id]

# Set value
state["node_name", "var_name", context_id] = value

# Check if exists
state.has("node_name", "var_name", context_id)

# Get Cell object (cho advanced use)
cell = state.get_cell("node_name", "var_name")
```

### Context ID

Context ID được dùng cho iteration nodes (ForLoop, Map, etc.):

```python
# Không có context (single execution)
state["node", "var", None]

# Với context (iteration)
state["loop.inner_node", "result", "iter_0"]
state["loop.inner_node", "result", "iter_1"]
```

Mỗi iteration tạo context_id riêng để không ghi đè lên nhau.

## Cell - Lưu trữ đa context

Cell là container lưu giá trị với hỗ trợ nhiều context:

```python
from hush.core.states import Cell

cell = Cell(default_value=0)

# Ghi vào các context khác nhau
cell[None] = 10          # context mặc định
cell["iter_0"] = 20      # context "iter_0"
cell["iter_1"] = 30      # context "iter_1"

# Đọc
cell[None]      # → 10
cell["iter_0"]  # → 20
cell["iter_1"]  # → 30
```

## Ví dụ Data Flow

### Pipeline đơn giản

```python
with GraphNode(name="pipeline") as graph:
    # Node 1: Fetch data
    fetch = CodeNode(
        name="fetch",
        code_fn=lambda url: {"data": requests.get(url).json()},
        inputs={"url": PARENT["api_url"]}
    )

    # Node 2: Process data (đọc từ fetch)
    process = CodeNode(
        name="process",
        code_fn=lambda data: {"cleaned": clean_data(data)},
        inputs={"data": fetch["data"]}
    )

    # Node 3: Save (đọc từ process, ghi ra PARENT)
    save = CodeNode(
        name="save",
        code_fn=lambda cleaned: {"saved": save_to_db(cleaned)},
        inputs={"cleaned": process["cleaned"]},
        outputs={"saved": PARENT}
    )

    START >> fetch >> process >> save >> END

graph.build()
```

**Data flow:**
```
PARENT["api_url"] → fetch["url"]
                          ↓
                    fetch["data"] → process["data"]
                                          ↓
                                    process["cleaned"] → save["cleaned"]
                                                               ↓
                                                         save["saved"] → PARENT["saved"]
```

### Parallel với merge

```python
with GraphNode(name="parallel") as graph:
    # Fork: 3 nodes đọc cùng input
    a = CodeNode(
        name="process_a",
        code_fn=lambda x: {"result": x + 1},
        inputs={"x": PARENT["value"]}
    )
    b = CodeNode(
        name="process_b",
        code_fn=lambda x: {"result": x * 2},
        inputs={"x": PARENT["value"]}
    )
    c = CodeNode(
        name="process_c",
        code_fn=lambda x: {"result": x ** 2},
        inputs={"x": PARENT["value"]}
    )

    # Merge: đọc từ cả 3
    merge = CodeNode(
        name="merge",
        code_fn=lambda a, b, c: {"sum": a + b + c},
        inputs={
            "a": a["result"],
            "b": b["result"],
            "c": c["result"],
        },
        outputs={"sum": PARENT}
    )

    START >> [a, b, c] >> merge >> END
```

**Data flow:**
```
PARENT["value"] ──┬──> a["x"] ──> a["result"] ──┐
                  ├──> b["x"] ──> b["result"] ──┼──> merge ──> PARENT["sum"]
                  └──> c["x"] ──> c["result"] ──┘
```

## Metadata và Tracing

MemoryState cũng lưu metadata cho observability:

```python
# IDs
state.user_id       # UUID của user
state.session_id    # UUID của session
state.request_id    # UUID của request này

# Execution order
state.execution_order  # List các nodes đã execute

# Trace metadata
state.trace_metadata   # Dict metadata cho từng node

# Dynamic tags
state.add_tag("cache-hit")
state.add_tags(["error", "retry"])
```

## Best Practices

### 1. Sử dụng explicit inputs/outputs

```python
# Rõ ràng - dễ trace
inputs={"query": PARENT["q"], "limit": PARENT["max"]}
outputs={"results": PARENT, "count": PARENT}

# Ít rõ ràng - khó debug
inputs={"*": PARENT}
outputs={"*": PARENT}
```

### 2. Đặt tên biến nhất quán

```python
# Tốt - cùng tên xuyên suốt pipeline
node_a: outputs "data"
node_b: inputs {"data": node_a["data"]}

# Tránh - rename không cần thiết
node_a: outputs "data"
node_b: inputs {"input": node_a["data"]}  # Rename "data" → "input"
```

### 3. Tận dụng Ref operations

```python
# Thay vì tạo node chỉ để transform
transform = CodeNode(
    name="transform",
    code_fn=lambda x: {"y": x["key"].upper()},
    inputs={"x": PARENT["data"]}
)

# Dùng Ref operations
processor = CodeNode(
    name="processor",
    inputs={"text": PARENT["data"]["key"].upper()},  # Trực tiếp trong Ref
    ...
)
```

## Tiếp theo

- [ResourceHub](resource-hub.md) - Cấu hình và quản lý resources
- [Tracing](tracing.md) - Debug và monitor workflows
