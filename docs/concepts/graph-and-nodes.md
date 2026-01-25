# Graph và Nodes

## GraphNode là gì?

**GraphNode** là container chứa và quản lý một workflow. Mỗi workflow trong Hush là một GraphNode, có thể chứa nhiều node con và các node con này có thể chứa GraphNode khác (nested).

```python
from hush.core import GraphNode, CodeNode, START, END

# Tạo workflow với context manager
with GraphNode(name="my-workflow") as graph:
    step1 = CodeNode(name="step1", code_fn=lambda: {"x": 1})
    step2 = CodeNode(name="step2", code_fn=lambda x: {"result": x * 2})

    # Kết nối nodes
    START >> step1 >> step2 >> END

# Build và chạy
graph.build()
```

### Đặc điểm của GraphNode

1. **Context manager**: Sử dụng `with GraphNode() as graph:` để tự động đăng ký nodes
2. **Parallel execution**: Nodes không phụ thuộc nhau sẽ chạy song song tự động
3. **Nested support**: GraphNode có thể chứa GraphNode khác
4. **Auto schema**: Inputs/outputs tự động infer từ nodes con

## START, END, PARENT

### START và END

`START` và `END` là special markers để đánh dấu điểm bắt đầu và kết thúc của workflow:

```python
from hush.core import START, END

# Node sau START là entry point
START >> first_node

# Node trước END là exit point
last_node >> END

# Nhiều entry/exit points
START >> [node_a, node_b]  # 2 entry points (chạy song song)
[node_x, node_y] >> END    # 2 exit points
```

### PARENT

`PARENT` là reference đến graph cha, dùng để:

1. **Đọc inputs từ workflow**:
```python
node = CodeNode(
    name="processor",
    inputs={"data": PARENT["input_data"]}  # Đọc từ workflow input
)
```

2. **Ghi outputs ra workflow**:
```python
node = CodeNode(
    name="processor",
    outputs={"result": PARENT}  # Ghi ra workflow output
)
```

3. **Forward tất cả keys với wildcard**:
```python
node = CodeNode(
    inputs={
        "explicit_key": PARENT["key1"],
        "*": PARENT  # Forward tất cả keys khác
    }
)
```

## Các loại Node

### 1. CodeNode

Thực thi Python function. Hush tự động:
- Parse inputs từ function parameters
- Parse outputs từ return dict
- Wrap sync function thành async

```python
from hush.core import CodeNode

# Cách 1: Inline function
node = CodeNode(
    name="calculate",
    code_fn=lambda x, y: {"sum": x + y, "product": x * y}
)

# Cách 2: Named function
def process_data(text: str, limit: int = 100) -> dict:
    """Xử lý text data."""
    processed = text.strip()[:limit]
    return {"output": processed, "length": len(processed)}

node = CodeNode(name="process", code_fn=process_data)

# Cách 3: Decorator @code_node
from hush.core import code_node

@code_node
def my_function(a, b):
    return {"result": a + b}

# Sử dụng như factory
node = my_function(inputs={"a": PARENT["x"], "b": 10})
```

### 2. PromptNode (hush-providers)

Build messages từ templates để gửi cho LLM:

```python
from hush.providers.nodes import PromptNode

# Cách 1: String prompt (user message)
prompt = PromptNode(
    name="prompt",
    inputs={
        "prompt": "Tóm tắt văn bản: {text}",
        "text": PARENT["document"]
    }
)

# Cách 2: Dict với system/user
prompt = PromptNode(
    name="prompt",
    inputs={
        "prompt": {
            "system": "Bạn là trợ lý chuyên {role}.",
            "user": "Hãy giúp tôi: {task}"
        },
        "role": "viết code",
        "task": PARENT["user_request"]
    }
)

# Cách 3: Full messages array (multimodal)
prompt = PromptNode(
    name="prompt",
    inputs={
        "prompt": [
            {"role": "system", "content": "You are a vision assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image:"},
                {"type": "image_url", "image_url": {"url": "{image_url}"}}
            ]}
        ],
        "image_url": PARENT["image"]
    }
)
```

### 3. LLMNode (hush-providers)

Gọi LLM APIs (OpenAI, Azure, Gemini, etc.):

```python
from hush.providers.nodes import LLMNode

llm = LLMNode(
    name="llm",
    resource="llm:gpt-4o",  # Tham chiếu từ ResourceHub
    inputs={
        "messages": prompt["messages"],
        "temperature": 0.7,
        "max_tokens": 1000
    },
    outputs={"content": PARENT["response"]}
)
```

### 4. BranchNode

Conditional routing - định tuyến workflow theo điều kiện:

```python
from hush.core import BranchNode

# Cách 1: String conditions
branch = BranchNode(
    name="router",
    cases={
        "score >= 90": "excellent",
        "score >= 70": "good",
        "score >= 50": "average",
    },
    default="fail",
    inputs={"score": PARENT["score"]}
)

# Kết nối: branch chỉ trigger 1 target được chọn (special-cased trong execution)
START >> branch
branch >> [excellent_node, good_node, average_node, fail_node]
[excellent_node, good_node, average_node, fail_node] >> END
```

```python
# Cách 2: Fluent builder với Ref
from hush.core.nodes.flow.branch_node import Branch

branch = (Branch("router")
    .if_(PARENT["score"] >= 90, "excellent")
    .if_(PARENT["score"] >= 70, "good")
    .otherwise("fail"))
```

### 5. ForLoopNode

Iterate tuần tự - xử lý từng item một:

```python
from hush.core import ForLoopNode
from hush.core.nodes.iteration.base import Each

with ForLoopNode(
    name="process_items",
    inputs={
        "item": Each(PARENT["items"]),  # Iterate qua mỗi item
        "config": PARENT["config"]       # Broadcast cho tất cả iterations
    }
) as loop:
    process = CodeNode(
        name="process",
        code_fn=lambda item, config: {"result": transform(item, config)}
    )
    START >> process >> END
```

### 6. MapNode

Iterate song song - xử lý nhiều items cùng lúc:

```python
from hush.core import MapNode
from hush.core.nodes.iteration.base import Each

with MapNode(
    name="parallel_process",
    inputs={
        "url": Each(PARENT["urls"]),  # Iterate song song
        "timeout": 30                  # Broadcast
    },
    max_concurrency=10  # Giới hạn concurrent tasks
) as map_node:
    fetch = CodeNode(
        name="fetch",
        code_fn=lambda url, timeout: {"data": requests.get(url, timeout=timeout).json()}
    )
    START >> fetch >> END
```

### Khi nào dùng ForLoopNode vs MapNode?

| Tiêu chí | ForLoopNode | MapNode |
|----------|-------------|---------|
| Execution | Tuần tự (sequential) | Song song (parallel) |
| Dependencies | Items có thể phụ thuộc nhau | Items độc lập |
| Memory | Thấp hơn | Cao hơn (nhiều tasks cùng lúc) |
| Use case | Chain processing, stateful | I/O bound, batch processing |

## Kết nối Nodes với Edges

### Hard Edge (>>)

Kết nối tuần tự - node sau chờ node trước hoàn thành:

```python
# Linear flow
START >> a >> b >> c >> END

# Fork pattern (parallel)
START >> a >> [b1, b2, b3] >> merge >> END
# a chạy xong → b1, b2, b3 chạy song song → merge chờ TẤT CẢ

# Diamond pattern
START >> a >> [b, c] >> d >> END
```

### Soft Edge (~)

Soft edge đánh dấu kết nối "mềm" - node đích chỉ cần **BẤT KỲ MỘT** soft predecessor hoàn thành, không cần chờ tất cả.

**Use case chính:** Merge node sau nhiều predecessors nhưng chỉ một số thực sự chạy.

```python
# Pattern: Fork-Merge với chỉ 1 nhánh chạy
START >> a

# a fork ra b, c, d nhưng chỉ 1 trong 3 thực sự chạy (ví dụ do logic bên ngoài)
a >> ~b >> merge
a >> ~c >> merge
a >> ~d >> merge

# merge chờ BẤT KỲ 1 soft predecessor (b, c, hoặc d)
merge >> END
```

**Cách ready_count hoạt động:**
- Hard edge: đếm riêng từng predecessor
- Soft edge: tất cả soft predecessors đếm chung là 1

```python
# Ví dụ: A >> D, B >> ~D, C >> ~D
# ready_count[D] = 2 (1 từ hard edge A, 1 từ tất cả soft edges B+C)
# D chạy khi: A hoàn thành AND (B HOẶC C hoàn thành)
```

**Với BranchNode:** BranchNode được special-cased - chỉ trigger 1 target được chọn:

```python
# BranchNode tự động chỉ trigger 1 target
START >> branch
branch >> [excellent, good, fail]  # Hard edges OK - branch chỉ trigger 1

# Nếu cần merge node SAU các branch targets:
[excellent, good, fail] >> ~merge  # Soft edges - merge chờ ANY 1
merge >> END
```

### Data Flow với node["key"]

Đọc output của node khác:

```python
# node_a outputs: {"x": 1, "y": 2}
# node_b đọc x từ node_a
node_b = CodeNode(
    name="node_b",
    inputs={"value": node_a["x"]}  # value = 1
)

# Output mapping
node["output_key"] >> PARENT["workflow_output"]
```

## Parallel Execution

Hush tự động phát hiện và chạy song song các nodes độc lập:

```python
with GraphNode(name="parallel-demo") as graph:
    # a, b, c không phụ thuộc nhau → chạy song song
    a = CodeNode(name="a", code_fn=lambda: {"result": "a"})
    b = CodeNode(name="b", code_fn=lambda: {"result": "b"})
    c = CodeNode(name="c", code_fn=lambda: {"result": "c"})

    # d phụ thuộc a, b, c → chờ tất cả hoàn thành
    d = CodeNode(
        name="d",
        code_fn=lambda a_result, b_result, c_result: {
            "combined": f"{a_result}-{b_result}-{c_result}"
        },
        inputs={
            "a_result": a["result"],
            "b_result": b["result"],
            "c_result": c["result"],
        }
    )

    START >> [a, b, c] >> d >> END
```

**Execution order:**
1. `a`, `b`, `c` chạy song song
2. Khi cả 3 hoàn thành → `d` chạy

## Nested Graphs

GraphNode có thể chứa GraphNode khác:

```python
# Inner graph
with GraphNode(name="inner") as inner_graph:
    step = CodeNode(name="step", code_fn=lambda x: {"y": x * 2})
    START >> step >> END

# Outer graph sử dụng inner graph như một node
with GraphNode(name="outer") as outer_graph:
    prepare = CodeNode(name="prepare", code_fn=lambda: {"x": 10})

    # inner_graph được treat như một node
    START >> prepare >> inner_graph >> END
```

## Best Practices

### 1. Đặt tên node có ý nghĩa

```python
# Tốt
fetch_user = CodeNode(name="fetch_user", ...)
validate_input = CodeNode(name="validate_input", ...)

# Tránh
n1 = CodeNode(name="node1", ...)
x = CodeNode(name="x", ...)
```

### 2. Sử dụng outputs mapping rõ ràng

```python
# Rõ ràng - biết chính xác output nào được expose
node = CodeNode(
    name="process",
    code_fn=process_fn,
    outputs={"result": PARENT, "status": PARENT}
)

# Ít rõ ràng - tất cả outputs của function được forward
node = CodeNode(
    name="process",
    code_fn=process_fn,
    outputs={"*": PARENT}  # Wildcard
)
```

### 3. Tách logic phức tạp thành nested graph

```python
# Thay vì 1 graph lớn với nhiều nodes
# Tách thành các subgraphs có ý nghĩa

with GraphNode(name="main") as main:
    with GraphNode(name="validation") as validation:
        # Validation logic
        START >> validate_schema >> validate_permissions >> END

    with GraphNode(name="processing") as processing:
        # Processing logic
        START >> transform >> enrich >> save >> END

    START >> validation >> processing >> END
```

### 4. Soft edges cho merge sau conditional paths

```python
# Khi có nhiều paths nhưng chỉ 1 chạy → dùng soft edges cho merge
[path_a, path_b, path_c] >> ~merge_node
```

## Tiếp theo

- [State Management](state-management.md) - Quản lý state trong workflow
- [ResourceHub](resource-hub.md) - Cấu hình và quản lý resources
- [Tracing](tracing.md) - Debug và monitor workflows
