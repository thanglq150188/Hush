# Xây dựng Workflows

Hướng dẫn này sẽ giúp bạn xây dựng workflows phức tạp với nhiều patterns khác nhau.

## Patterns cơ bản

### 1. Sequential Pipeline

Pipeline tuần tự là pattern đơn giản nhất - các nodes chạy lần lượt.

```python
from hush.core import GraphNode, CodeNode, START, END, PARENT

with GraphNode(name="sequential-pipeline") as graph:
    step1 = CodeNode(
        name="step1",
        code_fn=lambda x: {"result": x + 10},
        inputs={"x": PARENT["x"]}
    )
    step2 = CodeNode(
        name="step2",
        code_fn=lambda x: {"result": x * 2},
        inputs={"x": step1["result"]}
    )
    step3 = CodeNode(
        name="step3",
        code_fn=lambda x: {"result": x - 5},
        inputs={"x": step2["result"]},
        outputs={"*": PARENT}
    )

    START >> step1 >> step2 >> step3 >> END
```

**Kết quả**: Input `x=5` → `(5+10)*2-5 = 25`

### 2. Parallel Branches (Fan-out / Fan-in)

Chạy nhiều branches song song và merge kết quả.

```python
with GraphNode(name="parallel-branches") as graph:
    # Fork node
    prepare = CodeNode(
        name="prepare",
        code_fn=lambda x: {"value": x},
        inputs={"x": PARENT["x"]}
    )

    # Parallel branches
    branch_a = CodeNode(
        name="branch_a",
        code_fn=lambda x: {"result": x * 2},
        inputs={"x": prepare["value"]}
    )
    branch_b = CodeNode(
        name="branch_b",
        code_fn=lambda x: {"result": x * 3},
        inputs={"x": prepare["value"]}
    )
    branch_c = CodeNode(
        name="branch_c",
        code_fn=lambda x: {"result": x * 4},
        inputs={"x": prepare["value"]}
    )

    # Merge node - receives from ALL branches
    merge = CodeNode(
        name="merge",
        code_fn=lambda a, b, c: {"total": a + b + c},
        inputs={
            "a": branch_a["result"],
            "b": branch_b["result"],
            "c": branch_c["result"]
        },
        outputs={"*": PARENT}
    )

    # Syntax: [list] fans out, then merges
    START >> prepare >> [branch_a, branch_b, branch_c] >> merge >> END
```

**Kết quả**: Input `x=10` → `(10*2) + (10*3) + (10*4) = 90`

### 3. Conditional Routing (BranchNode)

Routing dựa trên conditions sử dụng `Branch` fluent API.

```python
from hush.core.nodes.flow.branch_node import Branch

with GraphNode(name="conditional-routing") as graph:
    # Evaluate input
    evaluate = CodeNode(
        name="evaluate",
        code_fn=lambda score: {"score": score},
        inputs={"score": PARENT["score"]}
    )

    # Branch với fluent syntax
    router = (Branch("grade_router")
        .if_(evaluate["score"] >= 90, "excellent")
        .if_(evaluate["score"] >= 70, "good")
        .if_(evaluate["score"] >= 50, "average")
        .otherwise("fail"))

    # Target nodes
    excellent = CodeNode(
        name="excellent",
        code_fn=lambda: {"grade": "A", "message": "Xuất sắc!"}
    )
    good = CodeNode(
        name="good",
        code_fn=lambda: {"grade": "B", "message": "Tốt!"}
    )
    average = CodeNode(
        name="average",
        code_fn=lambda: {"grade": "C", "message": "Trung bình"}
    )
    fail = CodeNode(
        name="fail",
        code_fn=lambda: {"grade": "F", "message": "Cần cố gắng hơn"}
    )

    # Merge node - uses soft edges since only ONE branch runs
    merge = CodeNode(
        name="merge",
        code_fn=lambda grade=None, message=None: {"grade": grade, "message": message},
        inputs={
            "grade": excellent["grade"],  # Will be None if not executed
            "message": excellent["message"]
        },
        outputs={"*": PARENT}
    )

    # BranchNode only triggers ONE target
    START >> evaluate >> router >> [excellent, good, average, fail]

    # Soft edges (~) for merge - only needs ONE predecessor to run
    [excellent, good, average, fail] >> ~merge
    merge >> END
```

**Quan trọng về Soft Edges**:
- BranchNode chỉ trigger MỘT target (không phải tất cả)
- Merge node cần soft edges (`>> ~`) vì không phải tất cả predecessors sẽ chạy
- Hard edges (`>>`) yêu cầu TẤT CẢ predecessors phải hoàn thành
- Soft edges (`>> ~`) chỉ cần BẤT KỲ MỘT predecessor hoàn thành

### 4. Loop Patterns

#### ForLoopNode - Iterate qua collection

```python
from hush.core.nodes.iteration.for_loop_node import ForLoopNode
from hush.core.nodes.iteration.base import Each

with GraphNode(name="for-loop-example") as graph:
    with ForLoopNode(
        name="process_items",
        inputs={"item": Each(PARENT["items"])}
    ) as loop:
        process = CodeNode(
            name="process",
            code_fn=lambda item: {"processed": item * 2},
            inputs={"item": PARENT["item"]}
        )
        process["processed"] >> PARENT["processed"]
        START >> process >> END

    loop["processed"] >> PARENT["results"]
    START >> loop >> END
```

**Kết quả**: `items=[1,2,3]` → `results=[2,4,6]`

#### MapNode - Parallel iteration

```python
from hush.core.nodes.iteration.map_node import MapNode
from hush.core.nodes.iteration.base import Each

with GraphNode(name="map-example") as graph:
    with MapNode(
        name="parallel_process",
        inputs={"item": Each(PARENT["items"])},
        max_concurrency=5  # Limit concurrent executions
    ) as map_node:
        process = CodeNode(
            name="process",
            code_fn=lambda item: {"result": item ** 2},
            inputs={"item": PARENT["item"]}
        )
        process["result"] >> PARENT["result"]
        START >> process >> END

    map_node["result"] >> PARENT["results"]
    START >> map_node >> END
```

#### WhileLoopNode - Loop until condition

```python
from hush.core.nodes.iteration.while_loop_node import WhileLoopNode

with GraphNode(name="while-loop-example") as graph:
    with WhileLoopNode(
        name="accumulate",
        inputs={"counter": 0, "sum": 0},
        stop_condition="sum >= 100",
        max_iterations=50
    ) as loop:
        step = CodeNode(
            name="step",
            code_fn=lambda counter, sum: {
                "new_counter": counter + 1,
                "new_sum": sum + counter
            },
            inputs={
                "counter": PARENT["counter"],
                "sum": PARENT["sum"]
            }
        )
        step["new_counter"] >> PARENT["counter"]
        step["new_sum"] >> PARENT["sum"]
        START >> step >> END

    loop["sum"] >> PARENT["final_sum"]
    loop["counter"] >> PARENT["iterations"]
    START >> loop >> END
```

## Nested Workflows (Subgraph)

GraphNode có thể chứa GraphNode khác, tạo nested workflows.

```python
# Inner workflow
with GraphNode(name="text-processor") as text_processor:
    clean = CodeNode(
        name="clean",
        code_fn=lambda text: {"cleaned": text.strip().lower()},
        inputs={"text": PARENT["text"]}
    )
    tokenize = CodeNode(
        name="tokenize",
        code_fn=lambda text: {"tokens": text.split()},
        inputs={"text": clean["cleaned"]},
        outputs={"*": PARENT}
    )
    START >> clean >> tokenize >> END

# Outer workflow uses inner as subgraph
with GraphNode(name="document-pipeline") as pipeline:
    with GraphNode(
        name="processor",
        inputs={"text": PARENT["document"]}
    ) as processor:
        clean = CodeNode(
            name="clean",
            code_fn=lambda text: {"cleaned": text.strip().lower()},
            inputs={"text": PARENT["text"]}
        )
        tokenize = CodeNode(
            name="tokenize",
            code_fn=lambda text: {"tokens": text.split()},
            inputs={"text": clean["cleaned"]}
        )
        tokenize["tokens"] >> PARENT["tokens"]
        START >> clean >> tokenize >> END

    analyze = CodeNode(
        name="analyze",
        code_fn=lambda tokens: {"word_count": len(tokens)},
        inputs={"tokens": processor["tokens"]},
        outputs={"*": PARENT}
    )

    START >> processor >> analyze >> END
```

## Output Mapping Syntax

Hush hỗ trợ nhiều cách để map outputs.

### Cách 1: Trong constructor (truyền thống)

```python
node = CodeNode(
    name="compute",
    code_fn=lambda x: {"a": x+1, "b": x+2},
    inputs={"x": PARENT["x"]},
    outputs={"*": PARENT}  # Map tất cả outputs
)
```

### Cách 2: Operator >> (mới)

```python
node = CodeNode(
    name="compute",
    code_fn=lambda x: {"a": x+1, "b": x+2},
    inputs={"x": PARENT["x"]}
)
# Map specific outputs
node["a"] >> PARENT["result_a"]
node["b"] >> PARENT["result_b"]
```

### Cách 3: Node-to-Node mapping

```python
producer = CodeNode(
    name="producer",
    code_fn=lambda: {"value": 42},
    inputs={}
)
consumer = CodeNode(
    name="consumer",
    code_fn=lambda input_val: {"result": input_val * 2},
    inputs={}
)
# Map producer output to consumer input
producer["value"] >> consumer["input_val"]
```

## Composing Reusable Components

### Sử dụng @code_node decorator

```python
from hush.core import code_node

@code_node
def validate_input(data: dict):
    """Validate input data."""
    if not data:
        return {"valid": False, "error": "Empty data"}
    return {"valid": True, "validated": data}

@code_node
def transform_data(data: dict):
    """Transform data."""
    return {"transformed": {k.upper(): v for k, v in data.items()}}

@code_node
def save_result(data: dict):
    """Save result."""
    return {"saved": True, "id": "abc123"}

# Use in workflow
with GraphNode(name="data-pipeline") as graph:
    v = validate_input(inputs={"data": PARENT["input"]})
    t = transform_data(inputs={"data": v["validated"]})
    s = save_result(inputs={"data": t["transformed"]}, outputs={"*": PARENT})

    START >> v >> t >> s >> END
```

### Factory Pattern cho reusable workflows

```python
def create_llm_chain(name: str, system_prompt: str):
    """Factory để tạo LLM chain."""
    from hush.providers import PromptNode, LLMNode

    with GraphNode(name=name) as chain:
        prompt = PromptNode(
            name="prompt",
            inputs={
                "prompt": {"system": system_prompt, "user": "{query}"},
                "query": PARENT["query"]
            }
        )
        llm = LLMNode(
            name="llm",
            resource_key="gpt-4",
            inputs={"messages": prompt["messages"]},
            outputs={"content": PARENT["response"]}
        )
        START >> prompt >> llm >> END

    return chain

# Use factory
summarizer = create_llm_chain(
    name="summarizer",
    system_prompt="Bạn là assistant chuyên tóm tắt văn bản."
)
translator = create_llm_chain(
    name="translator",
    system_prompt="Bạn là assistant chuyên dịch văn bản."
)
```

## Testing Workflows

### Unit test individual nodes

```python
import pytest
from hush.core import StateSchema

@pytest.mark.asyncio
async def test_node_logic():
    with GraphNode(name="test") as graph:
        node = CodeNode(
            name="double",
            code_fn=lambda x: {"result": x * 2},
            inputs={"x": PARENT["x"]},
            outputs={"*": PARENT}
        )
        START >> node >> END

    graph.build()
    schema = StateSchema(graph)
    state = schema.create_state(inputs={"x": 5})
    result = await graph.run(state)

    assert result["result"] == 10
```

### Integration test full workflow

```python
@pytest.mark.asyncio
async def test_full_workflow():
    from hush.core import Hush

    with GraphNode(name="pipeline") as graph:
        # ... define nodes ...
        START >> node1 >> node2 >> END

    engine = Hush(graph)
    result = await engine.run(inputs={"x": 10})

    assert result["output"] == expected_value
```

## Debugging Tips

### 1. Sử dụng state.show()

```python
result = await engine.run(inputs={"query": "Hello"})
state = result["$state"]
state.show()  # Print toàn bộ state structure
```

### 2. Kiểm tra execution order

```python
state = result["$state"]
for node_name, parent_name, context_id in state.execution_order:
    print(f"{node_name} (parent: {parent_name})")
```

### 3. Access intermediate values

```python
# Access node outputs directly
value = state["pipeline.step1", "result", None]
```

### 4. Sử dụng Tracer

```python
from hush.core.tracers import LocalTracer

tracer = LocalTracer(tags=["debug"])
result = await engine.run(inputs={...}, tracer=tracer)

# Traces lưu tại ~/.hush/traces.db
```

### 5. Check graph structure

```python
engine = Hush(graph)
engine.show()  # Print graph structure

# Check edges
for node_name, targets in graph.edges.items():
    print(f"{node_name} -> {targets}")
```

## Best Practices

1. **Đặt tên nodes rõ ràng**: Sử dụng tên mô tả chức năng (e.g., `validate_input`, `fetch_data`)

2. **Giữ nodes nhỏ và focused**: Mỗi node nên làm một việc cụ thể

3. **Sử dụng type hints**: Giúp debug và documentation

4. **Map outputs explicitly**: Tránh `outputs={"*": PARENT}` nếu không cần thiết

5. **Test từng node riêng**: Unit test trước khi integration test

6. **Sử dụng tracer trong development**: Giúp debug và hiểu flow

## Tiếp theo

- [Tích hợp LLM](llm-integration.md) - Sử dụng PromptNode và LLMNode
- [Thực thi song song](parallel-execution.md) - Tối ưu performance với parallel execution
- [Xử lý lỗi](error-handling.md) - Error handling và retry patterns
