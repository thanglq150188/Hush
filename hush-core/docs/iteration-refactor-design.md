# Thiết kế lại Iteration Nodes

## 1. Bối cảnh

### 1.1. Hệ thống hiện tại

Hush-core là workflow engine cho phép định nghĩa các node và kết nối chúng thành graph. Các iteration nodes (ForLoopNode, MapNode, WhileLoopNode, AsyncIterNode) cho phép lặp qua collections hoặc điều kiện.

**Cấu trúc node hierarchy:**
```
workflow (GraphNode)
  └── outer (ForLoopNode)
        └── outer.__inner__ (GraphNode)     ← Inner graph
              └── inner (ForLoopNode)
                    └── inner.__inner__ (GraphNode)
                          └── process (CodeNode)
```

**State storage:** Mỗi node lưu outputs vào `MemoryState` với key `(node_name, var_name, context_id)`.

**PARENT reference:** Syntax `PARENT["x"]` cho phép child node đọc biến từ parent node. Lúc build, `PARENT` được normalize thành `Ref(father_node, "x")`.

### 1.2. Vấn đề cần giải quyết

#### A. Inner Graph gây phức tạp

```
ForLoopNode (BaseIterationNode)
  └── _graph: GraphNode (inner graph)
        └── _nodes: {node1, node2, ...}
```

**Vấn đề:**
- Phải inject inputs vào cả iteration node VÀ inner graph:
  ```python
  # BaseIterationNode.inject_inputs()
  state[self.full_name, var_name, context_id] = value
  state[inner_graph_name, var_name, context_id] = value  # Duplicate!
  ```
- Schema phải tạo `pull_ref` từ inner_graph → iteration_node
- Với nested loops, state lookup có thể cần 4+ hops

#### B. Context ID dạng chuỗi (chain)

```python
# Hiện tại
task_id = f"for[{i}]" if not context_id else f"{context_id}.for[{i}]"

# Với 3 nested loops:
outer: context_id = "for[0]"
middle: context_id = "for[0].for[1]"
inner: context_id = "for[0].for[1].for[2]"
```

**Vấn đề:**
- Context ID dài, khó đọc
- Parse context để tìm parent phức tạp
- Không cần thiết vì hierarchy đã có trong `node.full_name`

#### C. PARENT Resolution sai context

```python
# inner_loop chạy với context_id = "for[0].for[1]"
# PARENT["x"] cần lấy từ outer_loop
# Nhưng outer_loop lưu "x" ở context "for[0]", không phải "for[0].for[1]"
# → Context ID mismatch → Bug!
```

#### D. Multi-hop state resolution

Khi `process` cần đọc `PARENT["x"]` từ `outer` trong nested loops:

```
state["inner.__inner__", "x_val", "for[0].for[1]"]
  → pull từ state["inner", "x_val", "for[0].for[1]"]
    → pull từ state["outer.__inner__", "x", "for[0].for[1]"]
      → pull từ state["outer", "x", "for[0].for[1]"]
        → KHÔNG TÌM THẤY! outer chỉ có "for[0]"
```

**4 hops** và cuối cùng vẫn fail do context mismatch.

---

## 2. Thiết kế mới

### 2.1. Iteration Nodes kế thừa từ GraphNode

**Cấu trúc mới:**
```
ForLoopNode (extends GraphNode)
  └── _nodes: {node1, node2, ...}  (trực tiếp sở hữu)
```

**Hierarchy mới:**
```
workflow (GraphNode)
  └── outer (ForLoopNode extends GraphNode)
        └── inner (ForLoopNode extends GraphNode)
              └── process (CodeNode)
```

**Lợi ích:**
- Không còn inner graph indirection
- Không cần duplicate injection
- Reuse logic của GraphNode (edges, ready_count, parallel execution)
- Schema đơn giản hơn, state lookup O(1)

### 2.2. Context Model mới

**Hai tham số cho tất cả `run()` methods:**

| Tham số | Mục đích | Mặc định |
|---------|----------|----------|
| `context_id` | Context của node này cho biến của chính nó | `None` |
| `parent_context` | Context của PARENT để resolve PARENT refs | `None` |

**Không có chain.** Mỗi node chỉ biết:
- Context của chính nó
- Context của parent trực tiếp

**Hierarchy nằm trong `full_name`:**
```python
state["workflow.outer", "x", "loop[0]"]              # outer's x
state["workflow.outer.inner", "y", "loop[1]"]        # inner's y
state["workflow.outer.inner.proc", "z", "loop[2]"]   # proc's z
```

### 2.3. Signature mới

**BaseNode:**
```python
async def run(
    self,
    state: 'MemoryState',
    context_id: Optional[str] = None,
    parent_context: Optional[str] = None
) -> Dict[str, Any]

def get_inputs(
    self,
    state: 'MemoryState',
    context_id: Optional[str],
    parent_context: Optional[str] = None
) -> Dict[str, Any]

def get_outputs(
    self,
    state: 'MemoryState',
    context_id: Optional[str],
    parent_context: Optional[str] = None
) -> Dict[str, Any]
```

### 2.4. State Resolution Logic

```python
def get_inputs(self, state, context_id, parent_context=None):
    result = {}

    for var_name, param in self.inputs.items():
        if isinstance(param.value, Ref):
            ref = param.value

            # Xác định context dựa trên ref target
            if ref.raw_node is self.father:
                # PARENT ref → dùng parent_context
                ctx = parent_context
            else:
                # Sibling hoặc khác → dùng context_id
                ctx = context_id

            raw = state[ref.node, ref.var, ctx]
            value = ref._fn(raw) if raw is not None else None
        else:
            value = param.value

        if value is not None:
            result[var_name] = value
        elif param.default is not None:
            result[var_name] = param.default

    return result
```

### 2.5. Class Hierarchy mới

```python
class ForLoopNode(GraphNode):
    """Sequential iteration - xử lý items tuần tự."""
    type: NodeType = "for"

class MapNode(GraphNode):
    """Parallel iteration - xử lý items đồng thời."""
    type: NodeType = "map"

class WhileLoopNode(GraphNode):
    """Condition-based iteration - lặp đến khi stop_condition."""
    type: NodeType = "while"

class AsyncIterNode(GraphNode):
    """Streaming iteration - xử lý async iterable với output có thứ tự."""
    type: NodeType = "stream"
```

---

## 3. Ví dụ Execution Flow

### Code:
```python
with GraphNode(name="workflow") as graph:
    with ForLoopNode(name="outer", inputs={"x": Each([10, 20])}) as outer:
        with ForLoopNode(name="inner", inputs={"y": Each([1, 2]), "x_val": PARENT["x"]}) as inner:
            process = CodeNode(
                name="proc",
                inputs={"x": PARENT["x_val"], "y": PARENT["y"]},
                code_fn=lambda x, y: {"result": x + y},
                outputs=PARENT
            )
            START >> process >> END
        START >> inner >> END
    START >> outer >> END
```

### Execution trace:

```
graph.run(context_id=None, parent_context=None)
│
└─► outer.run(context_id=None, parent_context=None)
    │
    │   # outer resolve Each([10, 20]) → items = [10, 20]
    │   # outer iteration 0: x=10
    │   # Lưu: state["workflow.outer", "x", "loop[0]"] = 10
    │
    ├─► outer._run_graph(context_id="loop[0]", parent_context="loop[0]")
    │   │
    │   └─► inner.run(context_id="loop[0]", parent_context="loop[0]")
    │       │
    │       │   # inner resolve PARENT["x"] dùng parent_context="loop[0]"
    │       │   # → state["workflow.outer", "x", "loop[0]"] = 10 ✓
    │       │
    │       │   # inner iteration 0: y=1, x_val=10
    │       │   # Lưu: state["workflow.outer.inner", "y", "loop[0]"] = 1
    │       │   # Lưu: state["workflow.outer.inner", "x_val", "loop[0]"] = 10
    │       │
    │       ├─► inner._run_graph(context_id="loop[0]", parent_context="loop[0]")
    │       │   │
    │       │   └─► process.run(context_id="loop[0]", parent_context="loop[0]")
    │       │           # PARENT["x_val"] → ref.raw_node is inner → dùng parent_context
    │       │           # → state["workflow.outer.inner", "x_val", "loop[0]"] = 10 ✓
    │       │           # PARENT["y"] → state["workflow.outer.inner", "y", "loop[0]"] = 1 ✓
    │       │           # result = 10 + 1 = 11
    │       │
    │       │   # inner iteration 1: y=2
    │       │   # Lưu: state["workflow.outer.inner", "y", "loop[1]"] = 2
    │       │
    │       └─► inner._run_graph(context_id="loop[1]", parent_context="loop[1]")
    │               # process result = 10 + 2 = 12
    │
    │   # outer iteration 1: x=20
    │   # Lưu: state["workflow.outer", "x", "loop[1]"] = 20
    │
    └─► outer._run_graph(context_id="loop[1]", parent_context="loop[1]")
            # inner resolve PARENT["x"] = 20 ✓
            # inner loops với x_val=20, y=1,2 → results: 21, 22
```

---

## 4. Edge Cases

### 4.1. Sibling node reference ✅

```python
with ForLoopNode(name="loop", inputs={"x": Each([1, 2])}) as loop:
    a = CodeNode(name="a", inputs={"x": PARENT["x"]}, outputs={"result": None})
    b = CodeNode(name="b", inputs={"a_result": a["result"]})  # Sibling ref
    START >> a >> b >> END
```

**Phân tích:**
- `a` chạy với `context_id="loop[0]"`, `parent_context="loop[0]"`
- `a` đọc `PARENT["x"]` → `ref.raw_node is self.father` (loop) → dùng `parent_context`
- `a` lưu `result` vào `state["workflow.loop.a", "result", "loop[0]"]`
- `b` đọc `a["result"]` → `ref.raw_node is a`, NOT `self.father` → dùng `context_id="loop[0]"`
- ✅ Đọc đúng

### 4.2. Access Grandparent ⚠️

```python
with ForLoopNode(name="outer", inputs={"x": Each([1, 2])}) as outer:
    with ForLoopNode(name="inner", inputs={"y": Each([10, 20])}) as inner:
        process = CodeNode(inputs={
            "y": PARENT["y"],      # OK - từ inner
            "x": PARENT["x"],      # ❌ PARENT = inner, không phải outer!
        })
```

**Không hỗ trợ.** Giải pháp - pass data xuống qua từng level:

```python
with ForLoopNode(name="outer", inputs={"x": Each([1, 2])}) as outer:
    with ForLoopNode(name="inner", inputs={
        "y": Each([10, 20]),
        "x_from_outer": PARENT["x"]  # Pass xuống
    }) as inner:
        process = CodeNode(inputs={
            "y": PARENT["y"],
            "x": PARENT["x_from_outer"]  # ✅ Access từ inner's PARENT
        })
```

### 4.3. Ref đến node ngoài iteration ⚠️

```python
config = CodeNode(name="config", outputs={"setting": "default"})

with ForLoopNode(name="loop", inputs={"x": Each([1, 2, 3])}) as loop:
    process = CodeNode(inputs={
        "x": PARENT["x"],
        "setting": config["setting"]  # ❌ Context mismatch!
    })
```

**Vấn đề:** `config` lưu ở `context_id=None` (main), `process` đọc với `context_id="loop[0]"`.

**Giải pháp - Broadcast qua PARENT:**

```python
with ForLoopNode(
    name="loop",
    inputs={"x": Each([1, 2, 3]), "setting": config["setting"]}  # Broadcast
) as loop:
    process = CodeNode(inputs={"x": PARENT["x"], "setting": PARENT["setting"]})  # ✅
```

### 4.4. WhileLoopNode carry forward ✅

```python
with WhileLoopNode(name="loop", inputs={"counter": 0}, stop_condition="counter >= 5") as loop:
    inc = CodeNode(inputs={"c": PARENT["counter"]}, ...)
    inc["new_c"] >> PARENT["counter"]
```

WhileLoopNode dùng local `step_inputs` dict để carry forward giữa các iterations, không dựa vào state multi-hop.

### 4.5. MapNode parallel execution ✅

```python
with MapNode(name="map", inputs={"x": Each([1, 2, 3, 4, 5])}) as map_node:
    process = CodeNode(inputs={"val": PARENT["x"]})
```

Tất cả iterations chạy song song với `context_id` riêng biệt (`loop[0]`, `loop[1]`, ...). Mỗi iteration có cùng `parent_context` (context của MapNode).

---

## 5. API Contract

### 5.1. Inputs cho Iteration Nodes

```python
# ✅ ĐÚNG: Tất cả inputs qua Each() hoặc broadcast
with ForLoopNode(inputs={
    "x": Each([1, 2, 3]),           # Iterate - mỗi iteration nhận 1 item
    "y": Each([10, 20, 30]),        # Iterate - zipped với x
    "config": config_node["value"]   # Broadcast - resolve 1 lần, dùng cho tất cả
}) as loop:
    process = CodeNode(inputs={
        "x": PARENT["x"],
        "config": PARENT["config"]
    })

# ❌ SAI: Direct ref đến node ngoài loop trong child node
with ForLoopNode(inputs={"x": Each([1, 2, 3])}) as loop:
    process = CodeNode(inputs={
        "x": PARENT["x"],
        "config": config_node["value"]  # Context mismatch!
    })
```

### 5.2. Outputs từ Iteration Nodes

```python
# Column-oriented output format
# Mỗi output variable là List các giá trị từ tất cả iterations

# Input: Each([1, 2, 3])
# Inner graph mỗi iteration output: {"result": x * 2, "status": "ok"}
# ForLoopNode output: {
#     "result": [2, 4, 6],
#     "status": ["ok", "ok", "ok"],
#     "iteration_metrics": {...}
# }
```

### 5.3. Nested Loops

```python
# ✅ ĐÚNG: Pass data xuống qua PARENT
with ForLoopNode(name="outer", inputs={"x": Each([1, 2])}) as outer:
    with ForLoopNode(name="inner", inputs={
        "y": Each([10, 20]),
        "outer_x": PARENT["x"]
    }) as inner:
        process = CodeNode(inputs={
            "y": PARENT["y"],
            "x": PARENT["outer_x"]
        })

# ❌ SAI: Cố gắng access grandparent
with ForLoopNode(name="outer", inputs={"x": Each([1, 2])}) as outer:
    with ForLoopNode(name="inner", inputs={"y": Each([10, 20])}) as inner:
        process = CodeNode(inputs={
            "y": PARENT["y"],
            "x": PARENT["x"]  # PARENT = inner, không có "x"!
        })
```

---

## 6. Quyết định thiết kế

### 6.1. Giữ PARENT thay vì direct node reference

**Đã xem xét:** Bỏ `PARENT`, cho phép `outer["x"]` thay vì `PARENT["x"]`.

**Lợi ích nếu bỏ PARENT:**
- Rõ ràng hơn - biết chính xác node nào
- Hỗ trợ grandparent access: `outer["x"]` từ bất kỳ đâu

**Vấn đề:**
- Cần `ancestor_contexts` map để track context của tất cả ancestors
- Phức tạp hóa implementation
- User phải tự quản lý node references

**Quyết định: Giữ PARENT** vì:
1. Syntax quen thuộc, đã hoạt động tốt
2. Check `ref.raw_node is self.father` đơn giản
3. Hiếm khi cần access grandparent
4. Có workaround (pass data xuống qua từng level)

### 6.2. Không hỗ trợ PARENT.PARENT

**Lý do:**
- Phức tạp hóa context tracking (cần track nhiều levels)
- Có workaround đơn giản (pass data xuống)
- Giữ thiết kế đơn giản và predictable

---

## 7. Implementation Details

### 7.1. ForLoopNode mới

```python
class ForLoopNode(GraphNode):
    """Sequential iteration - xử lý items tuần tự."""
    type: NodeType = "for"

    __slots__ = ['_each', '_broadcast_inputs', '_raw_inputs']

    def __init__(self, inputs: Optional[Dict[str, Any]] = None, **kwargs):
        self._raw_inputs = inputs or {}
        super().__init__(**kwargs)

        # Tách Each() sources khỏi broadcast
        self._each = {}
        self._broadcast_inputs = {}
        for var_name, value in self._raw_inputs.items():
            if isinstance(value, Each):
                self._each[var_name] = value.source
            else:
                self._broadcast_inputs[var_name] = value

    async def run(
        self,
        state: 'MemoryState',
        context_id: Optional[str] = None,
        parent_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Thực thi for loop tuần tự."""

        # Resolve each và broadcast values dùng parent_context
        # (vì có thể là PARENT refs cần resolve từ father's context)
        each_values = self._resolve_values(self._each, state, parent_context)
        broadcast_values = self._resolve_values(self._broadcast_inputs, state, parent_context)

        # Build iteration data - zip Each values, broadcast others
        iteration_data = self._build_iteration_data(each_values, broadcast_values)

        final_results = []
        for i, loop_data in enumerate(iteration_data):
            iter_context = f"loop[{i}]"  # Local context, KHÔNG chain!

            # Inject inputs vào state
            for var_name, value in loop_data.items():
                state[self.full_name, var_name, iter_context] = value

            # Chạy child nodes với iter_context là parent_context của chúng
            result = await self._run_graph(state, iter_context, iter_context)
            final_results.append(result)

        # Transpose results: [{a:1,b:2}, {a:3,b:4}] → {a:[1,3], b:[2,4]}
        _outputs = self._transpose_results(final_results)
        self.store_result(state, _outputs, context_id)
        return _outputs

    async def _run_graph(
        self,
        state: 'MemoryState',
        context_id: str,
        parent_context: str
    ) -> Dict[str, Any]:
        """Chạy child nodes - reuse logic từ GraphNode nhưng truyền parent_context."""

        active_tasks = {}
        ready_count = self.ready_count.copy()
        soft_satisfied = set()

        # Start entry nodes
        for entry in self.entries:
            task = asyncio.create_task(
                self._nodes[entry].run(state, context_id, parent_context)
            )
            active_tasks[entry] = task

        while active_tasks:
            done_tasks, _ = await asyncio.wait(
                active_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED
            )

            for task in done_tasks:
                node_name = task.get_name()
                active_tasks.pop(node_name)

                for next_node in self.nexts[node_name]:
                    edge = self._edges_lookup.get((node_name, next_node))
                    is_soft = edge and edge.soft

                    if is_soft:
                        if next_node in soft_satisfied:
                            continue
                        soft_satisfied.add(next_node)

                    ready_count[next_node] -= 1
                    if ready_count[next_node] == 0:
                        task = asyncio.create_task(
                            self._nodes[next_node].run(state, context_id, parent_context)
                        )
                        active_tasks[next_node] = task

        return self.get_outputs(state, context_id, parent_context)
```

### 7.2. GraphNode.run() thay đổi

```python
async def run(
    self,
    state: 'MemoryState',
    context_id: Optional[str] = None,
    parent_context: Optional[str] = None  # THÊM MỚI
) -> Dict[str, Any]:
    # ... existing code ...

    # Start entry nodes - truyền parent_context
    for entry in self.entries:
        task = asyncio.create_task(
            name=entry,
            coro=self._nodes[entry].run(state, context_id, parent_context)  # THÊM
        )
        active_tasks[entry] = task

    # ... trong loop ...
    task = asyncio.create_task(
        name=next_node,
        coro=self._nodes[next_node].run(state, context_id, parent_context)  # THÊM
    )
```

### 7.3. Schema cleanup

```python
# CŨ - schema.py _load_from()
if hasattr(node, '_graph') and node._graph:
    self._load_from(node._graph)
    inner_graph_name = node._graph.full_name
    for var_name in inputs.keys():
        self._register(inner_graph_name, var_name, Ref(node_name, var_name))

# MỚI - XÓA đoạn code trên
# Không cần link inner graph vì iteration nodes giờ là GraphNode trực tiếp
```

---

## 8. So sánh trước/sau

| Aspect | Thiết kế cũ | Thiết kế mới |
|--------|-------------|--------------|
| Cấu trúc | Inner graph indirection | Trực tiếp kế thừa GraphNode |
| Injection | Duplicate (node + inner graph) | Single injection |
| Context ID | Chain: `"for[0].for[1].for[2]"` | Local: `"loop[0]"` |
| PARENT resolution | Cùng context_id → sai | Dùng `parent_context` → đúng |
| Schema | Phức tạp với pull_ref linking | Đơn giản, không cần linking |
| State lookup | Multi-hop (4+ hops có thể) | Direct O(1) |
| Debug | Khó trace nested loops | Rõ ràng - biết chính xác context |

---

## 9. Checklist Implementation

### Cần thay đổi:

- [ ] `base.py`: Thêm `parent_context` vào `run()`, `get_inputs()`, `get_outputs()`
- [ ] `base.py`: Cập nhật `get_inputs()` check `ref.raw_node is self.father`
- [ ] `graph_node.py`: Truyền `parent_context` cho child nodes trong execution loop
- [ ] `iteration/for_loop_node.py`: Kế thừa `GraphNode`, bỏ `_graph`, implement `_run_graph()`
- [ ] `iteration/map_node.py`: Kế thừa `GraphNode`, bỏ `_graph`, parallel `_run_graph()`
- [ ] `iteration/while_loop_node.py`: Kế thừa `GraphNode`, bỏ `_graph`, giữ `step_inputs`
- [ ] `iteration/async_iter_node.py`: Kế thừa `GraphNode`, bỏ `_graph`, streaming
- [ ] `iteration/base.py`: Xóa `BaseIterationNode`, giữ `Each()` và `_calculate_iteration_metrics()`
- [ ] `states/schema.py`: Xóa inner graph linking trong `_load_from()`

### Cần giữ nguyên:

- `PARENT` syntax và normalize logic trong `_normalize_params()`
- `Each()` wrapper cho iteration variables
- `_resolve_values()` để resolve Refs
- `_build_iteration_data()` để zip Each values với broadcast
- `_transpose_results()` để convert row → column format
- `_calculate_iteration_metrics()` utility
- WhileLoopNode `step_inputs` carry forward logic
- Soft edge (`>> ~`) syntax

### Tests cần cập nhật:

- [ ] `tests/nodes/iteration/` - Cập nhật cho thiết kế mới
- [ ] Test nested loops với PARENT resolution
- [ ] Test sibling refs trong iteration
- [ ] Test grandparent workaround (pass data xuống)

---

## 10. Thứ tự Implementation

1. **Phase 1: Core changes**
   - Thêm `parent_context` vào `BaseNode.run()`, `get_inputs()`, `get_outputs()`
   - Cập nhật `get_inputs()` với logic check `ref.raw_node is self.father`
   - Cập nhật `GraphNode.run()` truyền `parent_context` cho child nodes

2. **Phase 2: ForLoopNode**
   - Refactor kế thừa `GraphNode`
   - Implement `_run_graph()` helper method
   - Xóa `_graph` attribute, dùng `_nodes` trực tiếp

3. **Phase 3: Schema cleanup**
   - Xóa inner graph linking trong `schema.py`
   - Test state resolution không còn multi-hop

4. **Phase 4: Other iteration nodes**
   - Refactor `MapNode` (parallel execution)
   - Refactor `WhileLoopNode` (giữ step_inputs carry forward)
   - Refactor `AsyncIterNode` (streaming)

5. **Phase 5: Cleanup và tests**
   - Xóa `BaseIterationNode` class
   - Di chuyển utilities vào file phù hợp
   - Cập nhật tất cả tests
