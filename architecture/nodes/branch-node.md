# BranchNode - Conditional Routing

## Overview

`BranchNode` đánh giá các điều kiện và định tuyến workflow đến các nodes khác nhau.

Location: `hush-core/hush/core/nodes/flow/branch_node.py`

## Class Definition

```python
class BranchNode(BaseNode):
    type: NodeType = "branch"

    __slots__ = [
        'given_candidates',  # List explicit candidates
        'conditions',        # List compiled conditions
        'default',          # Default target name
        'cases',            # Dict[condition_str, target_name]
        'ref_cases',        # List[(Ref, target_name)] cho fluent API
    ]
```

## Hai cách tạo BranchNode

### 1. String Conditions (Trực tiếp)

```python
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
```

### 2. Fluent Builder (Branch class)

```python
branch = (Branch("router")
    .if_(PARENT["score"] >= 90, excellent_node)
    .if_(PARENT["score"] >= 70, good_node)
    .if_(PARENT["score"] >= 50, average_node)
    .otherwise(fail_node))
```

## Condition Compilation

Điều kiện được precompile để tối ưu performance:

```python
def _compile_conditions(self) -> List[tuple]:
    """Precompile tất cả điều kiện."""
    compiled_conditions = []

    for condition, target in self.cases.items():
        try:
            compiled_code = compile(condition, f'<condition: {condition}>', 'eval')
            compiled_conditions.append((compiled_code, condition, target))
        except SyntaxError as e:
            raise ValueError(f"Cú pháp điều kiện không hợp lệ: {condition}")

    return compiled_conditions
```

## Input Parsing

BranchNode tự động parse inputs từ điều kiện:

```python
def _parse_cases(self, cases, ref_cases) -> tuple:
    # Luôn có anchor input để override routing
    inputs = {"anchor": Param(type=str, default=None)}

    # Parse biến từ string conditions
    for condition in cases:
        vars = extract_condition_variables(condition)
        for var_name in vars:
            inputs[var_name] = Param(required=True)

    # Parse từ ref_cases
    for ref, target in ref_cases:
        inputs[ref.var] = Param(required=True)

    outputs = {
        "target": Param(type=str, required=True),
        "matched": Param(type=str)
    }

    return inputs, outputs
```

## Execution

### Core Function

```python
def _create_core_function(self):
    def core(**inputs) -> Dict[str, str]:
        # 1. Check anchor override trước
        anchor = inputs.get('anchor')
        if anchor:
            return {"target": anchor, "matched": "anchor"}

        # 2. Evaluate conditions
        target, matched = self._evaluate_conditions(inputs)
        return {"target": target, "matched": matched}

    return core
```

### Condition Evaluation

```python
def _evaluate_conditions(self, inputs) -> tuple:
    safe_inputs = dict(inputs)

    # 1. String-based conditions (precompiled)
    for compiled_cond, condition_str, target in self.conditions:
        try:
            result = eval(compiled_cond, {"__builtins__": {}}, safe_inputs)
            if result:
                return target, condition_str
        except Exception:
            continue

    # 2. Ref-based conditions (supports .apply())
    for ref, target in self.ref_cases:
        try:
            value = safe_inputs.get(ref.var)
            result = ref.execute(value)  # Execute all ops
            if result:
                return target, f"ref:{ref.var}"
        except Exception:
            continue

    # 3. Default target
    if self.default:
        return self.default, "default"

    return None, None
```

## Anchor Override

Anchor cho phép override routing dynamically:

```python
# Trong workflow
branch = BranchNode(
    name="router",
    cases={"score >= 90": "excellent"},
    default="average",
    inputs={
        "score": PARENT["score"],
        "anchor": PARENT.get("force_route", None)  # Override nếu có
    }
)

# Runtime: nếu force_route = "excellent", sẽ route đến excellent
# bất kể score là bao nhiêu
```

## Graph Integration

### Soft Edges với Branch

Branch outputs thường dùng soft edges vì chỉ 1 nhánh được thực thi:

```python
with GraphNode(name="workflow") as g:
    branch = BranchNode(
        name="router",
        cases={"score >= 70": "pass"},
        default="fail"
    )

    pass_handler = CodeNode(name="pass", ...)
    fail_handler = CodeNode(name="fail", ...)
    merge = CodeNode(name="merge", ...)

    START >> branch
    branch >> ~pass_handler >> merge   # Soft edge
    branch >> ~fail_handler >> merge   # Soft edge
    merge >> END
```

### GraphNode Execution

GraphNode xử lý branch đặc biệt:

```python
# Trong GraphNode.run()
if node.type == "branch":
    branch_target = node.get_target(state, context_id)
    if branch_target != END.name:
        next_nodes = [branch_target]  # Chỉ 1 target
    else:
        next_nodes = []
else:
    next_nodes = self.nexts[node_name]  # Tất cả successors
```

## Fluent Builder

### Branch Class

```python
class Branch:
    """Fluent builder để tạo BranchNode."""

    def __init__(self, name: str, **kwargs):
        self._name = name
        self._cases: List[Tuple[Ref, str]] = []
        self._default: Optional[str] = None
        self._kwargs = kwargs

    def if_(self, condition: Ref, target) -> 'Branch':
        """Thêm case với Ref condition."""
        target_name = target.name if hasattr(target, 'name') else target
        self._cases.append((condition, target_name))
        return self

    def otherwise(self, target) -> 'BranchNode':
        """Set default và build node."""
        self._default = target.name if hasattr(target, 'name') else target
        return self._build()
```

### Ref Conditions

Ref hỗ trợ comparison operators:

```python
# Tạo Ref với comparison
PARENT["score"] >= 90    # Ref với op: ('>=', 90)
PARENT["status"] == "ok" # Ref với op: ('==', "ok")
PARENT["items"].apply(len) > 0  # Ref với apply() và comparison

# Fluent API sử dụng
(Branch("router")
    .if_(PARENT["score"] >= 90, "excellent")
    .if_(PARENT["items"].apply(len) > 0, "has_items")
    .otherwise("default"))
```

## Metadata

```python
def specific_metadata(self) -> Dict[str, Any]:
    return {
        "cases": self.cases,
        "default_target": self.default,
        "candidates": self.candidates,
        "num_conditions": len(self.conditions)
    }
```

## Candidates

```python
@property
def candidates(self) -> List[str]:
    """Danh sách tất cả possible targets."""
    if self.given_candidates:
        return self.given_candidates

    targets = list(self.cases.values())
    targets.extend(target for _, target in self.ref_cases)

    if self.default:
        targets.append(self.default)

    return targets
```
