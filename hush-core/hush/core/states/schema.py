"""StateSchema v2 - thiết kế đơn giản với index riêng cho mỗi biến."""

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type, TYPE_CHECKING

from hush.core.states.ref import Ref

if TYPE_CHECKING:
    from hush.core.states.state import MemoryState

__all__ = ["StateSchema"]


class StateSchema:
    """Định nghĩa cấu trúc state của workflow với độ phân giải O(1) dựa trên index.

    Mỗi cặp (node, var) có index duy nhất riêng. Các tham chiếu được lưu dưới dạng
    object Ref với idx trỏ đến source_idx.

    Cấu trúc dữ liệu:
        _indexer: {(node, var): idx} - ánh xạ biến sang index
        _values: [value, ...] - giá trị mặc định theo index
        _refs: [Ref, ...] - Ref với idx=source_idx (None nếu không phải ref)
    """

    __slots__ = ("name", "_indexer", "_values", "_refs")

    def __init__(self, node=None, name: str = None):
        """Khởi tạo schema.

        Args:
            node: Node tùy chọn để load các connection
            name: Tên workflow tùy chọn (suy ra từ node nếu không cung cấp)
        """
        self._indexer: Dict[Tuple[str, str], int] = {}
        self._values: List[Any] = []
        self._refs: List[Optional[Ref]] = []  # Ref với idx=source_idx, hoặc None

        self.name = name
        if node is not None:
            self.name = name or node.full_name
            self._load_from(node)
            self._build()

    # =========================================================================
    # Xây dựng Schema
    # =========================================================================

    def _load_from(self, node) -> None:
        """Thu thập đệ quy tất cả biến từ cây node."""
        node_name = node.full_name
        inputs = node.inputs or {}
        input_schema = getattr(node, 'input_schema', {}) or {}
        output_schema = getattr(node, 'output_schema', {}) or {}

        # Đăng ký các biến input
        for var_name, value in inputs.items():
            self._register(node_name, var_name, value)

        # Đăng ký default từ input_schema (nếu chưa được set)
        for var_name, param in input_schema.items():
            if var_name not in inputs:
                default = getattr(param, 'default', None)
                self._register(node_name, var_name, default)

        # Đăng ký các biến output
        for var_name, param in output_schema.items():
            if (node_name, var_name) not in self._indexer:
                default = getattr(param, 'default', None)
                self._register(node_name, var_name, default)

        # Đăng ký output connection - KHÔNG cho phép ops
        # outputs={"new_counter": PARENT["counter"]} nghĩa là:
        # node.new_counter -> PARENT.counter (output ref)
        # Khi đọc node.new_counter, đẩy giá trị sang PARENT.counter
        for var_name, ref in (node.outputs or {}).items():
            if isinstance(ref, Ref):
                if ref.has_ops:
                    raise ValueError(f"Output ref {node_name}.{var_name} -> {ref.node}.{ref.var} không được có operation")
                # node_name.var_name -> ref.node.ref.var (output ref)
                self._register(node_name, var_name, Ref(ref.node, ref.var, is_output=True))

        # Đăng ký các biến metadata
        for meta_var in ("start_time", "end_time", "error"):
            self._register(node_name, meta_var, None)

        # Load đệ quy các node con
        if hasattr(node, '_nodes') and node._nodes:
            for child in node._nodes.values():
                self._load_from(child)

        # Load đệ quy inner graph (cho iteration node)
        if hasattr(node, '_graph') and node._graph:
            self._load_from(node._graph)
            # Liên kết biến inner graph <- biến iteration node (cho PARENT access)
            inner_graph_name = node._graph.full_name
            # Input: inner_graph.var -> iteration_node.var (PARENT đọc từ iteration node)
            for var_name in inputs.keys():
                self._register(inner_graph_name, var_name, Ref(node_name, var_name))
            # Lưu ý: Output ref được xử lý bởi logic output connection ở trên

    def _register(self, node: str, var: str, value: Any) -> None:
        """Đăng ký một biến (có thể gọi nhiều lần, Ref luôn được ưu tiên)."""
        key = (node, var)
        if key in self._indexer:
            # Đã đăng ký - cập nhật nếu giá trị mới là Ref hoặc giá trị hiện tại là None
            idx = self._indexer[key]
            current = self._values[idx]
            if isinstance(value, Ref) or (current is None and value is not None):
                self._values[idx] = value
            return

        # Biến mới - gán index
        idx = len(self._values)
        self._indexer[key] = idx
        self._values.append(value)  # Có thể là Ref, được resolve trong _build()
        self._refs.append(None)

    def _build(self) -> None:
        """Resolve tất cả giá trị Ref sang source index và lưu vào _refs."""
        for key, idx in self._indexer.items():
            value = self._values[idx]
            if isinstance(value, Ref):
                source_key = (value.node, value.var)
                source_idx = self._indexer.get(source_key, -1)
                value.idx = source_idx  # Set source index trên Ref
                self._refs[idx] = value
                self._values[idx] = None  # Xóa Ref, giá trị lấy từ source

    # =========================================================================
    # Các Method Phân Giải Core (O(1))
    # =========================================================================

    def get_index(self, node: str, var: str) -> int:
        """Lấy storage index của một biến. Trả về -1 nếu không tìm thấy."""
        return self._indexer.get((node, var), -1)

    def get_default(self, idx: int) -> Any:
        """Lấy giá trị mặc định cho một index."""
        if 0 <= idx < len(self._values):
            return self._values[idx]
        return None

    def get_ref(self, idx: int) -> Optional[Ref]:
        """Lấy object Ref cho một index (None nếu không phải tham chiếu)."""
        if 0 <= idx < len(self._refs):
            return self._refs[idx]
        return None

    def get_source(self, idx: int) -> Tuple[int, Optional[Callable]]:
        """Lấy source index và function transform cho một tham chiếu.

        Returns:
            (source_idx, fn) - source_idx là -1 nếu không phải tham chiếu
        """
        ref = self._refs[idx] if 0 <= idx < len(self._refs) else None
        if ref:
            return ref.idx, ref._fn
        return -1, None

    def resolve(self, node: str, var: str) -> Tuple[str, str]:
        """Resolve một biến về vị trí nguồn (nếu nó là tham chiếu)."""
        idx = self._indexer.get((node, var), -1)
        if idx >= 0:
            ref = self._refs[idx]
            if ref and ref.idx >= 0:
                # Tìm key cho source_idx
                for key, i in self._indexer.items():
                    if i == ref.idx:
                        return key
        return node, var

    @property
    def num_indices(self) -> int:
        """Số lượng storage index."""
        return len(self._values)

    # =========================================================================
    # Các Method Xây Dựng Thủ Công
    # =========================================================================

    def set(self, node: str, var: str, value: Any) -> "StateSchema":
        """Set giá trị mặc định cho một biến."""
        key = (node, var)
        if key in self._indexer:
            idx = self._indexer[key]
            self._values[idx] = value
        else:
            self._register(node, var, value)
        return self

    def link(self, node: str, var: str, source_node: str, source_var: Optional[str] = None, fn: Optional[Callable] = None) -> "StateSchema":
        """Liên kết một biến với biến khác (với transform tùy chọn)."""
        key = (node, var)
        source_key = (source_node, source_var or var)

        # Đảm bảo cả hai tồn tại
        if key not in self._indexer:
            self._register(node, var, None)
        if source_key not in self._indexer:
            self._register(source_node, source_var or var, None)

        idx = self._indexer[key]
        source_idx = self._indexer[source_key]
        # Tạo Ref với idx = source_idx
        ref = Ref(source_node, source_var or var)
        ref.idx = source_idx
        if fn:
            object.__setattr__(ref, '_fn', fn)
        self._refs[idx] = ref
        return self

    def get(self, node: str, var: str) -> Any:
        """Lấy giá trị mặc định của một biến."""
        idx = self._indexer.get((node, var), -1)
        if idx >= 0:
            return self._values[idx]
        return None

    def is_ref(self, node: str, var: str) -> bool:
        """Kiểm tra một biến có phải là tham chiếu không."""
        idx = self._indexer.get((node, var), -1)
        if idx >= 0:
            return self._refs[idx] is not None
        return False

    # =========================================================================
    # Tạo State
    # =========================================================================

    def create_state(
        self,
        inputs: Dict[str, Any] = None,
        state_class: Type["MemoryState"] = None,
        **kwargs
    ) -> "MemoryState":
        """Tạo state mới từ schema này."""
        if state_class is None:
            from hush.core.states.state import MemoryState
            state_class = MemoryState
        return state_class(schema=self, inputs=inputs, **kwargs)

    # =========================================================================
    # Collection Interface
    # =========================================================================

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        """Duyệt qua tất cả cặp (node, var)."""
        return iter(self._indexer.keys())

    def __getitem__(self, key: Tuple[str, str]) -> int:
        """Lấy index của (node, var). Raise KeyError nếu không tìm thấy."""
        if key in self._indexer:
            return self._indexer[key]
        raise KeyError(f"{key} không tìm thấy trong schema: {self.name}")

    def __contains__(self, key: Tuple[str, str]) -> bool:
        """Kiểm tra (node, var) đã được đăng ký chưa."""
        return key in self._indexer

    def __len__(self) -> int:
        """Số lượng biến đã đăng ký."""
        return len(self._indexer)

    def __repr__(self) -> str:
        refs = sum(1 for ref in self._refs if ref is not None)
        return f"StateSchema(name='{self.name}', vars={len(self._indexer)}, refs={refs})"

    # =========================================================================
    # Debug
    # =========================================================================

    @staticmethod
    def _ops_to_str(ops: List[Tuple[str, Any]]) -> str:
        """Chuyển đổi danh sách ops thành string dễ đọc như x['key'].upper()"""
        result = "x"
        for op, args in ops:
            a = args[0] if args else None
            match op:
                case 'getitem': result += f"[{a!r}]"
                case 'getattr': result += f".{a}"
                case 'call':
                    ca, kw = args
                    args_str = ", ".join([repr(x) for x in ca] + [f"{k}={v!r}" for k, v in kw.items()])
                    result += f"({args_str})"
                case 'add': result = f"({result} + {a!r})"
                case 'radd': result = f"({a!r} + {result})"
                case 'sub': result = f"({result} - {a!r})"
                case 'rsub': result = f"({a!r} - {result})"
                case 'mul': result = f"({result} * {a!r})"
                case 'rmul': result = f"({a!r} * {result})"
                case 'truediv': result = f"({result} / {a!r})"
                case 'rtruediv': result = f"({a!r} / {result})"
                case 'floordiv': result = f"({result} // {a!r})"
                case 'rfloordiv': result = f"({a!r} // {result})"
                case 'mod': result = f"({result} % {a!r})"
                case 'rmod': result = f"({a!r} % {result})"
                case 'pow': result = f"({result} ** {a!r})"
                case 'rpow': result = f"({a!r} ** {result})"
                case 'neg': result = f"(-{result})"
                case 'apply':
                    func, _, _ = args
                    func_name = getattr(func, '__name__', repr(func))
                    result = f"{func_name}({result})"
                case _: result += f".{op}(...)"
        return result

    def show(self) -> None:
        """Hiển thị debug cấu trúc schema."""
        print(f"\n=== StateSchema: {self.name} ===")

        # Xây dựng reverse index cho hiển thị
        idx_to_key = {idx: key for key, idx in self._indexer.items()}

        for node, var in self:
            idx = self._indexer[(node, var)]
            ref = self._refs[idx]
            value = self._values[idx]

            if ref is not None:
                # Tham chiếu: hiển thị source với ops và cờ is_output
                source_key = idx_to_key.get(ref.idx, ("?", "?"))
                ops_str = ""
                if ref.has_ops:
                    ops_str = f" {self._ops_to_str(ref._ops)}"
                output_str = " (output)" if ref.is_output else ""
                print(f"{node}.{var} -> [{idx}] -> {source_key[0]}.{source_key[1]}[{ref.idx}]{ops_str}{output_str}")
            else:
                # Terminal: hiển thị giá trị
                print(f"{node}.{var} -> [{idx}] = {value}")

        print(f"Tổng: {len(self._values)} biến")


def main():
    """Test StateSchema v2 với các ví dụ GraphNode."""
    from hush.core.nodes.graph.graph_node import GraphNode, START, END, PARENT
    from hush.core.nodes.transform.code_node import CodeNode

    def test(name: str, condition: bool):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        if not condition:
            raise AssertionError(f"Test failed: {name}")

    # =========================================================================
    # Test 1: Simple linear graph (A -> B -> C)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 1: Simple linear graph")
    print("=" * 60)

    with GraphNode(name="linear_graph") as graph:
        node_a = CodeNode(
            name="node_a",
            code_fn=lambda x: {"result": x + 10},
            inputs={"x": PARENT["x"]}
        )

        node_b = CodeNode(
            name="node_b",
            code_fn=lambda x: {"result": x * 2},
            inputs={"x": node_a["result"]}
        )

        node_c = CodeNode(
            name="node_c",
            code_fn=lambda x: {"result": x - 5},
            inputs={"x": node_b["result"]},
            outputs={"result": PARENT["result"]}
        )

        START >> node_a >> node_b >> node_c >> END

    graph.build()
    schema = StateSchema(graph)
    schema.show()

    # Verify structure
    test("schema name", schema.name == "linear_graph")
    test("node_a.x is ref", schema.is_ref("linear_graph.node_a", "x"))
    test("node_b.x is ref", schema.is_ref("linear_graph.node_b", "x"))
    test("node_c.x is ref", schema.is_ref("linear_graph.node_c", "x"))
    # node_c.result is output ref pointing to linear_graph.result
    test("node_c.result is output ref", schema.is_ref("linear_graph.node_c", "result"))
    node_c_result_ref = schema._refs[schema.get_index("linear_graph.node_c", "result")]
    test("node_c.result refs linear_graph.result", node_c_result_ref.node == "linear_graph" and node_c_result_ref.var == "result")
    test("node_c.result is_output=True", node_c_result_ref.is_output == True)

    # Verify indices are unique
    all_indices = [schema.get_index(n, v) for n, v in schema]
    test("all indices unique", len(all_indices) == len(set(all_indices)))

    # =========================================================================
    # Test 2: Ref with operations (getitem, method calls)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 2: Ref with operations")
    print("=" * 60)

    with GraphNode(name="ref_ops_graph") as graph2:
        node_a = CodeNode(
            name="data_source",
            code_fn=lambda: {"data": {"items": [1, 2, 3, 4, 5], "name": "test"}},
            inputs={}
        )

        # Ref with chained getitem
        node_b = CodeNode(
            name="extract_items",
            code_fn=lambda items: {"count": len(items), "sum": sum(items)},
            inputs={"items": node_a["data"]["items"]}
        )

        # Ref with method call
        node_c = CodeNode(
            name="transform_name",
            code_fn=lambda name: {"upper_name": name},
            inputs={"name": node_a["data"]["name"].upper()}
        )

        node_d = CodeNode(
            name="merge_results",
            code_fn=lambda count, total, name: {"result": f"{name}: {count} items, sum={total}"},
            inputs={
                "count": node_b["count"],
                "total": node_b["sum"],
                "name": node_c["upper_name"]
            },
            outputs={"result": PARENT["result"]}
        )

        START >> node_a >> [node_b, node_c] >> node_d >> END

    graph2.build()
    schema2 = StateSchema(graph2)
    schema2.show()

    # Verify refs with ops
    items_idx = schema2.get_index("ref_ops_graph.extract_items", "items")
    name_idx = schema2.get_index("ref_ops_graph.transform_name", "name")
    test("items has ref", schema2._refs[items_idx] is not None)
    test("name has ref", schema2._refs[name_idx] is not None)

    # Test the fn actually works
    test_data = {"items": [1, 2, 3], "name": "hello"}
    test("items fn extracts items", schema2._refs[items_idx]._fn(test_data) == [1, 2, 3])
    test("name fn extracts and uppers", schema2._refs[name_idx]._fn(test_data) == "HELLO")

    # =========================================================================
    # Test 3: Ref with apply()
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 3: Ref with apply()")
    print("=" * 60)

    with GraphNode(name="ref_apply_graph") as graph3:
        node_a = CodeNode(
            name="list_source",
            code_fn=lambda: {"numbers": [5, 2, 8, 1, 9, 3]},
            inputs={}
        )

        node_b = CodeNode(
            name="get_length",
            code_fn=lambda length: {"length": length},
            inputs={"length": node_a["numbers"].apply(len)}
        )

        node_c = CodeNode(
            name="sort_numbers",
            code_fn=lambda sorted_nums: {"sorted": sorted_nums},
            inputs={"sorted_nums": node_a["numbers"].apply(sorted)}
        )

        node_d = CodeNode(
            name="sum_numbers",
            code_fn=lambda total: {"total": total},
            inputs={"total": node_a["numbers"].apply(sum)}
        )

        START >> node_a >> [node_b, node_c, node_d] >> END

    graph3.build()
    schema3 = StateSchema(graph3)
    schema3.show()

    # Test fns
    test_numbers = [5, 2, 8, 1, 9, 3]
    length_idx = schema3.get_index("ref_apply_graph.get_length", "length")
    sorted_idx = schema3.get_index("ref_apply_graph.sort_numbers", "sorted_nums")
    sum_idx = schema3.get_index("ref_apply_graph.sum_numbers", "total")

    test("length fn", schema3._refs[length_idx]._fn(test_numbers) == 6)
    test("sorted fn", schema3._refs[sorted_idx]._fn(test_numbers) == [1, 2, 3, 5, 8, 9])
    test("sum fn", schema3._refs[sum_idx]._fn(test_numbers) == 28)

    # =========================================================================
    # Test 4: Arithmetic operations
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 4: Arithmetic operations")
    print("=" * 60)

    with GraphNode(name="arithmetic_graph") as graph4:
        node_a = CodeNode(
            name="number_source",
            code_fn=lambda: {"value": 10},
            inputs={}
        )

        # (value + 5) * 2 - using separate input and output vars
        node_b = CodeNode(
            name="compute",
            code_fn=lambda x: {"result": x},  # just pass through
            inputs={"x": (node_a["value"] + 5) * 2},
            outputs={"result": PARENT["result"]}
        )

        START >> node_a >> node_b >> END

    graph4.build()
    schema4 = StateSchema(graph4)
    schema4.show()

    # Input ref has arithmetic operations
    x_idx = schema4.get_index("arithmetic_graph.compute", "x")
    test("arithmetic fn on input", schema4._refs[x_idx]._fn(10) == 30)  # (10 + 5) * 2 = 30
    # Output ref pushes to parent
    result_idx = schema4.get_index("arithmetic_graph.compute", "result")
    test("output ref exists", schema4._refs[result_idx] is not None)
    test("output ref is_output", schema4._refs[result_idx].is_output == True)

    # =========================================================================
    # Test 5: Manual building
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 5: Manual building")
    print("=" * 60)

    schema5 = StateSchema(name="manual_schema")
    schema5.set("node1", "x", 100)
    schema5.set("node1", "y", 200)
    schema5.link("node2", "x", "node1", "x")
    schema5.link("node2", "z", "node1", "y", fn=lambda v: v * 2)
    schema5.show()

    test("manual set", schema5.get("node1", "x") == 100)
    test("manual link", schema5.is_ref("node2", "x"))
    test("manual link with fn", schema5._refs[schema5.get_index("node2", "z")]._fn(200) == 400)

    # =========================================================================
    # Test 6: Simple nested graph
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 6: Simple nested graph")
    print("=" * 60)

    with GraphNode(name="outer") as outer:
        # Inner graph that doubles input
        with GraphNode(
            name="inner",
            inputs={"x": PARENT["x"]},
            outputs={"result": PARENT["inner_result"]}
        ) as inner:
            double = CodeNode(
                name="double",
                code_fn=lambda x: {"result": x * 2},
                inputs={"x": PARENT["x"]},
                outputs={"result": PARENT["result"]}
            )
            START >> double >> END

        # Outer node uses inner result
        final = CodeNode(
            name="final",
            code_fn=lambda x: {"result": x + 100},
            inputs={"x": inner["result"]},
            outputs={"result": PARENT["result"]}
        )

        START >> inner >> final >> END

    outer.build()
    schema6 = StateSchema(outer)
    schema6.show()

    # Verify structure
    test("outer.x exists", ("outer", "x") in schema6)
    test("outer.inner.x is ref to outer.x", schema6.is_ref("outer.inner", "x"))
    # With output refs: inner.result -> outer.inner_result (output), not outer.inner_result -> inner.result
    test("outer.inner.result is output ref", schema6.is_ref("outer.inner", "result"))
    inner_result_ref = schema6._refs[schema6.get_index("outer.inner", "result")]
    test("outer.inner.result refs outer.inner_result", inner_result_ref.node == "outer" and inner_result_ref.var == "inner_result")
    test("outer.inner.result is_output", inner_result_ref.is_output == True)
    # outer.result receives from final.result (output ref)
    test("outer.final.result is output ref", schema6.is_ref("outer.final", "result"))
    final_result_ref = schema6._refs[schema6.get_index("outer.final", "result")]
    test("outer.final.result refs outer.result", final_result_ref.node == "outer" and final_result_ref.var == "result")
    test("outer.final.result is_output", final_result_ref.is_output == True)

    # =========================================================================
    # Test 7: Nested graph with node feeding into nested graph
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 7: Nested graph with preceding node")
    print("=" * 60)

    with GraphNode(name="workflow") as workflow:
        # First node generates data
        source = CodeNode(
            name="source",
            code_fn=lambda: {"value": 5, "multiplier": 3},
            inputs={}
        )

        # Nested graph receives from source node
        with GraphNode(
            name="processor",
            inputs={"x": source["value"]},
            outputs={"result": PARENT["processed"]}
        ) as processor:
            compute = CodeNode(
                name="compute",
                code_fn=lambda x: {"result": x * 10},
                inputs={"x": PARENT["x"]},
                outputs={"result": PARENT["result"]}
            )
            START >> compute >> END

        # Final node uses both source and processor results
        merge = CodeNode(
            name="merge",
            code_fn=lambda processed, mult: {"result": processed * mult},
            inputs={
                "processed": processor["result"],
                "mult": source["multiplier"]
            },
            outputs={"result": PARENT["result"]}
        )

        START >> source >> processor >> merge >> END

    workflow.build()
    schema7 = StateSchema(workflow)
    schema7.show()

    # Verify refs
    test("processor.x refs source.value", schema7.is_ref("workflow.processor", "x"))
    test("merge.processed refs processor.result", schema7.is_ref("workflow.merge", "processed"))
    test("merge.mult refs source.multiplier", schema7.is_ref("workflow.merge", "mult"))

    # =========================================================================
    # Test 8: Deeply nested graphs (3 levels)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 8: Deeply nested graphs (3 levels)")
    print("=" * 60)

    with GraphNode(name="level1") as level1:
        with GraphNode(
            name="level2",
            inputs={"x": PARENT["x"]},
            outputs={"result": PARENT["result"]}
        ) as level2:
            with GraphNode(
                name="level3",
                inputs={"x": PARENT["x"]},
                outputs={"result": PARENT["result"]}
            ) as level3:
                core = CodeNode(
                    name="core",
                    code_fn=lambda x: {"result": x * 3},
                    inputs={"x": PARENT["x"]},
                    outputs={"result": PARENT["result"]}
                )
                START >> core >> END
            START >> level3 >> END
        START >> level2 >> END

    level1.build()
    schema8 = StateSchema(level1)
    schema8.show()

    # Verify deep nesting - inputs chain down
    test("level1.level2.x refs level1.x", schema8.is_ref("level1.level2", "x"))
    test("level1.level2.level3.x refs level1.level2.x", schema8.is_ref("level1.level2.level3", "x"))
    test("level1.level2.level3.core.x refs level1.level2.level3.x", schema8.is_ref("level1.level2.level3.core", "x"))
    # With output refs: outputs chain up via output refs
    test("level1.level2.result is output ref", schema8.is_ref("level1.level2", "result"))
    level2_ref = schema8._refs[schema8.get_index("level1.level2", "result")]
    test("level1.level2.result refs level1.result", level2_ref.node == "level1" and level2_ref.var == "result")
    test("level1.level2.result is_output", level2_ref.is_output == True)

    # =========================================================================
    # Test 9: Nested graph with Ref operations
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 9: Nested graph with Ref operations")
    print("=" * 60)

    with GraphNode(name="ops_workflow") as ops_workflow:
        data = CodeNode(
            name="data",
            code_fn=lambda: {"info": {"items": [1, 2, 3], "name": "test"}},
            inputs={}
        )

        # Nested graph receives transformed data
        with GraphNode(
            name="processor",
            inputs={
                "items": data["info"]["items"],  # getitem chain
                "name": data["info"]["name"].upper()  # getitem + method call
            },
            outputs={"result": PARENT["result"]}
        ) as processor:
            process = CodeNode(
                name="process",
                code_fn=lambda items, name: {"result": f"{name}: {len(items)} items"},
                inputs={"items": PARENT["items"], "name": PARENT["name"]},
                outputs={"result": PARENT["result"]}
            )
            START >> process >> END

        START >> data >> processor >> END

    ops_workflow.build()
    schema9 = StateSchema(ops_workflow)
    schema9.show()

    # Verify ops on nested graph inputs
    items_ref = schema9._refs[schema9.get_index("ops_workflow.processor", "items")]
    name_ref = schema9._refs[schema9.get_index("ops_workflow.processor", "name")]
    test("processor.items has ops", items_ref is not None and items_ref.has_ops)
    test("processor.name has ops", name_ref is not None and name_ref.has_ops)

    # Test the fns
    test_info = {"items": [1, 2], "name": "hello"}
    test("items fn works", items_ref._fn(test_info) == [1, 2])
    test("name fn works", name_ref._fn(test_info) == "HELLO")

    # =========================================================================
    # Test 10: Multiple nested graphs at same level
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 10: Multiple nested graphs (parallel)")
    print("=" * 60)

    with GraphNode(name="parallel") as parallel:
        source = CodeNode(
            name="source",
            code_fn=lambda: {"value": 10},
            inputs={}
        )

        # Two nested graphs in parallel
        with GraphNode(
            name="branch_a",
            inputs={"x": source["value"]},
            outputs={"result": PARENT["result_a"]}
        ) as branch_a:
            add = CodeNode(
                name="add",
                code_fn=lambda x: {"result": x + 5},
                inputs={"x": PARENT["x"]},
                outputs={"result": PARENT["result"]}
            )
            START >> add >> END

        with GraphNode(
            name="branch_b",
            inputs={"x": source["value"]},
            outputs={"result": PARENT["result_b"]}
        ) as branch_b:
            mul = CodeNode(
                name="mul",
                code_fn=lambda x: {"result": x * 2},
                inputs={"x": PARENT["x"]},
                outputs={"result": PARENT["result"]}
            )
            START >> mul >> END

        # Merge results
        merge = CodeNode(
            name="merge",
            code_fn=lambda a, b: {"result": a + b},
            inputs={"a": branch_a["result"], "b": branch_b["result"]},
            outputs={"result": PARENT["result"]}
        )

        START >> source >> [branch_a, branch_b] >> merge >> END

    parallel.build()
    schema10 = StateSchema(parallel)
    schema10.show()

    # Verify parallel branches
    test("branch_a.x refs source.value", schema10.is_ref("parallel.branch_a", "x"))
    test("branch_b.x refs source.value", schema10.is_ref("parallel.branch_b", "x"))
    test("merge.a refs branch_a.result", schema10.is_ref("parallel.merge", "a"))
    test("merge.b refs branch_b.result", schema10.is_ref("parallel.merge", "b"))

    # =========================================================================
    # Test 11: Iteration node (WhileLoopNode) with output refs
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 11: Iteration node with output refs")
    print("=" * 60)

    from hush.core.nodes.iteration.while_loop_node import WhileLoopNode

    with WhileLoopNode(
        name="counter_loop",
        inputs={"counter": 0},
        stop_condition="counter >= 5",
        max_iterations=10
    ) as loop:
        inc = CodeNode(
            name="increment",
            code_fn=lambda counter: {"new_counter": counter + 1},
            inputs={"counter": PARENT["counter"]},
            outputs={"new_counter": PARENT["counter"]}
        )
        START >> inc >> END

    loop.build()
    schema11 = StateSchema(loop)
    schema11.show()

    # Verify input ref: inner_graph.counter -> loop.counter (for PARENT access)
    inner_counter_idx = schema11.get_index("counter_loop.__inner__", "counter")
    inner_counter_ref = schema11._refs[inner_counter_idx]
    test("inner_graph.counter is ref", inner_counter_ref is not None)
    test("inner_graph.counter refs loop.counter", inner_counter_ref.node == "counter_loop" and inner_counter_ref.var == "counter")
    test("inner_graph.counter is input ref", inner_counter_ref.is_output == False)

    # Verify output ref: increment.new_counter -> inner_graph.counter (output)
    inc_new_counter_idx = schema11.get_index("counter_loop.__inner__.increment", "new_counter")
    inc_new_counter_ref = schema11._refs[inc_new_counter_idx]
    test("increment.new_counter is ref", inc_new_counter_ref is not None)
    test("increment.new_counter refs inner_graph.counter", inc_new_counter_ref.node == "counter_loop.__inner__" and inc_new_counter_ref.var == "counter")
    test("increment.new_counter is output ref", inc_new_counter_ref.is_output == True)

    # Verify loop.counter has no ref (it stores the value directly)
    loop_counter_idx = schema11.get_index("counter_loop", "counter")
    loop_counter_ref = schema11._refs[loop_counter_idx]
    test("loop.counter has no ref", loop_counter_ref is None)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
