"""Workflow state với Cell-based storage và độ phân giải O(1) dựa trên index."""

from typing import Any, Dict, List, Optional, Tuple
import uuid

from hush.core.states.schema import StateSchema
from hush.core.states.cell import Cell, DEFAULT_CONTEXT

__all__ = ["MemoryState"]

_uuid4 = uuid.uuid4


class MemoryState:
    """Workflow state trong bộ nhớ với Cell-based storage và truy cập O(1) theo index.

    Mỗi biến có Cell riêng xử lý nhiều context (các iteration của loop).
    Tham chiếu được resolve qua schema._refs với tự động áp dụng fn.

    Luồng dữ liệu:
        __setitem__: idx = schema[node, var], cells[idx][ctx] = value
        __getitem__: idx = schema[node, var], nếu ref: theo ref.idx, áp dụng ref._fn
    """

    __slots__ = ("schema", "_cells", "_execution_order", "_user_id", "_session_id", "_request_id")

    def __init__(
        self,
        schema: StateSchema,
        inputs: Dict[str, Any] = None,
        user_id: str = None,
        session_id: str = None,
        request_id: str = None,
    ) -> None:
        """Khởi tạo MemoryState.

        Args:
            schema: StateSchema định nghĩa cấu trúc state
            inputs: Giá trị input ban đầu cho workflow
            user_id: ID người dùng (tự động tạo nếu không cung cấp)
            session_id: ID phiên (tự động tạo nếu không cung cấp)
            request_id: ID yêu cầu (tự động tạo nếu không cung cấp)
        """
        self.schema = schema
        self._cells: List[Cell] = [Cell(v) for v in schema._values]
        self._execution_order: List[Dict[str, str]] = []
        self._user_id = user_id or str(_uuid4())
        self._session_id = session_id or str(_uuid4())
        self._request_id = request_id or str(_uuid4())

        # Áp dụng input ban đầu
        if inputs:
            for var, value in inputs.items():
                idx = schema.get_index(schema.name, var)
                if idx >= 0:
                    self._cells[idx][None] = value

    # =========================================================================
    # Truy Cập Core
    # =========================================================================

    def __setitem__(self, key: Tuple[str, str, Optional[str]], value: Any) -> None:
        """Set giá trị: state[node, var, ctx] = value

        Nếu cell này có output ref, ngay lập tức đẩy giá trị đến target.
        Điều này cho phép output ref lan truyền giá trị khi ghi, không chỉ khi đọc.
        """
        node, var, ctx = key
        idx = self.schema.get_index(node, var)
        if idx < 0:
            raise KeyError(f"({node}, {var}) không có trong schema")
        self._cells[idx][ctx] = value

        # Nếu cell này có output ref, đẩy giá trị đến target
        ref = self.schema._refs[idx]
        if ref and ref.is_output and ref.idx >= 0:
            result = ref._fn(value)
            self._set_by_index(ref.idx, result, ctx)

    def __getitem__(self, key: Tuple[str, str, Optional[str]]) -> Any:
        """Lấy giá trị: state[node, var, ctx] - resolve ref và cache kết quả.

        Hai loại ref:
        - Input ref (is_output=False): Theo ref đến source, áp dụng fn, cache và trả về
        - Output ref (is_output=True): Lấy giá trị cell hiện tại, đẩy đến target, trả về giá trị

        Xử lý ref chain bằng cách resolve đệ quy.
        """
        node, var, ctx = key
        idx = self.schema.get_index(node, var)
        if idx < 0:
            return None
        return self._get_by_index(idx, ctx)

    def _get_by_index(self, idx: int, ctx: Optional[str]) -> Any:
        """Internal: lấy giá trị theo index, theo ref chain nếu cần."""
        if idx < 0 or idx >= len(self._cells):
            return None

        cell = self._cells[idx]
        ref = self.schema._refs[idx]

        # Chuẩn hóa ctx để kiểm tra tồn tại (Cell chuyển None thành DEFAULT_CONTEXT)
        ctx_key = ctx if ctx is not None else DEFAULT_CONTEXT

        # Kiểm tra đây có phải output ref không
        if ref and ref.is_output:
            # Output ref: lấy giá trị cell hiện tại và đẩy đến target
            if ctx_key in cell.contexts:
                value = cell[ctx]
                # Áp dụng fn và đẩy đến target cell
                result = ref._fn(value)
                if ref.idx >= 0:
                    self._cells[ref.idx][ctx] = result
                return value
            return None

        # Kiểm tra giá trị đã tồn tại trong cell này chưa
        if ctx_key in cell.contexts:
            return cell[ctx]

        # Kiểm tra đây có phải input ref không
        if ref and ref.idx >= 0:
            # Input ref: đệ quy lấy giá trị source
            source_value = self._get_by_index(ref.idx, ctx)
            if source_value is not None:
                result = ref._fn(source_value)
                cell[ctx] = result
                return result
            return None

        return cell[ctx]

    def get(self, node: str, var: str, ctx: Optional[str] = None) -> Any:
        """Lấy giá trị với tham số explicit."""
        return self[node, var, ctx]

    def get_cell(self, node: str, var: str) -> Cell:
        """Lấy object Cell cho một biến."""
        idx = self.schema.get_index(node, var)
        if idx < 0:
            raise KeyError(f"({node}, {var}) không có trong schema")
        return self._cells[idx]

    # =========================================================================
    # Truy Cập theo Index (O(1))
    # =========================================================================

    def get_by_index(self, idx: int, ctx: Optional[str] = None) -> Any:
        """Truy cập cell trực tiếp theo index."""
        if 0 <= idx < len(self._cells):
            return self._cells[idx][ctx]
        raise IndexError(f"Index {idx} ngoài phạm vi")

    def set_by_index(self, idx: int, value: Any, ctx: Optional[str] = None) -> None:
        """Gán giá trị cell trực tiếp theo index."""
        if 0 <= idx < len(self._cells):
            self._cells[idx][ctx] = value
        else:
            raise IndexError(f"Index {idx} ngoài phạm vi")

    def _set_by_index(self, idx: int, value: Any, ctx: Optional[str]) -> None:
        """Internal: set giá trị theo index, theo output ref chain nếu cần."""
        if idx < 0 or idx >= len(self._cells):
            return
        self._cells[idx][ctx] = value

        # Nếu cell này cũng có output ref, tiếp tục lan truyền
        ref = self.schema._refs[idx]
        if ref and ref.is_output and ref.idx >= 0:
            result = ref._fn(value)
            self._set_by_index(ref.idx, result, ctx)

    # =========================================================================
    # Theo Dõi Thực Thi
    # =========================================================================

    def record_execution(self, node_name: str, parent: str, context_id: str) -> None:
        """Ghi lại thực thi node cho observability."""
        self._execution_order.append({
            "node": node_name,
            "parent": parent,
            "context_id": context_id
        })

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def name(self) -> str:
        """Tên của workflow."""
        return self.schema.name

    @property
    def execution_order(self) -> List[Dict[str, str]]:
        """Danh sách thứ tự thực thi các node."""
        return self._execution_order.copy()

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata của state bao gồm user_id, session_id, request_id."""
        return {
            "user_id": self._user_id,
            "session_id": self._session_id,
            "request_id": self._request_id,
        }

    @property
    def user_id(self) -> str:
        """ID người dùng."""
        return self._user_id

    @property
    def session_id(self) -> str:
        """ID phiên."""
        return self._session_id

    @property
    def request_id(self) -> str:
        """ID yêu cầu."""
        return self._request_id

    # =========================================================================
    # Collection Interface
    # =========================================================================

    def __contains__(self, key: Tuple[str, str]) -> bool:
        """Kiểm tra (node, var) có tồn tại trong schema không."""
        return key in self.schema

    def __len__(self) -> int:
        """Số lượng cell."""
        return len(self._cells)

    def __iter__(self):
        """Duyệt qua các cặp (node, var)."""
        return iter(self.schema)

    # =========================================================================
    # Context Manager và Tiện ích
    # =========================================================================

    def __enter__(self) -> "MemoryState":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', cells={len(self._cells)})"

    def __hash__(self) -> int:
        return hash(self._request_id)

    def __eq__(self, other) -> bool:
        if not isinstance(other, MemoryState):
            return False
        return self._request_id == other._request_id

    def show(self) -> None:
        """Hiển thị debug các giá trị state hiện tại."""
        print(f"\n=== {self.__class__.__name__}: {self.name} ===")

        for node, var in self.schema:
            idx = self.schema.get_index(node, var)
            cell = self._cells[idx]
            ref = self.schema._refs[idx]

            if not cell.contexts:
                # Chưa có giá trị
                if ref:
                    print(f"{node}.{var} -> ref[{ref.idx}] (chưa có giá trị)")
                else:
                    print(f"{node}.{var} -> {cell.default_value}")
            elif len(cell.contexts) == 1:
                # Một context
                ctx = cell.versions[0]
                value = cell.contexts[ctx]
                value_str = repr(value)[:50] + "..." if len(repr(value)) > 50 else repr(value)
                print(f"{node}.{var} [{ctx}] = {value_str}")
            else:
                # Nhiều context
                print(f"{node}.{var}:")
                for ctx in cell.versions:
                    value = cell.contexts[ctx]
                    value_str = repr(value)[:50] + "..." if len(repr(value)) > 50 else repr(value)
                    print(f"  [{ctx}] = {value_str}")


def main():
    """Test MemoryState với các ví dụ GraphNode - inject giá trị và verify ref."""
    from hush.core.nodes.graph.graph_node import GraphNode, START, END, PARENT
    from hush.core.nodes.transform.code_node import CodeNode

    def test(name: str, condition: bool):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        if not condition:
            raise AssertionError(f"Test failed: {name}")

    # =========================================================================
    # Test 1: Simple linear graph - inject and follow refs
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 1: Simple linear graph with value injection")
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
    state = MemoryState(schema, inputs={"x": 5})

    # Verify input was set
    test("input x = 5", state["linear_graph", "x", None] == 5)

    # Simulate node_a execution: reads x, writes result
    x_val = state["linear_graph.node_a", "x", None]  # Should resolve to linear_graph.x
    test("node_a.x refs linear_graph.x", x_val == 5)

    state["linear_graph.node_a", "result", None] = x_val + 10  # 15
    test("node_a.result = 15", state["linear_graph.node_a", "result", None] == 15)

    # Simulate node_b execution
    x_val = state["linear_graph.node_b", "x", None]  # Should resolve to node_a.result
    test("node_b.x refs node_a.result", x_val == 15)

    state["linear_graph.node_b", "result", None] = x_val * 2  # 30
    test("node_b.result = 30", state["linear_graph.node_b", "result", None] == 30)

    # Simulate node_c execution
    x_val = state["linear_graph.node_c", "x", None]  # Should resolve to node_b.result
    test("node_c.x refs node_b.result", x_val == 30)

    state["linear_graph.node_c", "result", None] = x_val - 5  # 25
    test("node_c.result = 25", state["linear_graph.node_c", "result", None] == 25)

    # Final result should ref node_c.result
    test("linear_graph.result refs node_c.result", state["linear_graph", "result", None] == 25)

    # =========================================================================
    # Test 2: Ref with operations - getitem, method calls
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

        node_b = CodeNode(
            name="extract_items",
            code_fn=lambda items: {"count": len(items), "sum": sum(items)},
            inputs={"items": node_a["data"]["items"]}
        )

        node_c = CodeNode(
            name="transform_name",
            code_fn=lambda name: {"upper_name": name},
            inputs={"name": node_a["data"]["name"].upper()}
        )

        START >> node_a >> [node_b, node_c] >> END

    graph2.build()
    schema2 = StateSchema(graph2)
    state2 = MemoryState(schema2)

    # Inject data_source output
    state2["ref_ops_graph.data_source", "data", None] = {"items": [10, 20, 30], "name": "hello"}

    # Read with ops applied
    items = state2["ref_ops_graph.extract_items", "items", None]
    test("items extracts via x['items']", items == [10, 20, 30])

    name = state2["ref_ops_graph.transform_name", "name", None]
    test("name extracts and uppers via x['name'].upper()", name == "HELLO")

    # =========================================================================
    # Test 3: Ref with apply() - len, sorted, sum
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
    state3 = MemoryState(schema3)

    # Inject list_source output
    state3["ref_apply_graph.list_source", "numbers", None] = [5, 2, 8, 1, 9, 3]

    # Read with apply() fns
    test("length via apply(len)", state3["ref_apply_graph.get_length", "length", None] == 6)
    test("sorted via apply(sorted)", state3["ref_apply_graph.sort_numbers", "sorted_nums", None] == [1, 2, 3, 5, 8, 9])
    test("sum via apply(sum)", state3["ref_apply_graph.sum_numbers", "total", None] == 28)

    # =========================================================================
    # Test 4: Multiple contexts (loop simulation)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 4: Multiple contexts (loop simulation)")
    print("=" * 60)

    with GraphNode(name="loop_graph") as graph4:
        node_a = CodeNode(
            name="accumulator",
            code_fn=lambda x: {"result": x},
            inputs={"x": PARENT["x"]}
        )
        START >> node_a >> END

    graph4.build()
    schema4 = StateSchema(graph4)
    state4 = MemoryState(schema4, inputs={"x": 0})

    # Simulate loop iterations with different contexts
    state4["loop_graph", "x", "iter_0"] = 10
    state4["loop_graph", "x", "iter_1"] = 20
    state4["loop_graph", "x", "iter_2"] = 30

    test("x[iter_0] = 10", state4["loop_graph", "x", "iter_0"] == 10)
    test("x[iter_1] = 20", state4["loop_graph", "x", "iter_1"] == 20)
    test("x[iter_2] = 30", state4["loop_graph", "x", "iter_2"] == 30)

    # Refs should work per context
    test("node_a.x[iter_0] refs loop_graph.x[iter_0]", state4["loop_graph.accumulator", "x", "iter_0"] == 10)
    test("node_a.x[iter_1] refs loop_graph.x[iter_1]", state4["loop_graph.accumulator", "x", "iter_1"] == 20)
    test("node_a.x[iter_2] refs loop_graph.x[iter_2]", state4["loop_graph.accumulator", "x", "iter_2"] == 30)

    # Check cell versions
    cell = state4.get_cell("loop_graph", "x")
    test("cell has 4 versions (main + 3 iters)", len(cell.versions) == 4)

    # =========================================================================
    # Test 5: Nested graph with value flow
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 5: Nested graph with value flow")
    print("=" * 60)

    with GraphNode(name="outer") as outer:
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

        final = CodeNode(
            name="final",
            code_fn=lambda x: {"result": x + 100},
            inputs={"x": inner["result"]},
            outputs={"result": PARENT["result"]}
        )

        START >> inner >> final >> END

    outer.build()
    schema5 = StateSchema(outer)
    state5 = MemoryState(schema5, inputs={"x": 7})

    # Verify first-level ref: outer.inner.x -> outer.x
    test("outer.x = 7", state5["outer", "x", None] == 7)
    test("outer.inner.x refs outer.x", state5["outer.inner", "x", None] == 7)

    # Simulate execution flow (executor handles ref chain):
    # 1. Set inner.x for double to read (executor resolves ref chain)
    state5["outer.inner", "x", None] = 7
    test("outer.inner.double.x refs outer.inner.x", state5["outer.inner.double", "x", None] == 7)

    # 2. double executes
    state5["outer.inner.double", "result", None] = 7 * 2  # 14
    test("double.result = 14", state5["outer.inner.double", "result", None] == 14)

    # 3. inner.result refs double.result (one-level ref)
    test("inner.result refs double.result", state5["outer.inner", "result", None] == 14)

    # 4. Copy inner.result to its cell so outer-level refs can see it
    state5["outer.inner", "result", None] = 14

    # 5. outer.inner_result refs inner.result
    test("outer.inner_result refs inner.result", state5["outer", "inner_result", None] == 14)

    # 6. final.x refs inner.result
    test("final.x refs inner.result", state5["outer.final", "x", None] == 14)

    # 7. final executes
    state5["outer.final", "result", None] = 14 + 100
    test("final.result = 114", state5["outer.final", "result", None] == 114)

    # 8. outer.result refs final.result
    test("outer.result refs final.result", state5["outer", "result", None] == 114)

    # =========================================================================
    # Test 6: Index-based access
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test 6: Index-based access")
    print("=" * 60)

    schema6 = StateSchema(name="index_test")
    schema6.set("node1", "x", 100)
    schema6.set("node1", "y", 200)
    state6 = MemoryState(schema6)

    idx_x = schema6.get_index("node1", "x")
    idx_y = schema6.get_index("node1", "y")

    state6.set_by_index(idx_x, 42)
    state6.set_by_index(idx_y, 84)

    test("get_by_index(idx_x) = 42", state6.get_by_index(idx_x) == 42)
    test("get_by_index(idx_y) = 84", state6.get_by_index(idx_y) == 84)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("All MemoryState tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
