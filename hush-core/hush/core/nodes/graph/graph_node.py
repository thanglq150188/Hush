"""Graph node để quản lý subgraph các node trong workflow."""

from datetime import datetime
from time import perf_counter
from typing import Dict, Literal, Any, Optional

from hush.core.utils.common import Param
from collections import defaultdict
import asyncio
import traceback

from hush.core.configs.edge_config import EdgeConfig, EdgeType
from hush.core.configs.node_config import NodeType
from hush.core.nodes.base import BaseNode, START, END, PARENT
from hush.core.states import MemoryState, StateSchema, Ref
from hush.core.utils.context import _current_graph
from hush.core.utils.bimap import BiMap
from hush.core.loggings import LOGGER


NodeFlowType = Literal["MERGE", "FORK", "BLOOM", "BRANCH", "NORMAL", "OTHER"]


class GraphNode(BaseNode):
    """Node chứa và quản lý một subgraph các node.

    Cho phép tổ chức các node thành subgraph tái sử dụng với thực thi
    song song các nhánh và điều khiển luồng phù hợp.

    Hỗ trợ:
    - Context manager (with GraphNode() as graph)
    - Entry/exit node tự động
    - Thực thi song song các node độc lập
    - Soft/hard edge cho branch merging
    """

    __slots__ = [
        '_token',
        '_nodes',
        'entries',
        'exits',
        'prevs',
        'nexts',
        'ready_count',
        'flowtype_map',
        '_edges',
        '_edges_lookup',
        '_is_building'
    ]

    type: NodeType = "graph"

    def __init__(self, **kwargs):
        """Khởi tạo GraphNode."""
        super().__init__(**kwargs)
        self._token = None
        self._is_building = True
        self._nodes: Dict[str, BaseNode] = {}
        self._edges = []
        self._edges_lookup = {}
        self.entries = []
        self.exits = []
        self.prevs = defaultdict(list)
        self.nexts = defaultdict(list)
        self.flowtype_map = BiMap[str, NodeFlowType]()

    def __enter__(self):
        """Vào chế độ context manager."""
        self._token = _current_graph.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Thoát chế độ context manager."""
        _current_graph.reset(self._token)

    def _setup_endpoints(self):
        """Khởi tạo entry/exit node."""
        LOGGER.debug(f"Graph '{self.name}': đang khởi tạo endpoints...")

        if not self.entries:
            self.entries = [node for node in self._nodes if not self.prevs[node]]

        if not self.exits:
            self.exits = [node for node in self._nodes if not self.nexts[node]]

        if not self.entries:
            LOGGER.error(f"Graph '{self.name}': không tìm thấy entry node. Kiểm tra kết nối START >> node.")
            raise ValueError("Graph phải có ít nhất một entry node.")
        if not self.exits:
            LOGGER.error(f"Graph '{self.name}': không tìm thấy exit node. Kiểm tra kết nối node >> END.")
            raise ValueError("Graph phải có ít nhất một exit node.")

    def _setup_schema(self):
        """Khởi tạo input/output schema và inputs dictionary."""
        LOGGER.debug(f"Graph '{self.name}': đang tạo schema...")
        input_schema = {}
        output_schema = {}

        for _, node in self._nodes.items():
            # Kiểm tra inputs: nếu ref trỏ đến self (father), đó là graph input
            for var, ref in node.inputs.items():
                if isinstance(ref, Ref) and ref.raw_node is self:
                    # PARENT["x"] resolve thành father - đây là graph input
                    if var not in node.input_schema:
                        LOGGER.error(
                            f"Graph '{self.name}': biến '{var}' không có trong input schema của node '{node.name}'"
                        )
                        raise KeyError(
                            f"Biến không có trong input schema: "
                            f"{self.name}:{ref.var} <-- {node.name}.{var}"
                        )
                    input_schema[ref.var] = node.input_schema[var]

            # Kiểm tra outputs: nếu ref trỏ đến self (father), đó là graph output
            for var, ref in node.outputs.items():
                if isinstance(ref, Ref) and ref.raw_node is self:
                    # PARENT["x"] resolve thành father - đây là graph output
                    if var not in node.output_schema:
                        LOGGER.error(
                            f"Graph '{self.name}': biến '{var}' không có trong output schema của node '{node.name}'"
                        )
                        raise KeyError(
                            f"Biến không có trong output schema: "
                            f"{self.name}:{ref.var} <-- {node.name}.{var}"
                        )
                    output_schema[ref.var] = node.output_schema[var]

        self.input_schema = input_schema
        self.output_schema = output_schema

    def _build_flow_type(self):
        """Xác định flow type của mỗi node dựa trên pattern kết nối."""
        LOGGER.debug(f"Graph '{self.name}': đang xác định flow type của các node...")
        self.flowtype_map = BiMap[str, NodeFlowType]()

        # Phát hiện orphan node (không có kết nối nào)
        orphan_nodes = []

        for name, node in self._nodes.items():
            prev_count = len(self.prevs[name])
            next_count = len(self.nexts[name])

            # Kiểm tra orphan node (không phải start/end, không phải inner graph, và không có kết nối)
            if prev_count == 0 and next_count == 0 and not node.start and not node.end and name != BaseNode.INNER_PROCESS:
                orphan_nodes.append(name)

            flow_type: NodeFlowType = "OTHER"

            if node.type == "branch":
                flow_type = "BRANCH"
                for target in node.candidates:
                    if target in self._nodes and len(self.nexts[target]) == 1:
                        self.flowtype_map[target] = "NORMAL"

            elif prev_count > 1 and next_count > 1:
                flow_type = "BLOOM"
            elif prev_count > 1 and next_count == 1:
                flow_type = "MERGE"
            elif prev_count == 1 and next_count > 1:
                flow_type = "FORK"
            elif prev_count == 1 and next_count == 1:
                flow_type = "NORMAL"

            self.flowtype_map[name] = flow_type

        # Cảnh báo về orphan node
        if orphan_nodes:
            LOGGER.warning(
                f"Graph '{self.full_name}': phát hiện orphan node (không có edge): {orphan_nodes}. "
                "Các node này sẽ không bao giờ được thực thi."
            )

    def build(self):
        """Build graph bằng cách build các node con trước, sau đó graph này."""
        for node in self._nodes.values():
            if hasattr(node, 'build'):
                node.build()

        self._setup_schema()
        self._build_flow_type()
        self._setup_endpoints()

        # Tính ready_count - chỉ đếm hard edge (không phải soft)
        # Soft edge (tạo bằng >) không tính vào ready_count
        # Điều này cho phép branch output merge mà không deadlock
        self.ready_count = {}
        for name in self._nodes:
            hard_pred_count = 0
            for pred in self.prevs[name]:
                edge = self._edges_lookup.get((pred, name))
                if edge and not edge.soft:
                    hard_pred_count += 1
                elif edge is None:
                    # Edge không tìm thấy trong lookup (không nên xảy ra, nhưng vẫn đếm)
                    hard_pred_count += 1
            self.ready_count[name] = hard_pred_count

        self._is_building = False

    @staticmethod
    def get_current_graph() -> Optional['GraphNode']:
        """Lấy graph hiện tại từ context."""
        try:
            return _current_graph.get()
        except LookupError:
            return None

    def add_node(self, node: BaseNode) -> BaseNode:
        """Thêm một node vào graph."""
        if not self._is_building:
            raise RuntimeError("Không thể thêm node sau khi graph đã được build")

        if node in [START, END]:
            return node

        # Cảnh báo nếu node cùng tên đã tồn tại (sẽ bị ghi đè)
        if node.name in self._nodes:
            LOGGER.warning(
                f"Graph '{self.name}': node '{node.name}' đã tồn tại và sẽ bị ghi đè"
            )

        self._nodes[node.name] = node

        if hasattr(node, 'start') and node.start:
            if node.name not in self.entries:
                self.entries.append(node.name)

        if hasattr(node, 'end') and node.end:
            if node.name not in self.exits:
                self.exits.append(node.name)

        return node

    def add_edge(self, source: str, target: str, type: EdgeType = "normal", soft: bool = False):
        """Thêm một edge giữa hai node.

        Args:
            source: Tên node nguồn
            target: Tên node đích
            type: Loại edge (normal, lookback, condition)
            soft: Nếu True, edge không tính vào ready_count.
                  Dùng cho branch output khi chỉ một nhánh được thực thi.
                  Tạo bằng toán tử >: case_a > merge_node
        """
        if not self._is_building:
            raise RuntimeError("Không thể thêm edge sau khi graph đã được build!")

        if source == START.name:
            if target not in self._nodes:
                raise ValueError(f"Node đích '{target}' không tìm thấy")

            target_node = self._nodes[target]
            target_node.start = True

            if target not in self.entries:
                self.entries.append(target)

            return

        if target == END.name:
            if source not in self._nodes:
                raise ValueError(f"Node nguồn '{source}' không tìm thấy")

            source_node = self._nodes[source]
            source_node.end = True

            if source not in self.exits:
                self.exits.append(source)

            return

        if target == PARENT.name:
            return

        if source not in self._nodes:
            raise ValueError(f"Node nguồn '{source}' không tìm thấy")
        if target not in self._nodes:
            raise ValueError(f"Node đích '{target}' không tìm thấy")

        new_edge = EdgeConfig(from_node=source, to_node=target, type=type, soft=soft)
        if (source, target) not in self._edges_lookup:
            self._edges.append(new_edge)
            self._edges_lookup[source, target] = new_edge
            self.nexts[source].append(target)
            self.prevs[target].append(source)

    def show(self, indent=0):
        """Hiển thị cấu trúc graph (debug)."""
        prefix = "  " * indent
        print(f"{prefix}Graph: {self.name}")
        print(f"{prefix}Nodes:", list(self._nodes.keys()))
        print(f"{prefix}Edges:")
        for edge in self._edges:
            soft_marker = " (soft)" if edge.soft else ""
            print(f"{prefix}  {edge.from_node} -> {edge.to_node}: {edge.type}{soft_marker}")
        print(f"{prefix}Ready count:", dict(self.ready_count))

        for node in self._nodes.values():
            if isinstance(node, GraphNode):
                node.show(indent + 1)

    async def run(
        self,
        state: 'MemoryState',
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Thực thi graph bằng cách chạy tất cả node theo thứ tự dependency."""

        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        perf_start = perf_counter()
        _inputs = {}
        _outputs = {}

        try:
            _inputs = self.get_inputs(state, context_id=context_id)

            if self._is_building:
                raise ValueError(
                    f"Graph {self.name} not built. "
                    "Must call graph.build() before execution!!"
                )

            active_tasks: Dict[str, asyncio.Task] = {}

            ready_count: Dict[str, int] = self.ready_count.copy()

            for entry in self.entries:
                task = asyncio.create_task(
                    name=entry,
                    coro=self._nodes[entry].run(state, context_id)
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

                    node = self._nodes[node_name]

                    if node.type == "branch":
                        branch_target = node.get_target(state, context_id)
                        if branch_target != END.name:
                            next_nodes = [branch_target]
                        else:
                            next_nodes = []
                    else:
                        next_nodes = self.nexts[node_name]

                    for next_node in next_nodes:
                        ready_count[next_node] -= 1

                        if ready_count[next_node] == 0:
                            task = asyncio.create_task(
                                name=next_node,
                                coro=self._nodes[next_node].run(state, context_id)
                            )
                            active_tasks[next_node] = task

            _outputs = self.get_outputs(state, context_id=context_id)

        except Exception as e:
            error_msg = traceback.format_exc()
            LOGGER.error(f"Error in node {self.name}: {str(e)}")
            LOGGER.error(error_msg)
            state[self.full_name, "error", context_id] = error_msg

        finally:
            end_time = datetime.now()
            duration_ms = (perf_counter() - perf_start) * 1000
            self._log(request_id, context_id, _inputs, _outputs, duration_ms)
            state[self.full_name, "start_time", context_id] = start_time
            state[self.full_name, "end_time", context_id] = end_time
            return _outputs


if __name__ == "__main__":
    from hush.core.nodes.transform.code_node import CodeNode

    def test(name: str, condition: bool):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        if not condition:
            raise AssertionError(f"Test failed: {name}")

    async def main():
        # =====================================================================
        # Test 1: Simple linear graph (A -> B -> C)
        # =====================================================================
        print("\n" + "=" * 60)
        print("Test 1: Simple linear graph (A -> B -> C)")
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

        result = await graph.run(state)
        test("linear graph result = 25", result["result"] == 25)
        test("state has graph result", state["linear_graph", "result", None] == 25)
        test("state has node_a result", state["linear_graph.node_a", "result", None] == 15)
        test("state has node_b result", state["linear_graph.node_b", "result", None] == 30)

        # =====================================================================
        # Test 2: Ref with operations (getitem, method calls)
        # =====================================================================
        print("\n" + "=" * 60)
        print("Test 2: Ref with operations (getitem, apply)")
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
        state2 = MemoryState(schema2)

        result2 = await graph2.run(state2)
        test("ref ops result", result2["result"] == "TEST: 5 items, sum=15")

        # =====================================================================
        # Test 3: Ref with apply() for function application
        # =====================================================================
        print("\n" + "=" * 60)
        print("Test 3: Ref with apply() for function application")
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

            node_e = CodeNode(
                name="merge_all",
                code_fn=lambda in_length, sorted_nums, in_total: {
                    "length": in_length,
                    "sorted": sorted_nums,
                    "total": in_total
                },
                inputs={
                    "in_length": node_b["length"],
                    "sorted_nums": node_c["sorted"],
                    "in_total": node_d["total"]
                },
                outputs=PARENT
            )

            START >> node_a >> [node_b, node_c, node_d] >> node_e >> END

        graph3.build()
        schema3 = StateSchema(graph3)
        state3 = MemoryState(schema3)

        result3 = await graph3.run(state3)
        test("apply(len) = 6", result3["length"] == 6)
        test("apply(sorted)", result3["sorted"] == [1, 2, 3, 5, 8, 9])
        test("apply(sum) = 28", result3["total"] == 28)

        # =====================================================================
        # Test 4: Ref with chained operations
        # =====================================================================
        print("\n" + "=" * 60)
        print("Test 4: Ref with chained operations")
        print("=" * 60)

        with GraphNode(name="ref_chain_graph") as graph4:
            node_a = CodeNode(
                name="data_source",
                code_fn=lambda: {"data": {"users": [{"name": "alice"}, {"name": "bob"}]}},
                inputs={}
            )

            node_b = CodeNode(
                name="get_first_user_upper",
                code_fn=lambda name: {"first_user": name},
                inputs={"name": node_a["data"]["users"][0]["name"].upper()},
                outputs=PARENT
            )

            START >> node_a >> node_b >> END

        graph4.build()
        schema4 = StateSchema(graph4)
        state4 = MemoryState(schema4)

        result4 = await graph4.run(state4)
        test("chained ops: first_user = ALICE", result4["first_user"] == "ALICE")

        # =====================================================================
        # Test 5: Ref with arithmetic operations
        # =====================================================================
        print("\n" + "=" * 60)
        print("Test 5: Ref with arithmetic operations")
        print("=" * 60)

        with GraphNode(name="ref_arithmetic_graph") as graph5:
            node_a = CodeNode(
                name="number_source",
                code_fn=lambda: {"value": 10},
                inputs={}
            )

            node_b = CodeNode(
                name="compute",
                code_fn=lambda computed: {"result": computed},
                inputs={"computed": (node_a["value"] + 5) * 2},
                outputs=PARENT
            )

            START >> node_a >> node_b >> END

        graph5.build()
        schema5 = StateSchema(graph5)
        state5 = MemoryState(schema5)

        result5 = await graph5.run(state5)
        test("arithmetic (10 + 5) * 2 = 30", result5["result"] == 30)

        # =====================================================================
        # Test 6: Soft edges for branch merging
        # =====================================================================
        print("\n" + "=" * 60)
        print("Test 6: Soft edges (> operator) for branch merging")
        print("=" * 60)

        with GraphNode(name="soft_edge_graph") as graph6:
            branch_node = CodeNode(
                name="branch",
                code_fn=lambda choice: {"selected": choice},
                inputs={"choice": PARENT["choice"]}
            )

            case_a = CodeNode(
                name="case_a",
                code_fn=lambda: {"value": "A"},
                inputs={}
            )

            case_b = CodeNode(
                name="case_b",
                code_fn=lambda: {"value": "B"},
                inputs={}
            )

            merge = CodeNode(
                name="merge",
                code_fn=lambda value: {"result": f"Merged: {value}"},
                inputs={"value": case_a["value"]},
                outputs={"result": PARENT["result"]}
            )

            START >> branch_node >> [case_a, case_b]
            case_a > merge  # soft edge
            case_b > merge  # soft edge
            merge >> END

        graph6.build()
        test("soft edges: merge ready_count = 0", graph6.ready_count['merge'] == 0)

        # =====================================================================
        # Test 7: Mixed hard and soft edges
        # =====================================================================
        print("\n" + "=" * 60)
        print("Test 7: Mixed hard and soft edges")
        print("=" * 60)

        with GraphNode(name="mixed_edge_graph") as graph7:
            node_a = CodeNode(
                name="node_a",
                code_fn=lambda: {"x": 1},
                inputs={}
            )

            node_b = CodeNode(
                name="node_b",
                code_fn=lambda: {"y": 2},
                inputs={}
            )

            node_c = CodeNode(
                name="node_c",
                code_fn=lambda: {"z": 3},
                inputs={}
            )

            merge = CodeNode(
                name="merge",
                code_fn=lambda x: {"result": x},
                inputs={"x": node_a["x"]},
                outputs={"result": PARENT["result"]}
            )

            START >> node_a >> merge  # hard edge
            START >> node_b
            START >> node_c
            node_b > merge  # soft edge
            node_c > merge  # soft edge
            merge >> END

        graph7.build()
        test("mixed edges: merge ready_count = 1", graph7.ready_count['merge'] == 1)

        schema7 = StateSchema(graph7)
        state7 = MemoryState(schema7)
        result7 = await graph7.run(state7)
        test("mixed edges execution", result7["result"] == 1)

        # =====================================================================
        # Test 8: Nested graph receiving inputs from another node
        # =====================================================================
        print("\n" + "=" * 60)
        print("Test 8: Nested graph with value flow")
        print("=" * 60)

        with GraphNode(name="outer_graph") as outer:
            data_source = CodeNode(
                name="data_source",
                code_fn=lambda: {"config": {"value": 5, "multiplier": 3}},
                inputs={}
            )

            with GraphNode(
                name="inner_processor",
                inputs={"x": data_source["config"]["value"]},
                outputs={"result": PARENT["inner_result"]}
            ) as inner_graph:
                double_node = CodeNode(
                    name="double",
                    code_fn=lambda x: {"doubled": x * 2},
                    inputs={"x": PARENT["x"]}
                )
                add_ten = CodeNode(
                    name="add_ten",
                    code_fn=lambda val: {"result": val + 10},
                    inputs={"val": double_node["doubled"]},
                    outputs={"result": PARENT["result"]}
                )
                START >> double_node >> add_ten >> END

            final_node = CodeNode(
                name="final",
                code_fn=lambda inner_val, mult: {"final_result": inner_val * mult},
                inputs={
                    "inner_val": inner_graph["result"],
                    "mult": data_source["config"]["multiplier"]
                },
                outputs={"final_result": PARENT["final_result"]}
            )

            START >> data_source >> inner_graph >> final_node >> END

        outer.build()
        schema8 = StateSchema(outer)
        state8 = MemoryState(schema8)
        result8 = await outer.run(state8)

        test("nested: inner_result = 20", result8.get("inner_result") == 20)
        test("nested: final_result = 60", result8.get("final_result") == 60)

        # Verify intermediate state values
        test("state has data_source config", state8["outer_graph.data_source", "config", None] == {"value": 5, "multiplier": 3})
        test("state has double result", state8["outer_graph.inner_processor.double", "doubled", None] == 10)

        # =====================================================================
        # Test 9: Schema extraction
        # =====================================================================
        print("\n" + "=" * 60)
        print("Test 9: Schema extraction")
        print("=" * 60)

        test("linear_graph has 'x' in input_schema", "x" in graph.input_schema)
        test("linear_graph has 'result' in output_schema", "result" in graph.output_schema)
        test("outer_graph has 'inner_result' in output_schema", "inner_result" in outer.output_schema)
        test("outer_graph has 'final_result' in output_schema", "final_result" in outer.output_schema)

        # =====================================================================
        # Summary
        # =====================================================================
        print("\n" + "=" * 60)
        print("All GraphNode tests passed!")
        print("=" * 60)

    asyncio.run(main())


# Alias đơn giản cho cú pháp gọn hơn
graph = GraphNode
