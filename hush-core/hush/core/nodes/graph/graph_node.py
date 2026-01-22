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
from hush.core.states import MemoryState, Ref
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
        'has_soft_preds',  # Set of nodes that have soft predecessors
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
        self.has_soft_preds = set()  # Các node có soft predecessor

    def __enter__(self):
        """Vào chế độ context manager."""
        self._token = _current_graph.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Thoát chế độ context manager."""
        _current_graph.reset(self._token)

    def _setup_endpoints(self):
        """Khởi tạo entry/exit node."""
        LOGGER.debug("Graph [highlight]%s[/highlight]: đang khởi tạo endpoints...", self.name)

        if not self.entries:
            self.entries = [node for node in self._nodes if not self.prevs[node]]

        if not self.exits:
            self.exits = [node for node in self._nodes if not self.nexts[node]]

        if not self.entries:
            LOGGER.error("Graph [highlight]%s[/highlight]: không tìm thấy entry node. Kiểm tra kết nối START >> node.", self.name)
            raise ValueError("Graph phải có ít nhất một entry node.")
        if not self.exits:
            LOGGER.error("Graph [highlight]%s[/highlight]: không tìm thấy exit node. Kiểm tra kết nối node >> END.", self.name)
            raise ValueError("Graph phải có ít nhất một exit node.")

    def _setup_schema(self):
        """Khởi tạo inputs/outputs từ các node con.

        Scan các node con để tìm các ref trỏ đến PARENT (self) -
        đó chính là inputs/outputs của graph.
        """
        LOGGER.debug("Graph [highlight]%s[/highlight]: đang tạo schema...", self.name)
        graph_inputs = {}
        graph_outputs = {}

        for _, node in self._nodes.items():
            # Kiểm tra inputs: nếu ref trỏ đến self (father), đó là graph input
            for var, param in node.inputs.items():
                if isinstance(param.value, Ref) and param.value.raw_node is self:
                    # PARENT["x"] resolve thành father - đây là graph input
                    # Copy Param từ node con, giữ nguyên type/required/default/description
                    graph_inputs[param.value.var] = Param(
                        type=param.type,
                        required=param.required,
                        default=param.default,
                        description=param.description
                    )

            # Kiểm tra outputs: nếu ref trỏ đến self (father), đó là graph output
            for var, param in node.outputs.items():
                if isinstance(param.value, Ref) and param.value.raw_node is self:
                    # PARENT["x"] resolve thành father - đây là graph output
                    # Copy Param từ node con
                    graph_outputs[param.value.var] = Param(
                        type=param.type,
                        required=param.required,
                        default=param.default,
                        description=param.description
                    )

        # Merge với user-provided inputs/outputs (nếu có)
        self.inputs = self._merge_params(graph_inputs, self.inputs)
        self.outputs = self._merge_params(graph_outputs, self.outputs)

    def _build_flow_type(self):
        """Xác định flow type của mỗi node dựa trên pattern kết nối."""
        LOGGER.debug("Graph [highlight]%s[/highlight]: đang xác định flow type của các node...", self.name)
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
                "Graph [highlight]%s[/highlight]: phát hiện orphan node [muted](không có edge)[/muted]: %s. Các node này sẽ không bao giờ được thực thi.",
                self.full_name, orphan_nodes
            )

    def build(self):
        """Build graph bằng cách build các node con trước, sau đó graph này."""
        for node in self._nodes.values():
            if hasattr(node, 'build'):
                node.build()

        self._setup_schema()
        self._build_flow_type()
        self._setup_endpoints()

        # Tính ready_count:
        # - Hard edge (>>) đếm từng cái một
        # - Soft edge (>) của cùng node đích đếm chung là 1 (chờ BẤT KỲ một soft pred hoàn thành)
        # Ví dụ: A >> D, B > D, C > D => ready_count[D] = 2 (1 hard + 1 soft group)
        self.ready_count = {}
        self.has_soft_preds = set()
        for name in self._nodes:
            hard_pred_count = 0
            has_soft = False
            for pred in self.prevs[name]:
                edge = self._edges_lookup.get((pred, name))
                if edge and edge.soft:
                    has_soft = True
                elif edge and not edge.soft:
                    hard_pred_count += 1
                elif edge is None:
                    # Edge không tìm thấy trong lookup (không nên xảy ra, nhưng vẫn đếm)
                    hard_pred_count += 1
            # Soft edges đếm chung là 1 nếu có
            if has_soft:
                self.has_soft_preds.add(name)
                hard_pred_count += 1
            self.ready_count[name] = hard_pred_count

        self._is_building = False
        self._post_build()

    def _post_build(self):
        """Hook for subclasses to run after build. Override in subclasses."""
        pass

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
                "Graph [highlight]%s[/highlight]: node [highlight]%s[/highlight] đã tồn tại và sẽ bị ghi đè",
                self.name, node.name
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
        context_id: Optional[str] = None,
        parent_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Thực thi graph bằng cách chạy tất cả node theo thứ tự dependency.

        Args:
            state: Workflow state
            context_id: Context của graph này
            parent_context: Context của PARENT để truyền cho child nodes
        """

        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        perf_start = perf_counter()
        _inputs = {}
        _outputs = {}

        try:
            _inputs = self.get_inputs(state, context_id=context_id, parent_context=parent_context)

            if self._is_building:
                raise ValueError(
                    f"Graph {self.name} not built. "
                    "Must call graph.build() before execution!!"
                )

            active_tasks: Dict[str, asyncio.Task] = {}

            ready_count: Dict[str, int] = self.ready_count.copy()
            # Track nodes đã nhận soft edge completion (chỉ đếm 1 lần)
            soft_satisfied: set = set()

            for entry in self.entries:
                task = asyncio.create_task(
                    name=entry,
                    coro=self._nodes[entry].run(state, context_id, parent_context)
                )
                active_tasks[entry] = task

            while active_tasks:
                done_tasks, _ = await asyncio.wait(
                    active_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Cache dict lookups for performance
                nodes = self._nodes
                nexts = self.nexts
                edges_lookup = self._edges_lookup

                for task in done_tasks:
                    node_name = task.get_name()
                    active_tasks.pop(node_name)

                    node = nodes[node_name]

                    if node.type == "branch":
                        branch_target = node.get_target(state, context_id)
                        if branch_target != END.name:
                            next_nodes = [branch_target]
                        else:
                            next_nodes = []
                    else:
                        next_nodes = nexts[node_name]

                    for next_node in next_nodes:
                        # Kiểm tra edge type
                        edge = edges_lookup.get((node_name, next_node))
                        is_soft = edge and edge.soft

                        if is_soft:
                            # Soft edge: chỉ đếm 1 lần cho tất cả soft predecessors
                            if next_node in soft_satisfied:
                                continue  # Đã có soft pred khác hoàn thành
                            soft_satisfied.add(next_node)

                        ready_count[next_node] -= 1

                        if ready_count[next_node] == 0:
                            task = asyncio.create_task(
                                name=next_node,
                                coro=nodes[next_node].run(state, context_id, parent_context)
                            )
                            active_tasks[next_node] = task

            _outputs = self.get_outputs(state, context_id=context_id, parent_context=parent_context)
            self.store_result(state, _outputs, context_id)

        except Exception as e:
            error_msg = traceback.format_exc()
            LOGGER.error("[title]\\[%s][/title] Error in node [highlight]%s[/highlight]: %s", request_id, self.name, str(e))
            LOGGER.error(error_msg)
            state[self.full_name, "error", context_id] = error_msg

        finally:
            end_time = datetime.now()
            duration_ms = (perf_counter() - perf_start) * 1000
            self._log(request_id, context_id, _inputs, _outputs, duration_ms)
            state[self.full_name, "start_time", context_id] = start_time
            state[self.full_name, "end_time", context_id] = end_time

            # Record trace metadata for observability
            state.record_trace_metadata(
                node_name=self.full_name,
                context_id=context_id,
                name=self.name,
                input_vars=list(self.inputs.keys()) if self.inputs else [],
                output_vars=list(self.outputs.keys()) if self.outputs else [],
                parent_name=parent_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                contain_generation=False,
                metadata=self.metadata(),
            )
            return _outputs


# Alias đơn giản cho cú pháp gọn hơn
graph = GraphNode
