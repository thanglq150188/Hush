"""Base class cho tất cả các node trong workflow."""

from abc import ABC
from typing import Dict, Any, Callable, Optional, List, TYPE_CHECKING
from datetime import datetime
from time import perf_counter
import traceback
import asyncio
import uuid

from hush.core.configs.node_config import NodeType
from hush.core.utils.context import get_current, _current_graph
from hush.core.utils.common import unique_name
from hush.core.loggings import LOGGER, format_log_data
from hush.core.states.ref import Ref

if TYPE_CHECKING:
    from hush.core.states import MemoryState


class BaseNode(ABC):
    """Base class cho tất cả các node trong workflow.

    Node là đơn vị xử lý cơ bản trong workflow. Mỗi node có:
    - inputs: Ánh xạ biến đầu vào (có thể là Ref hoặc literal)
    - outputs: Ánh xạ biến đầu ra
    - input_schema/output_schema: Định nghĩa các biến
    - core: Function thực thi logic chính

    Hỗ trợ kết nối node bằng operators:
    - >> : Kết nối tuần tự (hard edge)
    - >  : Kết nối điều kiện (soft edge, dùng cho branch)
    - << : Kết nối ngược
    """

    INNER_PROCESS = "__inner__"

    __slots__ = [
        'id',
        'name',
        'description',
        'type',
        'stream',
        'start',
        'end',
        'verbose',
        'sources',
        'targets',
        'inputs',
        'outputs',
        'input_schema',
        'output_schema',
        'core',
        'father',
        'contain_generation',
    ]

    def __init__(
        self,
        id: str = None,
        name: str = None,
        description: str = "",
        inputs: Dict[str, Any] = None,
        outputs: Dict[str, Any] = None,
        input_schema: Dict[str, Any] = None,
        output_schema: Dict[str, Any] = None,
        sources: List[str] = None,
        targets: List[str] = None,
        stream: bool = False,
        start: bool = False,
        end: bool = False,
        contain_generation: bool = False,
        verbose: bool = True
    ):
        self.id = id or uuid.uuid4().hex
        self.name = name or unique_name()
        self.description = description

        self.stream = stream
        self.start = start
        self.end = end
        self.verbose = verbose

        self.sources: List[str] = sources or []
        self.targets: List[str] = targets or []

        self.input_schema = input_schema or {}
        self.output_schema = output_schema or {}

        self.core: Optional[Callable] = None
        self.contain_generation = contain_generation
        # Đăng ký vào graph cha
        self.father = get_current()
        if self.father and hasattr(self.father, "add_node"):
            self.father.add_node(self)

        # Validate tên node
        if self.name and not self.name.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Tên node '{self.name}' chỉ được chứa ký tự alphanumeric, underscore và hyphen")

        # Chuẩn hóa các kết nối inputs/outputs
        self.inputs = self._normalize_connections(inputs, self.input_schema)
        self.outputs = self._normalize_connections(outputs, self.output_schema)

        # Lỗi nếu có key trùng trong cả inputs và outputs
        if self.inputs and self.outputs:
            overlapping_keys = set(self.inputs.keys()) & set(self.outputs.keys())
            if overlapping_keys:
                raise ValueError(
                    f"Node '{self.name}' có key trùng giữa input/output: {overlapping_keys}. "
                    "Tên biến input và output phải khác nhau."
                )

    def _normalize_connections(
        self,
        connections: Any,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Chuẩn hóa ánh xạ kết nối về format thống nhất.

        Chuyển đổi các format đầu vào khác nhau thành: {var_name: Ref} hoặc {var_name: literal}

        Các format được hỗ trợ:
            - inputs=some_node → {k: Ref(some_node, k) for k in schema}
            - inputs={"var": some_node} → {"var": Ref(some_node, "var")}
            - inputs={"var": some_node["other"]} → {"var": Ref(some_node, "other")}
            - inputs={"var": Ref(node, "other")} → {"var": Ref(node, "other")}
            - inputs={("a", "b"): some_node} → {"a": Ref(some_node, "a"), "b": Ref(some_node, "b")}
            - inputs={"var": PARENT["x"]} → {"var": Ref(father, "x")} (PARENT resolve thành father node)
        """
        if connections is None:
            return {}

        def resolve_node(node):
            """Resolve PARENT thành father node."""
            if hasattr(node, 'name') and node.name == "__PARENT__":
                return self.father if self.father else node
            return node

        # Trường hợp: connections là một node reference (inputs=some_node)
        if hasattr(connections, "name"):
            resolved = resolve_node(connections)
            if schema:
                return {k: Ref(resolved, k) for k in schema}
            return {}

        # Trường hợp: connections là dict
        if isinstance(connections, dict):
            result = {}
            for key, value in connections.items():
                # Xử lý Ref trực tiếp - giữ nguyên operations!
                if isinstance(value, Ref):
                    resolved = resolve_node(value.raw_node)
                    # Giữ nguyên _ops khi tạo Ref mới với resolved node
                    result[key] = Ref(resolved, value.var, value.ops)
                # Xử lý dict format cũ {"var": node["other_var"]} để tương thích ngược
                elif isinstance(value, dict) and value:
                    ref_node, ref_var = next(iter(value.items()))
                    resolved = resolve_node(ref_node)
                    result[key] = Ref(resolved, ref_var)
                # Xử lý tuple keys: {("a", "b"): node} → mở rộng thành cả hai
                elif isinstance(key, tuple):
                    for k in key:
                        if hasattr(value, "name"):
                            resolved = resolve_node(value)
                            result[k] = Ref(resolved, k)
                        else:
                            result[k] = value
                else:
                    # Single key: {"var": node} → {"var": Ref(node, "var")}
                    if hasattr(value, "name"):
                        resolved = resolve_node(value)
                        result[key] = Ref(resolved, key)
                    else:
                        # Giá trị literal
                        result[key] = value
            return result

        return {}

    @property
    def full_name(self) -> str:
        """Đường dẫn phân cấp đầy đủ của node."""
        if self.father:
            return f"{self.father.full_name}.{self.name}"
        return self.name

    def identity(self, context_id: str) -> str:
        """Đường dẫn đầy đủ kèm context_id."""
        return f"{self.full_name}[{context_id or 'main'}]"

    def __getitem__(self, item: str) -> Ref:
        """Cho phép cú pháp node["var"] để tham chiếu output cụ thể."""
        return Ref(self, item)

    def __rshift__(self, other):
        """node >> other: kết nối node này đến other."""
        edge_type = "condition" if self.type == "branch" else "normal"

        if isinstance(other, list):
            for node in other:
                if hasattr(self.father, "add_edge"):
                    self.father.add_edge(self.name, node.name, edge_type)
            return other
        elif hasattr(other, 'name'):
            if hasattr(self.father, "add_edge"):
                self.father.add_edge(self.name, other.name, edge_type)
            return other
        return NotImplemented

    def __lshift__(self, other):
        """node << other: kết nối other đến node này."""
        edge_type = "condition" if self.type == "branch" else "normal"

        if isinstance(other, list):
            for node in other:
                if hasattr(self.father, "add_edge"):
                    self.father.add_edge(node.name, self.name, edge_type)
            return other
        elif hasattr(other, 'name'):
            if hasattr(self.father, "add_edge"):
                self.father.add_edge(other.name, self.name, edge_type)
            return self
        return NotImplemented

    def __rrshift__(self, other):
        """[n1, n2] >> self"""
        self.__lshift__(other)
        return self

    def __rlshift__(self, other):
        """[n1, n2] << self"""
        self.__rshift__(other)
        return self

    def __gt__(self, other):
        """node > other: soft edge (không tính vào ready_count).

        Dùng cho output của branch khi chỉ một nhánh được thực thi.
        Ví dụ: case_a > merge_node (merge chờ BẤT KỲ MỘT predecessor)
        """
        edge_type = "condition" if self.type == "branch" else "normal"

        if isinstance(other, list):
            for node in other:
                if hasattr(self.father, "add_edge"):
                    self.father.add_edge(self.name, node.name, edge_type, soft=True)
            return other
        elif hasattr(other, 'name'):
            if hasattr(self.father, "add_edge"):
                self.father.add_edge(self.name, other.name, edge_type, soft=True)
            return other
        return NotImplemented

    def __lt__(self, other):
        """node < other: soft edge ngược.

        Dùng cho output của branch khi chỉ một nhánh được thực thi.
        Ví dụ: merge_node < case_a (merge chờ BẤT KỲ MỘT predecessor)
        """
        edge_type = "condition" if self.type == "branch" else "normal"

        if isinstance(other, list):
            for node in other:
                if hasattr(self.father, "add_edge"):
                    self.father.add_edge(node.name, self.name, edge_type, soft=True)
            return other
        elif hasattr(other, 'name'):
            if hasattr(self.father, "add_edge"):
                self.father.add_edge(other.name, self.name, edge_type, soft=True)
            return self
        return NotImplemented

    def __call__(self, **kwargs) -> Dict[str, Any]:
        """Test nhanh: gọi node trực tiếp với inputs.

        Sử dụng:
            node = SomeNode(name="test", ...)
            result = node(a=1, b=2)
            # hoặc
            result = node(**{"a": 1, "b": 2})

        Returns:
            Dict các output từ việc thực thi node.
        """
        from hush.core.states import StateSchema, MemoryState

        # Lưu kwargs làm inputs trực tiếp (bỏ qua connection resolution)
        self.inputs = kwargs

        # Tạo schema từ node này (đăng ký inputs, outputs, metadata)
        schema = StateSchema(node=self)
        state = MemoryState(schema)

        # Chạy đồng bộ
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Nếu đã trong async context, tạo loop mới
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(asyncio.run, self.run(state)).result()
        else:
            result = loop.run_until_complete(self.run(state))

        return result

    def is_base_node(self) -> bool:
        return True

    def get_inputs(self, state: 'MemoryState', context_id: str) -> Dict[str, Any]:
        """Lấy giá trị input từ state dựa trên ánh xạ kết nối.

        Sử dụng state[this_node, var_name, ctx] để:
        1. Resolve đến vị trí lưu trữ canonical qua schema index
        2. Tự động áp dụng các Ref operation (như ['key'].apply(len))

        Schema đã resolve Ref chain tại thời điểm build, nên ta đọc
        từ tên biến của chính node này - index và ops đã được tính trước.
        """
        result = {}

        for var_name, ref in self.inputs.items():
            if isinstance(ref, Ref):
                # Đọc từ biến của node này - schema xử lý index resolution và ops
                value = state[self.full_name, var_name, context_id]
                result[var_name] = value
            else:
                # Giá trị literal
                result[var_name] = ref

        return result

    def get_outputs(self, state: 'MemoryState', context_id: str) -> Dict[str, Any]:
        """Lấy giá trị output từ state.

        Đọc trực tiếp từ các biến output của node này. Output connections
        (outputs={...}) đã được schema resolve tại thời điểm build -
        chúng tạo ref ở vị trí đích, không phải ở node này.
        """
        result = {}
        for var_name in self.output_schema:
            result[var_name] = state[self.full_name, var_name, context_id]
        return result

    def store_result(
        self,
        state: 'MemoryState',
        result: Dict[str, Any],
        context_id: str
    ) -> None:
        """Lưu dict kết quả vào state.

        Sử dụng state[node, var, ctx] = value cho lưu trữ O(1) dựa trên index.
        """
        if not result:
            return

        for key, value in result.items():
            state[self.full_name, key, context_id] = value

    def _log(
        self,
        request_id: str,
        context_id: Optional[str],
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        duration_ms: float
    ) -> None:
        """Log tóm tắt thực thi node với inputs, outputs và duration."""
        if self.verbose:
            _context_id = context_id or "main"
            _request_id = request_id or "unknown"
            LOGGER.info(
                "%s - %s: %s\\[%s] (%.1fms) %s -> %s",
                _request_id, str(self.type).upper(), self.full_name, _context_id,
                duration_ms, format_log_data(inputs), format_log_data(outputs)
            )

    async def run(
        self,
        state: 'MemoryState',
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Thực thi node."""
        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        perf_start = perf_counter()
        _inputs = {}
        _outputs = {}

        try:
            _inputs = self.get_inputs(state, context_id)

            if asyncio.iscoroutinefunction(self.core):
                _outputs = await self.core(**_inputs)
            else:
                _outputs = self.core(**_inputs)

            self.store_result(state, _outputs, context_id)

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

    def get_input_variables(self) -> List[str]:
        """Trả về danh sách tên biến input."""
        return list(self.input_schema.keys()) if self.input_schema else []

    def get_output_variables(self) -> List[str]:
        """Trả về danh sách tên biến output."""
        return list(self.output_schema.keys()) if self.output_schema else []

    def specific_metadata(self) -> Dict[str, Any]:
        """Trả về metadata riêng của subclass. Override ở các subclass."""
        return {}

    def metadata(self) -> Dict[str, Any]:
        """Tạo dictionary metadata cho node."""
        def get_connect_key(value):
            if isinstance(value, Ref):
                return {value.node: value.var}
            return value

        result = {
            "id": self.id,
            "name": self.full_name,
            "type": str(self.type).replace('NodeType.', '').lower()
        }

        if self.description:
            result["description"] = self.description

        if self.inputs:
            result["input_connects"] = {k: get_connect_key(v) for k, v in self.inputs.items()}

        if self.outputs:
            result["output_connects"] = {k: get_connect_key(v) for k, v in self.outputs.items()}

        if self.sources:
            result["sources"] = f"<- ({','.join(self.sources)})"

        if self.targets:
            result["targets"] = f"-> ({','.join(self.targets)})"

        for flag in ["stream", "start", "end"]:
            if getattr(self, flag, False):
                result[flag] = True

        result.update({k: v for k, v in self.specific_metadata().items() if v})

        return result


class DummyNode(BaseNode):
    """Dummy node cho các marker START/END/PARENT."""

    type: NodeType = "dummy"

    def __init__(self, name: str):
        super().__init__(name=name)

    def __rshift__(self, other):
        """START >> node"""
        if self == START:
            current_graph = get_current()
            if current_graph and hasattr(current_graph, 'add_edge'):
                if isinstance(other, list):
                    for node in other:
                        current_graph.add_edge(self.name, node.name)
                    return other
                elif hasattr(other, 'name'):
                    current_graph.add_edge(self.name, other.name)
                    return other
        return super().__rshift__(other)

    def __rrshift__(self, other):
        """[nodes] >> START or [nodes] >> END"""
        current_graph = get_current()
        if current_graph and hasattr(current_graph, 'add_edge'):
            if self == START:
                if isinstance(other, list):
                    for node in other:
                        current_graph.add_edge(self.name, node.name)
                elif hasattr(other, 'name'):
                    current_graph.add_edge(self.name, other.name)
                return self

            elif self == END:
                if isinstance(other, list):
                    for node in other:
                        current_graph.add_edge(node.name, self.name)
                elif hasattr(other, 'name'):
                    current_graph.add_edge(other.name, self.name)
                return self

        return self

    def __rlshift__(self, other):
        """node >> END"""
        if self == END:
            current_graph = get_current()
            if current_graph and hasattr(current_graph, 'add_edge'):
                if isinstance(other, list):
                    for node in other:
                        current_graph.add_edge(node.name, self.name)
                    return self
                elif hasattr(other, 'name'):
                    current_graph.add_edge(other.name, self.name)
                    return self
        return self

    def __gt__(self, other):
        """START > node or node > END (soft edge)"""
        if self == START:
            current_graph = get_current()
            if current_graph and hasattr(current_graph, 'add_edge'):
                if isinstance(other, list):
                    for node in other:
                        current_graph.add_edge(self.name, node.name, soft=True)
                    return other
                elif hasattr(other, 'name'):
                    current_graph.add_edge(self.name, other.name, soft=True)
                    return other
        return super().__gt__(other)

    def __rgt__(self, other):
        """[nodes] > END (soft edge)"""
        current_graph = get_current()
        if current_graph and hasattr(current_graph, 'add_edge'):
            if self == END:
                if isinstance(other, list):
                    for node in other:
                        current_graph.add_edge(node.name, self.name, soft=True)
                elif hasattr(other, 'name'):
                    current_graph.add_edge(other.name, self.name, soft=True)
                return self
        return self


# Các dummy node toàn cục
START = DummyNode("__START__")
END = DummyNode("__END__")
PARENT = DummyNode("__PARENT__")
