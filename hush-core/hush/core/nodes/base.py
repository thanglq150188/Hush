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
from hush.core.utils.common import unique_name, Param
from hush.core.loggings import LOGGER, format_log_data
from hush.core.states.ref import Ref

if TYPE_CHECKING:
    from hush.core.states import MemoryState


class StarRef:
    """Đại diện cho node[...] - tất cả outputs của một node.

    Dùng với << để forward tất cả outputs:
        PARENT[...] << merge[...]  → set merge.outputs = PARENT

    Cú pháp: node[...] (dùng Ellipsis, không phải *)
    """

    __slots__ = ('_node',)

    def __init__(self, node: 'BaseNode'):
        self._node = node

    def __lshift__(self, other: 'StarRef'):
        """PARENT[...] << merge[...]

        Khi PARENT[...] << merge[...]:
        - self là StarRef của PARENT
        - other là StarRef của merge
        - Forward tất cả outputs của merge đến PARENT
        """
        if isinstance(other, StarRef):
            # self._node là PARENT (DummyNode), other._node là source node
            if hasattr(self._node, 'name') and self._node.name == "__PARENT__":
                source_node = other._node
                father = source_node.father
                # Forward mỗi output key đến PARENT
                for key in list(source_node.outputs.keys()):
                    param = source_node.outputs[key]
                    # Set value là Ref đến father (graph cha)
                    param.value = Ref(father, key)
        return other


class BaseNode(ABC):
    """Base class cho tất cả các node trong workflow.

    Node là đơn vị xử lý cơ bản trong workflow. Mỗi node có:
    - inputs: Dict[str, Param] - định nghĩa và kết nối biến đầu vào
    - outputs: Dict[str, Param] - định nghĩa và kết nối biến đầu ra
    - core: Function thực thi logic chính

    Mỗi Param chứa:
    - type: Kiểu dữ liệu (tự động infer nếu không chỉ định)
    - required: Có bắt buộc hay không
    - default: Giá trị mặc định
    - description: Mô tả
    - value: Ref hoặc literal value

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

        self.core: Optional[Callable] = None
        self.contain_generation = contain_generation
        # Đăng ký vào graph cha
        self.father = get_current()
        # Use getattr to avoid hasattr's double lookup
        add_node = getattr(self.father, "add_node", None)
        if add_node is not None:
            add_node(self)

        # Validate tên node
        if self.name and not self.name.replace('_', '').replace('-', '').isalnum():
            raise ValueError(f"Tên node '{self.name}' chỉ được chứa ký tự alphanumeric, underscore và hyphen")

        # Chuẩn hóa inputs/outputs thành Dict[str, Param]
        self.inputs: Dict[str, Param] = self._normalize_params(inputs)
        self.outputs: Dict[str, Param] = self._normalize_params(outputs)

        # Lỗi nếu có key trùng trong cả inputs và outputs
        if self.inputs and self.outputs:
            overlapping_keys = set(self.inputs.keys()) & set(self.outputs.keys())
            if overlapping_keys:
                raise ValueError(
                    f"Node '{self.name}' có key trùng giữa input/output: {overlapping_keys}. "
                    "Tên biến input và output phải khác nhau."
                )

    def _resolve_value(self, key: str, value: Any) -> Any:
        """Chuyển đổi value thành Ref hoặc giữ nguyên literal.

        Các format được hỗ trợ:
            - some_node → Ref(some_node, key)
            - some_node["other"] → Ref(some_node, "other")
            - Ref(node, "var") → giữ nguyên
            - PARENT["x"] → Ref(father, "x")
            - literal → giữ nguyên
        """
        def resolve_node(node):
            """Resolve PARENT thành father node."""
            if hasattr(node, 'name') and node.name == "__PARENT__":
                return self.father if self.father else node
            return node

        # Xử lý Ref trực tiếp - giữ nguyên operations
        if isinstance(value, Ref):
            resolved = resolve_node(value.raw_node)
            return Ref(resolved, value.var, value.ops)

        # Xử lý node reference: some_node → Ref(some_node, key)
        if hasattr(value, "name"):
            resolved = resolve_node(value)
            return Ref(resolved, key)

        # Giá trị literal
        return value

    def _normalize_params(
        self,
        params: Any
    ) -> Dict[str, Param]:
        """Chuẩn hóa inputs/outputs thành Dict[str, Param].

        Các format được hỗ trợ:
            - params=None → {}
            - params=PARENT → {} (sẽ xử lý sau khi có outputs parsed)
            - params={"var": Param(...)} → giữ nguyên, resolve value
            - params={"var": some_node} → {"var": Param(value=Ref(some_node, "var"))}
            - params={"var": some_node["other"]} → {"var": Param(value=Ref(some_node, "other"))}
            - params={"var": literal} → {"var": Param(value=literal)}
            - params={("a", "b"): some_node} → mở rộng thành cả hai keys
        """
        if params is None:
            return {}

        # Xử lý PARENT shorthand - trả về marker để xử lý trong _merge_params
        if hasattr(params, 'name') and params.name == "__PARENT__":
            return {"__FORWARD_TO_PARENT__": params}

        result = {}

        if isinstance(params, dict):
            for key, value in params.items():
                # Xử lý tuple keys: {("a", "b"): node} → mở rộng thành cả hai
                if isinstance(key, tuple):
                    for k in key:
                        resolved_value = self._resolve_value(k, value)
                        result[k] = Param(value=resolved_value)
                # Xử lý Param trực tiếp
                elif isinstance(value, Param):
                    # Resolve value trong Param nếu có
                    if value.value is not None:
                        value.value = self._resolve_value(key, value.value)
                    result[key] = value
                else:
                    # Tạo Param mới với value đã resolve (type auto-inferred)
                    resolved_value = self._resolve_value(key, value)
                    result[key] = Param(value=resolved_value)

        return result

    def _merge_params(
        self,
        schema: Dict[str, Param],
        user_provided: Dict[str, Any]
    ) -> Dict[str, Param]:
        """Merge schema (từ parsing) với user-provided inputs/outputs.

        - Nếu key đã tồn tại trong schema → chỉ gán value
        - Nếu key mới → tạo Param mới (type auto-inferred)
        - Nếu user_provided có marker __FORWARD_TO_PARENT__ → forward tất cả keys trong schema đến PARENT

        Args:
            schema: Dict[str, Param] từ parsing (ví dụ: từ function signature)
            user_provided: Dict từ user (Ref | literal | Param)

        Returns:
            Dict[str, Param] đã merge
        """
        # Copy schema để không mutate original
        result = {
            k: Param(type=v.type, required=v.required, default=v.default,
                     description=v.description, value=v.value)
            for k, v in schema.items()
        }

        if not user_provided:
            return result

        # Xử lý PARENT shorthand: outputs=PARENT → forward tất cả outputs đến parent
        if "__FORWARD_TO_PARENT__" in user_provided:
            parent_node = user_provided["__FORWARD_TO_PARENT__"]
            # Resolve PARENT thành father
            resolved_parent = self.father if self.father else parent_node
            # Tạo Ref cho mỗi key trong schema
            for key in result:
                result[key].value = Ref(resolved_parent, key)
            return result

        for key, value in user_provided.items():
            # Xử lý tuple keys
            if isinstance(key, tuple):
                for k in key:
                    self._merge_single_param(result, k, value)
            else:
                self._merge_single_param(result, key, value)

        return result

    def _merge_single_param(
        self,
        result: Dict[str, Param],
        key: str,
        value: Any
    ) -> None:
        """Merge một param vào result dict."""
        if key in result:
            # Key đã tồn tại → chỉ gán value
            if isinstance(value, Param):
                result[key].value = self._resolve_value(key, value.value) if value.value is not None else None
            else:
                result[key].value = self._resolve_value(key, value)
        else:
            # Key mới → tạo Param mới (type auto-inferred in Param.__post_init__)
            if isinstance(value, Param):
                if value.value is not None:
                    value.value = self._resolve_value(key, value.value)
                result[key] = value
            else:
                resolved_value = self._resolve_value(key, value)
                result[key] = Param(value=resolved_value)

    @property
    def full_name(self) -> str:
        """Đường dẫn phân cấp đầy đủ của node."""
        if self.father:
            return f"{self.father.full_name}.{self.name}"
        return self.name

    def identity(self, context_id: str) -> str:
        """Đường dẫn đầy đủ kèm context_id."""
        return f"{self.full_name}[{context_id or 'main'}]"

    def __getitem__(self, item) -> 'Ref':
        """Cho phép cú pháp node["var"] hoặc node[*] để tham chiếu output.

        - node["var"] → Ref đến output cụ thể
        - node[*] → StarRef đại diện tất cả outputs (dùng với << để forward)
        """
        if item is Ellipsis:  # node[...]
            return StarRef(self)
        return Ref(self, item)

    def __rshift__(self, other):
        """node >> other: kết nối node này đến other."""
        edge_type = "condition" if self.type == "branch" else "normal"
        # Cache add_edge lookup to avoid repeated hasattr
        add_edge = getattr(self.father, "add_edge", None)

        if isinstance(other, list):
            if add_edge is not None:
                for node in other:
                    add_edge(self.name, node.name, edge_type)
            return other
        elif getattr(other, 'name', None) is not None:
            if add_edge is not None:
                add_edge(self.name, other.name, edge_type)
            return other
        return NotImplemented

    def __lshift__(self, other):
        """node << other: kết nối other đến node này."""
        edge_type = "condition" if self.type == "branch" else "normal"
        add_edge = getattr(self.father, "add_edge", None)

        if isinstance(other, list):
            if add_edge is not None:
                for node in other:
                    add_edge(node.name, self.name, edge_type)
            return other
        elif getattr(other, 'name', None) is not None:
            if add_edge is not None:
                add_edge(other.name, self.name, edge_type)
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
        add_edge = getattr(self.father, "add_edge", None)

        if isinstance(other, list):
            if add_edge is not None:
                for node in other:
                    add_edge(self.name, node.name, edge_type, soft=True)
            return other
        elif getattr(other, 'name', None) is not None:
            if add_edge is not None:
                add_edge(self.name, other.name, edge_type, soft=True)
            return other
        return NotImplemented

    def __lt__(self, other):
        """node < other: soft edge ngược.

        Dùng cho output của branch khi chỉ một nhánh được thực thi.
        Ví dụ: merge_node < case_a (merge chờ BẤT KỲ MỘT predecessor)
        """
        edge_type = "condition" if self.type == "branch" else "normal"
        add_edge = getattr(self.father, "add_edge", None)

        if isinstance(other, list):
            if add_edge is not None:
                for node in other:
                    add_edge(node.name, self.name, edge_type, soft=True)
            return other
        elif getattr(other, 'name', None) is not None:
            if add_edge is not None:
                add_edge(other.name, self.name, edge_type, soft=True)
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

        # Tạo schema từ node này
        schema = StateSchema(node=self)
        # Truyền kwargs vào MemoryState để override inputs
        state = MemoryState(schema, inputs=kwargs)

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

        for var_name, param in self.inputs.items():
            # Luôn đọc từ state trước (có thể có giá trị từ MemoryState inputs hoặc Ref)
            value = state[self.full_name, var_name, context_id]
            if value is not None:
                result[var_name] = value
            elif param.value is not None and not isinstance(param.value, Ref):
                # Fallback: giá trị literal trong Param.value
                result[var_name] = param.value
            elif param.default is not None:
                # Fallback: giá trị default
                result[var_name] = param.default

        return result

    def get_outputs(self, state: 'MemoryState', context_id: str) -> Dict[str, Any]:
        """Lấy giá trị output từ state.

        Đọc trực tiếp từ các biến output của node này. Output connections
        (outputs={...}) đã được schema resolve tại thời điểm build -
        chúng tạo ref ở vị trí đích, không phải ở node này.
        """
        result = {}
        for var_name in self.outputs:
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
        return list(self.inputs.keys()) if self.inputs else []

    def get_output_variables(self) -> List[str]:
        """Trả về danh sách tên biến output."""
        return list(self.outputs.keys()) if self.outputs else []

    def specific_metadata(self) -> Dict[str, Any]:
        """Trả về metadata riêng của subclass. Override ở các subclass."""
        return {}

    def metadata(self) -> Dict[str, Any]:
        """Tạo dictionary metadata cho node."""
        def get_connect_key(param: Param):
            if isinstance(param.value, Ref):
                return {param.value.node: param.value.var}
            return param.value

        result = {
            "id": self.id,
            "name": self.full_name,
            "type": self.type  # Already a lowercase string (Literal type)
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
