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
        """Thu thập đệ quy tất cả biến từ cây node.

        inputs/outputs là Dict[str, Param] với:
        - Param.value: Ref hoặc literal value
        - Param.default: giá trị mặc định
        """
        node_name = node.full_name
        inputs = node.inputs or {}
        outputs = node.outputs or {}

        # Đăng ký các biến input
        for var_name, param in inputs.items():
            # Ưu tiên value (Ref hoặc literal), sau đó là default
            if param.value is not None:
                self._register(node_name, var_name, param.value)
            else:
                self._register(node_name, var_name, param.default)

        # Đăng ký các biến output
        for var_name, param in outputs.items():
            if isinstance(param.value, Ref):
                # Output ref: node.var -> target.var (push value to target)
                ref = param.value
                if ref.has_ops:
                    raise ValueError(f"Output ref {node_name}.{var_name} -> {ref.node}.{ref.var} không được có operation")
                self._register(node_name, var_name, Ref(ref.node, ref.var, is_output=True))
            elif (node_name, var_name) not in self._indexer:
                # Không phải output ref, đăng ký với default
                self._register(node_name, var_name, param.default)

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
