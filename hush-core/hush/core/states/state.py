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
