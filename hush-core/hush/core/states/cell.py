"""Cell lưu trữ giá trị đa context cho workflow state."""

from typing import Any, Optional, Dict, List

DEFAULT_CONTEXT = "main"


class Cell:
    """Lưu trữ một biến có thể có nhiều giá trị trong các context/iteration khác nhau.

    Mỗi biến trong workflow có thể tồn tại trong nhiều context khác nhau,
    ví dụ như các iteration của loop. Cell quản lý tất cả các giá trị này
    và theo dõi thứ tự thêm vào.

    Attributes:
        contexts: Dict ánh xạ context_id sang giá trị
        versions: Stack theo dõi thứ tự các context được thêm vào
        default_value: Giá trị mặc định khi context không tồn tại
    """

    __slots__ = ('contexts', 'versions', 'default_value')

    def __init__(self, default_value: Any = None):
        """Khởi tạo Cell với giá trị mặc định.

        Args:
            default_value: Giá trị trả về khi context không tồn tại
        """
        self.contexts: Dict[str, Any] = {}  # context_id -> value
        self.versions: List[str] = []  # Stack theo dõi thứ tự context
        self.default_value = default_value

    def __setitem__(self, context_id: Optional[str], value: Any) -> None:
        """Set giá trị cho context cụ thể.

        Args:
            context_id: ID của context (None = default context "main")
            value: Giá trị cần lưu
        """
        if context_id is None:
            context_id = DEFAULT_CONTEXT

        self.contexts[context_id] = value

        if context_id not in self.versions:
            self.versions.append(context_id)

    def __getitem__(self, context_id: Optional[str] = None) -> Any:
        """Lấy giá trị từ context cụ thể.

        Args:
            context_id: ID của context (None = default context "main")

        Returns:
            Giá trị của context hoặc default_value nếu không tồn tại
        """
        if context_id is None:
            context_id = DEFAULT_CONTEXT

        return self.contexts.get(context_id, self.default_value)

    def get_latest(self) -> Any:
        """Lấy giá trị từ context được thêm gần nhất.

        Returns:
            Giá trị của context mới nhất hoặc default_value nếu chưa có context nào
        """
        if self.versions:
            latest_context = self.versions[-1]
            return self.contexts[latest_context]
        return self.default_value

    def pop_context(self, context_id: str) -> Any:
        """Xóa context và trả về giá trị của nó.

        Args:
            context_id: ID của context cần xóa

        Returns:
            Giá trị của context đã xóa hoặc default_value nếu không tồn tại
        """
        if context_id in self.versions:
            self.versions.remove(context_id)
        return self.contexts.pop(context_id, self.default_value)

    def __delitem__(self, context_id: str) -> None:
        """Xóa context: del cell['context_id']"""
        self.pop_context(context_id)

    def __contains__(self, context_id: str) -> bool:
        """Kiểm tra context có tồn tại: 'context_id' in cell"""
        return context_id in self.contexts

    def __repr__(self) -> str:
        return f"Cell(contexts={self.contexts}, latest={self.versions[-1] if self.versions else None})"