from typing import Any, Optional, Dict, List

DEFAULT_CONTEXT = "main"


class Cell:
    """Storage cho một biến có thể có nhiều giá trị trong các context/loop khác nhau"""

    __slots__ = ('contexts', 'versions', 'default_value')

    def __init__(self, default_value: Any = None):
        self.contexts: Dict[str, Any] = {} # contextid -> value
        self.versions: List[str] = [] # Stack để track order của contexts
        self.default_value = default_value

    def __setitem__(self, context_id: Optional[str], value: Any) -> None:
        """Set giá trị cho context cụ thể. None = default context."""
        if context_id is None:
            context_id = DEFAULT_CONTEXT

        self.contexts[context_id] = value

        if context_id not in self.versions:
            self.versions.append(context_id)

    def __getitem__(self, context_id: Optional[str] = None) -> Any:
        """Get giá trị từ context cụ thể. None = default context."""
        if context_id is None:
            context_id = DEFAULT_CONTEXT

        return self.contexts.get(context_id, self.default_value)

    def get_latest(self) -> Any:
        """Get giá trị từ context gần nhất hoặc default"""
        if self.versions:
            latest_context = self.versions[-1]
            return self.contexts[latest_context]
        return self.default_value

    def pop_context(self, context_id: str) -> Any:
        """Remove context và trả về giá trị của nó"""
        if context_id in self.versions:
            self.versions.remove(context_id)
        return self.contexts.pop(context_id, self.default_value)

    def __delitem__(self, context_id: str) -> None:
        """del cell['context_id']"""
        self.pop_context(context_id)

    def __contains__(self, context_id: str) -> bool:
        """'context_id' in cell"""
        return context_id in self.contexts

    def __repr__(self) -> str:
        return f"VariableCell(contexts={self.contexts}, latest={self.versions[-1] if self.versions else None})"


if __name__ == "__main__":
    # Basic usage
    cell = Cell(default_value=0)

    # Set với context
    cell["loop1"] = 10
    cell["loop2"] = 20

    # None = default context ("main")
    cell[None] = 30  # Sets main context to 30

    print(f"loop1: {cell['loop1']}")  # 10
    print(f"loop2: {cell['loop2']}")  # 20
    print(f"main: {cell[None]}")  # 30
    print(f"latest: {cell.get_latest()}")  # 30 (main was added last)

    # Test with empty cell
    empty_cell = Cell(default_value=0)
    empty_cell[None] = 100  # Creates "main" context
    print(f"empty cell main: {empty_cell[None]}")  # 100

    empty_cell["loop1"] = 200
    print(f"empty cell latest: {empty_cell.get_latest()}")  # 200 (loop1 is latest)
    print(f"empty cell main: {empty_cell[None]}")  # 100

    empty_cell[None] = 150  # Updates main
    print(f"empty cell main: {empty_cell[None]}")  # 150
    print(f"empty cell latest: {empty_cell.get_latest()}")  # 200 (loop1 still latest)

    # Test __contains__ and __delitem__
    print(f"'loop1' in empty_cell: {'loop1' in empty_cell}")  # True
    del empty_cell["loop1"]
    print(f"'loop1' in empty_cell after del: {'loop1' in empty_cell}")  # False
    print(f"empty cell latest after del: {empty_cell.get_latest()}")  # 150 (main is now latest)

    # Test __repr__
    print(repr(empty_cell))  # VariableCell(contexts={'main': 150}, latest=main)