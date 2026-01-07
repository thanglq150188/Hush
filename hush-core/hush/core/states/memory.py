"""In-memory dict-based workflow state with index-based storage."""

from typing import Any, Dict, Optional, Tuple

from hush.core.states.schema import StateSchema
from hush.core.states.base import BaseState

__all__ = ["MemoryState"]


class MemoryState(BaseState):
    """In-memory workflow state using index-based storage.

    Fast, lightweight, suitable for single-process applications.
    Uses (idx, ctx) as keys for O(1) lookups.

    Example:
        schema = StateSchema(graph)
        state = MemoryState(schema, inputs={"query": "hello"})
        state["node", "var", None] = "value"
        value = state["node", "var", None]
    """

    __slots__ = ("_data",)

    def __init__(
        self,
        schema: StateSchema,
        inputs: Dict[str, Any] = None,
        user_id: str = None,
        session_id: str = None,
        request_id: str = None,
    ) -> None:
        # Initialize storage with defaults from schema
        self._data: Dict[Tuple[int, str], Any] = {}

        # Copy defaults from schema - use "main" as default context
        for idx, default_value in enumerate(schema._defaults):
            if default_value is not None:
                self._data[(idx, "main")] = default_value

        super().__init__(schema, inputs, user_id=user_id, session_id=session_id, request_id=request_id)

    # =========================================================================
    # Abstract Method Implementations (index-based)
    # =========================================================================

    def _get_value_by_idx(self, idx: int, ctx: Optional[str]) -> Any:
        return self._data.get((idx, ctx or "main"))

    def _set_value_by_idx(self, idx: int, ctx: Optional[str], value: Any) -> None:
        self._data[(idx, ctx or "main")] = value

    def _iter_stored(self):
        return iter(self._data.keys())

    # =========================================================================
    # Additional Methods
    # =========================================================================

    def __len__(self) -> int:
        """Number of stored values."""
        return len(self._data)
