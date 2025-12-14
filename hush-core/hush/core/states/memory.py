"""In-memory dict-based workflow state."""

from typing import Any, Dict, Optional, Tuple

from hush.core.states.schema import StateSchema
from hush.core.states.base import BaseState

__all__ = ["MemoryState"]


class MemoryState(BaseState):
    """In-memory workflow state using dict storage.

    Fast, lightweight, suitable for single-process applications.
    State creation is O(n) where n = number of defaults.

    Example:
        schema = StateSchema("my_workflow")
        schema.link("llm", "messages", "prompt", "output")

        state = MemoryState(schema, inputs={"query": "hello"})
        state["prompt", "output", None] = "Hello"
        value = state["llm", "messages", None]  # "Hello" via redirect
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
        # Copy defaults from schema - each state has its own values
        self._data: Dict[Tuple[str, str, str], Any] = {
            (node, var, "main"): value
            for (node, var), value in schema.defaults.items()
        }
        super().__init__(schema, inputs, user_id=user_id, session_id=session_id, request_id=request_id)

    # =========================================================================
    # Abstract Method Implementations
    # =========================================================================

    def _get_value(self, node: str, var: str, ctx: Optional[str]) -> Any:
        return self._data.get((node, var, ctx or "main"))

    def _set_value(self, node: str, var: str, ctx: Optional[str], value: Any) -> None:
        self._data[(node, var, ctx or "main")] = value

    def _iter_keys(self):
        return iter(self._data.keys())

    # =========================================================================
    # Additional Methods
    # =========================================================================

    def __len__(self) -> int:
        """Number of stored values."""
        return len(self._data)
