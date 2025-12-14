"""Abstract base class for workflow state."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import uuid

from hush.core.states.schema import StateSchema

__all__ = ["BaseState"]

# Pre-generate UUID function for speed
_uuid4 = uuid.uuid4


class BaseState(ABC):
    """Abstract base class for workflow state storage.

    All state backends (Memory, Redis, etc.) inherit from this class.
    Uses StateSchema for variable redirects (zero-copy linking).

    Subclasses only need to implement 3 methods:
    - _get_value(node, var, ctx) -> Any
    - _set_value(node, var, ctx, value) -> None
    - _iter_keys() -> Iterator

    Example:
        class MyCustomState(BaseState):
            def _get_value(self, node, var, ctx):
                return self.storage.get((node, var, ctx))

            def _set_value(self, node, var, ctx, value):
                self.storage[(node, var, ctx)] = value

            def _iter_keys(self):
                return iter(self.storage.keys())
    """

    __slots__ = ("schema", "_execution_order", "_user_id", "_session_id", "_request_id")

    def __init__(
        self,
        schema: StateSchema,
        inputs: Dict[str, Any] = None,
        user_id: str = None,
        session_id: str = None,
        request_id: str = None,
    ) -> None:
        """Initialize state with schema.

        Args:
            schema: StateSchema defining structure and redirects
            inputs: Initial input values {var_name: value}
            user_id: User identifier (auto-generated if None)
            session_id: Session identifier (auto-generated if None)
            request_id: Request identifier (auto-generated if None)
        """
        self.schema = schema
        self._execution_order: List[Dict[str, str]] = []

        # Direct assignment instead of dict
        self._user_id = user_id if user_id is not None else str(_uuid4())
        self._session_id = session_id if session_id is not None else str(_uuid4())
        self._request_id = request_id if request_id is not None else str(_uuid4())

        # Apply inputs (override defaults)
        if inputs:
            name = schema.name
            _set = self._set_value
            for var, value in inputs.items():
                _set(name, var, None, value)

    # =========================================================================
    # Abstract Methods - Subclasses implement these
    # =========================================================================

    @abstractmethod
    def _get_value(self, node: str, var: str, ctx: Optional[str]) -> Any:
        """Get value from storage. Returns None if not exists."""
        pass

    @abstractmethod
    def _set_value(self, node: str, var: str, ctx: Optional[str], value: Any) -> None:
        """Set value in storage."""
        pass

    @abstractmethod
    def _iter_keys(self):
        """Iterate over (node, var, ctx) keys in storage."""
        pass

    # =========================================================================
    # Core Access (with schema redirect)
    # =========================================================================

    def __setitem__(self, key: Tuple[str, str, Optional[str]], value: Any) -> None:
        """Set value with schema redirect: state[node, var, context] = value"""
        if len(key) != 3:
            raise ValueError("Key must be (node, var, context)")

        node, var, ctx = key
        real_node, real_var = self.schema.resolve(node, var)
        self._set_value(real_node, real_var, ctx, value)

    def __getitem__(self, key: Tuple[str, str, Optional[str]]) -> Any:
        """Get value with schema redirect: state[node, var, context]"""
        if len(key) != 3:
            raise ValueError("Key must be (node, var, context)")

        node, var, ctx = key
        real_node, real_var = self.schema.resolve(node, var)
        return self._get_value(real_node, real_var, ctx)

    def get(self, node: str, var: str, ctx: Optional[str] = None) -> Any:
        """Get value with explicit parameters."""
        real_node, real_var = self.schema.resolve(node, var)
        return self._get_value(real_node, real_var, ctx)

    def __contains__(self, key: Tuple[str, str]) -> bool:
        """Check if (node, var) has any value."""
        if len(key) != 2:
            return False
        node, var = key
        real_node, real_var = self.schema.resolve(node, var)
        for k in self._iter_keys():
            if k[0] == real_node and k[1] == real_var:
                return True
        return False

    def __iter__(self):
        """Iterate over (node, var, ctx) keys."""
        return self._iter_keys()

    # =========================================================================
    # Execution Tracking
    # =========================================================================

    def record_execution(
        self,
        node_name: str,
        parent: str,
        context_id: str
    ) -> None:
        """Record node execution for observability."""
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
        return self.schema.name

    @property
    def execution_order(self) -> List[Dict[str, str]]:
        return self._execution_order.copy()

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "user_id": self._user_id,
            "session_id": self._session_id,
            "request_id": self._request_id,
        }

    @property
    def user_id(self) -> str:
        return self._user_id

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def request_id(self) -> str:
        return self._request_id

    # =========================================================================
    # Context Manager & Utils
    # =========================================================================

    def __enter__(self) -> "BaseState":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"request_id='{self.request_id[:8]}...')"
        )

    def show(self) -> None:
        """Debug display of current state values."""
        print(f"\n=== {self.__class__.__name__}: {self.name} ===")

        # Group by (node, var)
        grouped: Dict[Tuple[str, str], Dict[Optional[str], Any]] = {}
        for key in self._iter_keys():
            node, var, ctx = key
            if (node, var) not in grouped:
                grouped[(node, var)] = {}
            grouped[(node, var)][ctx] = self._get_value(node, var, ctx)

        for (node, var), contexts in sorted(grouped.items()):
            if len(contexts) == 1:
                ctx, value = next(iter(contexts.items()))
                value_str = repr(value)[:50] + "..." if len(repr(value)) > 50 else repr(value)
                print(f"  {node}.{var}[{ctx}] = {value_str}")
            else:
                print(f"  {node}.{var}:")
                for ctx, value in contexts.items():
                    value_str = repr(value)[:50] + "..." if len(repr(value)) > 50 else repr(value)
                    print(f"    [{ctx}] = {value_str}")

        print(f"\nTotal keys: {sum(1 for _ in self._iter_keys())}")
