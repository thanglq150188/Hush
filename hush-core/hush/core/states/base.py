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

    def __hash__(self) -> int:
        return hash(self._request_id)

    def __eq__(self, other) -> bool:
        if not isinstance(other, BaseState):
            return False
        return self._request_id == other._request_id

    def show(self) -> None:
        """Debug display of current state values.

        Shows all stored values AND all schema references with their resolved values.
        """
        print(f"\n=== {self.__class__.__name__}: {self.name} ===")

        # Collect all keys from storage
        stored_keys = set(self._iter_keys())

        # Collect all schema refs to show resolved values
        schema_refs = set()
        for (node, var), value in self.schema.values.items():
            from hush.core.states.ref import Ref
            if isinstance(value, Ref):
                # Add all contexts for this ref
                for n, v, ctx in stored_keys:
                    if n == value.node and v == value.var:
                        schema_refs.add((node, var, ctx))

        # Combine and sort all keys
        all_keys = stored_keys | schema_refs

        for node, var, ctx in sorted(all_keys):
            ref = self.schema.get_ref(node, var)
            ctx_str = f"[{ctx}]" if ctx else ""

            if ref:
                # This is a reference - show both the ref and resolved value
                resolved_value = self._get_value(ref.node, ref.var, ctx)
                value_str = repr(resolved_value)[:50] + "..." if len(repr(resolved_value)) > 50 else repr(resolved_value)
                print(f"  {node}.{var}{ctx_str} -> {ref.node}.{ref.var} = {value_str}")
            else:
                # Direct value
                value = self._get_value(node, var, ctx)
                value_str = repr(value)[:50] + "..." if len(repr(value)) > 50 else repr(value)
                print(f"  {node}.{var}{ctx_str} = {value_str}")
