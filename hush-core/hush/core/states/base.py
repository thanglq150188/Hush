"""Abstract base class for workflow state with index-based resolution."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import uuid

from hush.core.states.schema import StateSchema
from hush.core.states.ref import Ref

__all__ = ["BaseState"]

# Pre-generate UUID function for speed
_uuid4 = uuid.uuid4


class BaseState(ABC):
    """Abstract base class for workflow state storage with index-based O(1) access.

    Uses StateSchema's index mapping for O(1) lookups regardless of reference depth.
    Ref operations (like ['key'].apply(len)) are applied automatically on read.

    The state maintains a values list where each index corresponds to a unique
    storage location. Multiple (node, var) keys can map to the same index (aliasing).

    Subclasses only need to implement storage for the values list:
    - _get_value_by_idx(idx, ctx) -> Any
    - _set_value_by_idx(idx, ctx, value) -> None
    - _iter_stored() -> Iterator[(idx, ctx)]

    Example:
        class MyCustomState(BaseState):
            def _get_value_by_idx(self, idx, ctx):
                return self.storage.get((idx, ctx))

            def _set_value_by_idx(self, idx, ctx, value):
                self.storage[(idx, ctx)] = value

            def _iter_stored(self):
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
            schema: StateSchema defining structure and index mappings
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
            for var, value in inputs.items():
                idx = schema.get_index(name, var)
                if idx >= 0:
                    self._set_value_by_idx(idx, None, value)

    # =========================================================================
    # Abstract Methods - Subclasses implement these (index-based)
    # =========================================================================

    @abstractmethod
    def _get_value_by_idx(self, idx: int, ctx: Optional[str]) -> Any:
        """Get value from storage by index. Returns None if not exists."""
        pass

    @abstractmethod
    def _set_value_by_idx(self, idx: int, ctx: Optional[str], value: Any) -> None:
        """Set value in storage by index."""
        pass

    @abstractmethod
    def _iter_stored(self):
        """Iterate over (idx, ctx) keys in storage."""
        pass

    # =========================================================================
    # Core Access (with index resolution and ops application)
    # =========================================================================

    def __setitem__(self, key: Tuple[str, str, Optional[str]], value: Any) -> None:
        """Set value: state[node, var, context] = value

        Resolves to canonical index and stores directly.
        """
        if len(key) != 3:
            raise ValueError("Key must be (node, var, context)")

        node, var, ctx = key
        idx = self.schema.get_index(node, var)
        if idx < 0:
            # Variable not in schema - create dynamic index
            # For now, raise error (strict mode)
            raise KeyError(f"Variable ({node}, {var}) not in schema")

        self._set_value_by_idx(idx, ctx, value)

    def __getitem__(self, key: Tuple[str, str, Optional[str]]) -> Any:
        """Get value: state[node, var, context]

        Resolves to canonical index, gets value, and applies any ops.
        """
        if len(key) != 3:
            raise ValueError("Key must be (node, var, context)")

        node, var, ctx = key
        idx, ops = self.schema.resolve_with_ops(node, var)
        if idx < 0:
            return None

        value = self._get_value_by_idx(idx, ctx)

        # Apply ops if any
        if ops and value is not None:
            value = self._apply_ops(value, ops)

        return value

    def _apply_ops(self, value: Any, ops: List) -> Any:
        """Apply recorded operations to a value."""
        # Create a temporary Ref to use its execute logic
        temp_ref = Ref("", "", ops)
        return temp_ref.execute(value)

    def get(self, node: str, var: str, ctx: Optional[str] = None) -> Any:
        """Get value with explicit parameters."""
        return self[node, var, ctx]

    def __contains__(self, key: Tuple[str, str]) -> bool:
        """Check if (node, var) has any value."""
        if len(key) != 2:
            return False
        node, var = key
        idx = self.schema.get_index(node, var)
        if idx < 0:
            return False
        for stored_idx, _ in self._iter_stored():
            if stored_idx == idx:
                return True
        return False

    def __iter__(self):
        """Iterate over (node, var, ctx) keys."""
        # Convert stored (idx, ctx) back to (node, var, ctx)
        for idx, ctx in self._iter_stored():
            canonical = self.schema._idx_to_key.get(idx)
            if canonical:
                yield (canonical[0], canonical[1], ctx)

    # =========================================================================
    # Legacy methods for backward compatibility
    # =========================================================================

    def _get_value(self, node: str, var: str, ctx: Optional[str]) -> Any:
        """Legacy: Get value directly without ops. Use __getitem__ instead."""
        idx = self.schema.get_index(node, var)
        if idx < 0:
            return None
        return self._get_value_by_idx(idx, ctx)

    def _set_value(self, node: str, var: str, ctx: Optional[str], value: Any) -> None:
        """Legacy: Set value directly. Use __setitem__ instead."""
        idx = self.schema.get_index(node, var)
        if idx >= 0:
            self._set_value_by_idx(idx, ctx, value)

    def _iter_keys(self):
        """Legacy: Iterate over (node, var, ctx) keys."""
        return iter(self)

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
        """Debug display of current state values."""
        print(f"\n=== {self.__class__.__name__}: {self.name} ===")

        # Collect all stored (idx, ctx) pairs
        stored = list(self._iter_stored())

        # Group by idx for display
        by_idx: Dict[int, List[str]] = {}
        for idx, ctx in stored:
            by_idx.setdefault(idx, []).append(ctx)

        for idx in sorted(by_idx.keys()):
            canonical = self.schema._idx_to_key.get(idx)
            if not canonical:
                continue

            # Find all vars that map to this idx
            aliases = [k for k, v in self.schema._indexer.items() if v == idx and k != canonical]

            for ctx in by_idx[idx]:
                ctx_str = f"[{ctx}]" if ctx else "[main]"
                value = self._get_value_by_idx(idx, ctx)
                value_str = repr(value)[:50] + "..." if len(repr(value)) > 50 else repr(value)

                # Show canonical location
                print(f"  {canonical[0]}.{canonical[1]}{ctx_str} = {value_str}")

                # Show aliases with their ops
                for alias in aliases:
                    ops = self.schema._ops.get(alias)
                    if ops:
                        # Show what value would be after ops
                        try:
                            resolved_value = self._apply_ops(value, ops)
                            resolved_str = repr(resolved_value)[:30] + "..." if len(repr(resolved_value)) > 30 else repr(resolved_value)
                            print(f"    -> {alias[0]}.{alias[1]} (via ops) = {resolved_str}")
                        except:
                            print(f"    -> {alias[0]}.{alias[1]} (ops: {len(ops)} operations)")
                    else:
                        print(f"    -> {alias[0]}.{alias[1]}")
