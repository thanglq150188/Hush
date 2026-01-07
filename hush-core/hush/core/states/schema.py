"""State schema for defining workflow state structure with index-based resolution."""

from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, TYPE_CHECKING

from hush.core.states.ref import Ref

if TYPE_CHECKING:
    from hush.core.states.base import BaseState

__all__ = ["StateSchema"]


class StateSchema:
    """Defines workflow state structure with index-based O(1) resolution.

    Uses Union-Find to flatten reference chains at build time, enabling O(1)
    lookups at runtime regardless of nesting depth.

    Data structures after build:
    - _indexer: {(node, var): idx} - maps variables to storage index
    - _defaults: [value, ...] - default values by index (None if no default)
    - _ops: {(node, var): ops_list} - operations to apply per variable
    - _idx_to_key: {idx: (node, var)} - canonical key for each index
    """

    __slots__ = ("name", "_indexer", "_defaults", "_ops", "_idx_to_key", "_next_idx", "_building", "_build_values")

    def __init__(self, node=None, name: str = None):
        """Initialize schema.

        Args:
            node: Optional node to load connections from (uses duck typing).
            name: Optional workflow name (inferred from node if not provided)
        """
        # Index-based structures
        self._indexer: Dict[Tuple[str, str], int] = {}  # (node, var) -> idx
        self._defaults: List[Any] = []  # idx -> default value (or None)
        self._ops: Dict[Tuple[str, str], List] = {}  # (node, var) -> ops list
        self._idx_to_key: Dict[int, Tuple[str, str]] = {}  # idx -> canonical (node, var)
        self._next_idx: int = 0

        # Temporary during building
        self._building: bool = True
        self._build_values: Dict[Tuple[str, str], Any] = {}  # temp storage during build

        # Handle backward compatibility: StateSchema("name")
        if isinstance(node, str):
            self.name = node
            self._building = False
        elif node is not None:
            self.load_from(node)
            self._build_index()
            self.name = name or getattr(node, 'name', 'unnamed')
            self._building = False
            self._build_values = {}  # Clear after build
        else:
            self.name = name or "unnamed"
            self._building = False

    # =========================================================================
    # Index Building (Union-Find style flattening)
    # =========================================================================

    def _build_index(self) -> None:
        """Build index mappings by flattening all reference chains."""
        # Collect all unique variables
        all_vars = set(self._build_values.keys())
        for v in self._build_values.values():
            if isinstance(v, Ref):
                all_vars.add((v.node, v.var))

        # Process each variable to assign index
        for node, var in all_vars:
            if (node, var) in self._indexer:
                continue

            # Resolve to find the canonical storage location and accumulated ops
            canonical, ops = self._resolve_chain(node, var)

            # Assign or reuse index for canonical location
            if canonical not in self._indexer:
                idx = self._next_idx
                self._next_idx += 1
                self._indexer[canonical] = idx
                self._idx_to_key[idx] = canonical

                # Get default value for canonical location
                default_value = self._build_values.get(canonical)
                if isinstance(default_value, Ref):
                    default_value = None  # Refs don't have defaults
                self._defaults.append(default_value)

            idx = self._indexer[canonical]

            # Map this var to the same index if different from canonical
            if (node, var) != canonical:
                self._indexer[(node, var)] = idx

            # Store ops for vars that have accumulated ops
            if ops:
                self._ops[(node, var)] = ops

    def _resolve_chain(self, node: str, var: str) -> Tuple[Tuple[str, str], List]:
        """Resolve a variable following the reference chain to find canonical storage.

        Returns:
            (canonical_key, accumulated_ops)
        """
        accumulated_ops = []
        visited = set()
        current = (node, var)

        while current not in visited:
            visited.add(current)
            value = self._build_values.get(current)

            if isinstance(value, Ref):
                # Accumulate ops from this Ref
                if value.has_ops:
                    accumulated_ops.extend(value.ops)
                # Follow the reference
                current = (value.node, value.var)
            else:
                # Reached terminal
                break

        return current, accumulated_ops

    # =========================================================================
    # Core Resolution Methods (O(1) at runtime)
    # =========================================================================

    def get_index(self, node: str, var: str) -> int:
        """Get the storage index for a variable. O(1). Returns -1 if not found."""
        return self._indexer.get((node, var), -1)

    def get_ops(self, node: str, var: str) -> Optional[List]:
        """Get operations to apply when reading this variable."""
        return self._ops.get((node, var))

    def get_default(self, idx: int) -> Any:
        """Get default value for an index."""
        if 0 <= idx < len(self._defaults):
            return self._defaults[idx]
        return None

    def resolve(self, node: str, var: str) -> Tuple[str, str]:
        """Resolve a variable to its canonical storage location."""
        idx = self._indexer.get((node, var))
        if idx is not None:
            return self._idx_to_key.get(idx, (node, var))
        return node, var

    def resolve_with_ops(self, node: str, var: str) -> Tuple[int, Optional[List]]:
        """Resolve variable and return index and ops.

        Returns:
            (idx, ops_list_or_none) - idx is -1 if not found
        """
        idx = self._indexer.get((node, var), -1)
        ops = self._ops.get((node, var))
        return idx, ops

    @property
    def num_indices(self) -> int:
        """Number of unique storage indices."""
        return self._next_idx

    # =========================================================================
    # Schema Building Methods
    # =========================================================================

    def link(self, node: str, var: str, source_node: str, source_var: Optional[str] = None) -> "StateSchema":
        """Register a variable connection (zero-copy redirect)."""
        self._build_values[(node, var)] = Ref(source_node, source_var or var)
        return self

    def set(self, node: str, var: str, value: Any) -> "StateSchema":
        """Register a default value for a variable."""
        self._build_values[(node, var)] = value
        return self

    def get(self, node: str, var: str) -> Any:
        """Get default value of a variable."""
        idx = self._indexer.get((node, var), -1)
        if idx >= 0:
            return self._defaults[idx]
        return None

    def is_ref(self, node: str, var: str) -> bool:
        """Check if a variable is a reference (has different canonical location)."""
        idx = self._indexer.get((node, var), -1)
        if idx >= 0:
            canonical = self._idx_to_key.get(idx)
            return canonical != (node, var)
        return False

    def get_ref(self, node: str, var: str) -> Optional[Ref]:
        """Get the Ref object if variable is a reference, else None.

        Note: After build, we reconstruct from indexer. Returns simple Ref without ops.
        """
        idx = self._indexer.get((node, var), -1)
        if idx >= 0:
            canonical = self._idx_to_key.get(idx)
            if canonical != (node, var):
                return Ref(canonical[0], canonical[1])
        return None

    # For backward compatibility
    @property
    def values(self) -> Dict[Tuple[str, str], Any]:
        """Backward compatibility: reconstruct values dict from index structures."""
        result = {}
        for key, idx in self._indexer.items():
            canonical = self._idx_to_key.get(idx)
            if canonical == key:
                # This is the canonical location - return default value
                result[key] = self._defaults[idx]
            else:
                # This is a reference - return Ref to canonical
                result[key] = Ref(canonical[0], canonical[1])
        return result

    def load_from(self, node) -> "StateSchema":
        """Load input/output connections from a node and its children."""
        node_name = node.full_name
        inputs = node.inputs or {}
        input_schema = getattr(node, 'input_schema', {}) or {}
        output_schema = getattr(node, 'output_schema', {}) or {}

        # Load this node's input connections as links or defaults
        for var_name, ref in inputs.items():
            if isinstance(ref, Ref):
                self._build_values[(node_name, var_name)] = ref
            else:
                self._build_values[(node_name, var_name)] = ref

        # Load defaults from input_schema for vars not set in inputs
        for var_name, param in input_schema.items():
            if var_name not in inputs and hasattr(param, 'default') and param.default is not None:
                self._build_values[(node_name, var_name)] = param.default

        # Load defaults from output_schema
        for var_name, param in output_schema.items():
            if hasattr(param, 'default') and param.default is not None:
                self._build_values[(node_name, var_name)] = param.default

        # Load output connections as links
        for var_name, ref in (node.outputs or {}).items():
            if isinstance(ref, Ref):
                # Output: ref.node.ref.var -> node.var (reverse direction)
                self._build_values[(ref.node, ref.var)] = Ref(node_name, var_name, ref.ops if ref.has_ops else None)

        # Register metadata variables for this node (start_time, end_time, error)
        for meta_var in ("start_time", "end_time", "error"):
            self._build_values[(node_name, meta_var)] = None

        # Recursively load children
        if hasattr(node, '_nodes') and node._nodes:
            for child in node._nodes.values():
                self.load_from(child)

        # Recursively load inner graph (for iteration nodes)
        if hasattr(node, '_graph') and node._graph:
            self.load_from(node._graph)

        return self

    # =========================================================================
    # State Creation
    # =========================================================================

    def create_state(
        self,
        inputs: Dict[str, Any] = None,
        state_class: Type["BaseState"] = None,
        **kwargs
    ) -> "BaseState":
        """Create a new state from this schema."""
        if state_class is None:
            from hush.core.states.memory import MemoryState
            state_class = MemoryState

        return state_class(schema=self, inputs=inputs, **kwargs)

    # =========================================================================
    # Collection Interface
    # =========================================================================

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        """Iterate over all (node, var) pairs."""
        return iter(self._indexer.keys())

    def __contains__(self, key: Tuple[str, str]) -> bool:
        """Check if (node, var) is registered."""
        return key in self._indexer

    def __len__(self) -> int:
        """Number of registered variables."""
        return len(self._indexer)

    def __repr__(self) -> str:
        refs = sum(1 for k in self._indexer if self._idx_to_key.get(self._indexer[k]) != k)
        return f"StateSchema(name='{self.name}', vars={len(self._indexer)}, refs={refs}, indices={self._next_idx})"

    # =========================================================================
    # Debug Methods
    # =========================================================================

    def show(self) -> None:
        """Debug display of schema structure."""
        print(f"\n=== StateSchema: {self.name} ===")

        ref_count = 0
        for node, var in sorted(self._indexer.keys()):
            parts = [f"{node}.{var}"]

            idx = self._indexer[(node, var)]
            parts.append(f"[idx={idx}]")

            canonical = self._idx_to_key.get(idx)
            if canonical != (node, var):
                parts.append(f"-> {canonical[0]}.{canonical[1]}")
                ref_count += 1
            else:
                default = self._defaults[idx]
                if default is not None:
                    value_str = repr(default)[:50] + "..." if len(repr(default)) > 50 else repr(default)
                    parts.append(f"= {value_str}")

            ops = self._ops.get((node, var))
            if ops:
                ops_str = str(ops)[:40] + "..." if len(str(ops)) > 40 else str(ops)
                parts.append(f"ops={ops_str}")

            print("  " + " ".join(parts))

        print(f"\nTotal: {len(self._indexer)} variables, {ref_count} refs, {self._next_idx} unique indices")
