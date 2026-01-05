"""State schema for defining workflow state structure."""

from typing import Any, Dict, Iterator, Optional, Tuple, Type, TYPE_CHECKING

from hush.core.states.ref import Ref

if TYPE_CHECKING:
    from hush.core.states.base import BaseState

__all__ = ["StateSchema"]


class StateSchema:
    """Defines workflow state structure. Built once, shared by all states.

    Schema stores variable values in a single dict where each value is either:
    - A direct value (any type except Ref)
    - A Ref object pointing to another variable (zero-copy redirect)

    Example:
        # From graph node (recommended)
        schema = StateSchema(graph)
        state = schema.create_state(inputs={"query": "hello"})

        # Manual building
        schema = StateSchema()
        schema.link("llm", "messages", "prompt", "output")
        schema.set("llm", "temperature", 0.7)

        # With Redis backend
        state = schema.create_state(
            inputs={"query": "hello"},
            state_class=RedisState,
            redis_client=redis_client
        )

        # Debug
        schema.show()
    """

    __slots__ = ("name", "values")

    def __init__(self, node=None, name: str = None):
        """Initialize schema.

        Args:
            node: Optional node to load connections from (uses duck typing).
                  Can also be a string for backward compatibility (treated as name).
            name: Optional workflow name (inferred from node if not provided)

        Examples:
            StateSchema(graph)           # Load from graph, name = graph.name
            StateSchema(graph, "custom") # Load from graph, name = "custom"
            StateSchema("my_workflow")   # Just name, no node (backward compat)
            StateSchema()                # name = "unnamed"
        """
        self.values: Dict[Tuple[str, str], Any] = {}

        # Handle backward compatibility: StateSchema("name")
        if isinstance(node, str):
            self.name = node
        elif node is not None:
            self.load_from(node)
            self.name = name or getattr(node, 'name', 'unnamed')
        else:
            self.name = name or "unnamed"

    # =========================================================================
    # Core Methods
    # =========================================================================

    def link(
        self,
        node: str,
        var: str,
        source_node: str,
        source_var: Optional[str] = None
    ) -> "StateSchema":
        """Register a variable connection (zero-copy redirect).

        After linking, accessing (node, var) will redirect to (source_node, source_var).

        Args:
            node: Target node name
            var: Target variable name
            source_node: Source node name
            source_var: Source variable name (defaults to var)

        Returns:
            Self for chaining
        """
        self.values[(node, var)] = Ref(source_node, source_var or var)
        return self

    def set(self, node: str, var: str, value: Any) -> "StateSchema":
        """Register a default value for a variable.

        Args:
            node: Node name
            var: Variable name
            value: Default value

        Returns:
            Self for chaining
        """
        self.values[(node, var)] = value
        return self

    def get(self, node: str, var: str) -> Any:
        """Get default value of a variable.

        Args:
            node: Node name
            var: Variable name

        Returns:
            Default value or None if not set.
            Returns None for Ref values (use resolve() for those).
        """
        value = self.values.get((node, var))
        if isinstance(value, Ref):
            return None
        return value

    def is_ref(self, node: str, var: str) -> bool:
        """Check if a variable is a reference."""
        return isinstance(self.values.get((node, var)), Ref)

    def get_ref(self, node: str, var: str) -> Optional[Ref]:
        """Get the Ref object if variable is a reference, else None."""
        value = self.values.get((node, var))
        if isinstance(value, Ref):
            return value
        return None

    def resolve(self, node: str, var: str) -> Tuple[str, str]:
        """Resolve a variable to its source (follow ref chain).

        Args:
            node: Node name
            var: Variable name

        Returns:
            (resolved_node, resolved_var) tuple
        """
        seen = set()
        while True:
            key = (node, var)
            if key in seen:
                # Circular reference, return current
                return key
            seen.add(key)

            value = self.values.get(key)
            if isinstance(value, Ref):
                node, var = value.node, value.var
            else:
                return key

    def load_from(self, node) -> "StateSchema":
        """Load input/output connections from a node and its children as links.

        Uses duck typing - works with any object that has:
        - full_name (str): The node's full hierarchical name
        - inputs (dict): Input connections {var: Ref or literal}
        - outputs (dict): Output connections {var: Ref or literal}
        - input_schema (dict, optional): Schema with Param defaults
        - output_schema (dict, optional): Schema with Param defaults
        - _nodes (dict, optional): Child nodes for recursive loading

        Args:
            node: Node-like object with full_name, inputs, and outputs attributes

        Returns:
            Self for chaining

        Example:
            schema = StateSchema("workflow")
            schema.load_from(graph_node)  # Loads all connections and defaults recursively
        """
        node_name = node.full_name
        inputs = node.inputs or {}
        input_schema = getattr(node, 'input_schema', {}) or {}
        output_schema = getattr(node, 'output_schema', {}) or {}

        # Load this node's input connections as links or defaults
        # inputs={"x": Ref(other_node, "y")} means: node.x -> other_node.y (link)
        # inputs={"x": 0.7} means: node.x = 0.7 (default)
        for var_name, ref in inputs.items():
            if isinstance(ref, Ref):
                # Reference to another node's output
                self.link(node_name, var_name, ref.node, ref.var)
            else:
                # Literal value - store as default
                self.set(node_name, var_name, ref)

        # Load defaults from input_schema for vars not set in inputs
        for var_name, param in input_schema.items():
            if var_name not in inputs and hasattr(param, 'default') and param.default is not None:
                self.set(node_name, var_name, param.default)

        # Load defaults from output_schema (for vars with default values)
        for var_name, param in output_schema.items():
            if hasattr(param, 'default') and param.default is not None:
                self.set(node_name, var_name, param.default)

        # Load this node's output connections as links
        # outputs={"result": Ref(father, "result")} means: father.result -> node.result
        for var_name, ref in (node.outputs or {}).items():
            if isinstance(ref, Ref):
                # Output connection: ref.node.ref.var -> node.var
                self.link(ref.node, ref.var, node_name, var_name)

        # Recursively load children (for graph nodes)
        if hasattr(node, '_nodes') and node._nodes:
            for child in node._nodes.values():
                self.load_from(child)

        # Recursively load inner graph (for iteration nodes like ForLoopNode)
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
        """Create a new state from this schema.

        Args:
            inputs: Initial input values {var_name: value}
            state_class: State backend class (default: MemoryState)
            **kwargs: Additional args passed to state constructor
                - For MemoryState: user_id, session_id, request_id
                - For RedisState: redis_client, prefix, ttl, ...

        Returns:
            New state instance

        Example:
            # In-memory (default)
            state = schema.create_state(inputs={"query": "hello"})

            # Redis backend
            state = schema.create_state(
                inputs={"query": "hello"},
                state_class=RedisState,
                redis_client=redis_client
            )
        """
        if state_class is None:
            from hush.core.states.memory import MemoryState
            state_class = MemoryState

        return state_class(schema=self, inputs=inputs, **kwargs)

    # =========================================================================
    # Collection Interface
    # =========================================================================

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        """Iterate over all (node, var) pairs."""
        return iter(self.values.keys())

    def __contains__(self, key: Tuple[str, str]) -> bool:
        """Check if (node, var) is registered."""
        return key in self.values

    def __len__(self) -> int:
        """Number of registered variables."""
        return len(self.values)

    def __repr__(self) -> str:
        refs = sum(1 for v in self.values.values() if isinstance(v, Ref))
        defaults = len(self.values) - refs
        return f"StateSchema(name='{self.name}', refs={refs}, defaults={defaults})"

    # =========================================================================
    # Debug Methods
    # =========================================================================

    def show(self) -> None:
        """Debug display of schema structure.

        Shows all variables, their links, and default values.
        Useful for debugging workflow dependency issues.
        """
        print(f"\n=== StateSchema: {self.name} ===")

        # Collect all unique variables (including ref targets)
        all_vars = set(self.values.keys())
        for v in self.values.values():
            if isinstance(v, Ref):
                all_vars.add((v.node, v.var))

        ref_count = 0
        for node, var in sorted(all_vars):
            parts = [f"{node}.{var}"]

            value = self.values.get((node, var))
            if isinstance(value, Ref):
                parts.append(f"-> {value.node}.{value.var}")
                ref_count += 1
            elif value is not None:
                value_str = repr(value)[:50] + "..." if len(repr(value)) > 50 else repr(value)
                parts.append(f"= {value_str}")

            print("  " + " ".join(parts))

        print(f"\nTotal: {len(all_vars)} variables, {ref_count} refs")
