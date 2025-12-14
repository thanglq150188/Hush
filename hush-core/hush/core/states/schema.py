"""State schema for defining workflow state structure."""

from typing import Any, Dict, Iterator, Optional, Tuple, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from hush.core.states.base import BaseState

__all__ = ["StateSchema"]


class StateSchema:
    """Defines workflow state structure. Built once, shared by all states.

    Schema stores:
    - links: variable connections (zero-copy redirects)
    - defaults: default values for variables

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

    __slots__ = ("name", "links", "defaults")

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
        self.links: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self.defaults: Dict[Tuple[str, str], Any] = {}

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
        self.links[(node, var)] = (source_node, source_var or var)
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
        self.defaults[(node, var)] = value
        return self

    def get(self, node: str, var: str) -> Any:
        """Get default value of a variable.

        Args:
            node: Node name
            var: Variable name

        Returns:
            Default value or None if not set
        """
        return self.defaults.get((node, var))

    def resolve(self, node: str, var: str) -> Tuple[str, str]:
        """Resolve a variable to its source (follow links).

        Args:
            node: Node name
            var: Variable name

        Returns:
            (resolved_node, resolved_var) tuple
        """
        return self.links.get((node, var), (node, var))

    def load_from(self, node) -> "StateSchema":
        """Load input connections from a node and its children as links.

        Uses duck typing - works with any object that has:
        - full_name (str): The node's full hierarchical name
        - inputs (dict): Input connections {var: {source_node: source_var} or literal}
        - _nodes (dict, optional): Child nodes for recursive loading

        Args:
            node: Node-like object with full_name and inputs attributes

        Returns:
            Self for chaining

        Example:
            schema = StateSchema("workflow")
            schema.load_from(graph_node)  # Loads all connections recursively
        """
        node_name = node.full_name

        # Load this node's input connections as links
        for var_name, ref in (node.inputs or {}).items():
            if isinstance(ref, dict) and ref:
                ref_node, ref_var = next(iter(ref.items()))
                if hasattr(ref_node, 'full_name'):
                    self.link(node_name, var_name, ref_node.full_name, ref_var)

        # Recursively load children (for graph nodes)
        if hasattr(node, '_nodes') and node._nodes:
            for child in node._nodes.values():
                self.load_from(child)

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
        seen = set()
        for key in self.links:
            if key not in seen:
                seen.add(key)
                yield key
        for key in self.defaults:
            if key not in seen:
                seen.add(key)
                yield key

    def __contains__(self, key: Tuple[str, str]) -> bool:
        """Check if (node, var) is registered."""
        return key in self.links or key in self.defaults

    def __len__(self) -> int:
        """Number of registered variables."""
        all_vars = set(self.links.keys()) | set(self.defaults.keys())
        return len(all_vars)

    def __repr__(self) -> str:
        return f"StateSchema(name='{self.name}', links={len(self.links)}, defaults={len(self.defaults)})"

    # =========================================================================
    # Debug Methods
    # =========================================================================

    def show(self) -> None:
        """Debug display of schema structure.

        Shows all variables, their links, and default values.
        Useful for debugging workflow dependency issues.
        """
        print(f"\n=== StateSchema: {self.name} ===")

        # Collect all unique variables
        all_vars = set(self.links.keys()) | set(self.defaults.keys())

        # Also include link targets
        for source_node, source_var in self.links.values():
            all_vars.add((source_node, source_var))

        for node, var in sorted(all_vars):
            parts = [f"{node}.{var}"]

            # Check if this is a link source
            if (node, var) in self.links:
                target = self.links[(node, var)]
                parts.append(f"-> {target[0]}.{target[1]}")

            # Check if has default
            if (node, var) in self.defaults:
                default = self.defaults[(node, var)]
                default_str = repr(default)[:50] + "..." if len(repr(default)) > 50 else repr(default)
                parts.append(f"= {default_str}")

            print("  " + " ".join(parts))

        print(f"\nTotal: {len(all_vars)} variables, {len(self.links)} links")

    def show_links(self) -> None:
        """Show only link mappings."""
        print(f"\n=== Links in {self.name} ===")
        for (node, var), (src_node, src_var) in self.links.items():
            print(f"  {node}.{var} -> {src_node}.{src_var}")
        print(f"Total: {len(self.links)} links")

    def show_defaults(self) -> None:
        """Show only default values."""
        print(f"\n=== Defaults in {self.name} ===")
        for (node, var), value in self.defaults.items():
            value_str = repr(value)[:50] + "..." if len(repr(value)) > 50 else repr(value)
            print(f"  {node}.{var} = {value_str}")
        print(f"Total: {len(self.defaults)} defaults")
