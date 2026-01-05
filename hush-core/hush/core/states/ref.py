"""Reference type for zero-copy variable linking."""

from typing import Union, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from hush.core.nodes.base import BaseNode

__all__ = ["Ref"]


class Ref:
    """Reference to another variable (zero-copy link).

    Stores a reference to a node's variable. The node can be either:
    - A BaseNode object (during graph building)
    - A string full_name (after serialization or in schema)

    The `node` property always returns the string full_name.

    Example:
        # During graph building
        ref = Ref(some_node, "output")
        ref.node  # Returns "graph.some_node" (full_name)

        # In schema
        ref = Ref("graph.some_node", "output")
        ref.node  # Returns "graph.some_node"

        # Usage in node connections
        node_b = CodeNode(inputs={"x": Ref(node_a, "result")})
    """

    __slots__ = ("_node", "var")

    def __init__(self, node: Union["BaseNode", str], var: str) -> None:
        """Create a reference to another variable.

        Args:
            node: Source node (BaseNode object or full_name string)
            var: Source variable name
        """
        self._node = node
        self.var = var

    @property
    def node(self) -> str:
        """Get the node's full_name (always returns string)."""
        if hasattr(self._node, 'full_name'):
            return self._node.full_name
        return self._node

    @property
    def raw_node(self) -> Union["BaseNode", str]:
        """Get the raw node reference (BaseNode or str)."""
        return self._node

    def as_tuple(self) -> Tuple[str, str]:
        """Return as (node_full_name, var) tuple."""
        return (self.node, self.var)

    def __repr__(self) -> str:
        return f"Ref({self.node!r}, {self.var!r})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Ref):
            return False
        return self.node == other.node and self.var == other.var

    def __hash__(self) -> int:
        return hash((self.node, self.var))
