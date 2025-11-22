"""Workflow indexer for managing node dependencies and variable mapping."""

from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING

from hush.core.utils.common import verify_data
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.nodes.base import BaseNode



class WorkflowIndexer:
    """
    Workflow indexer for managing node dependencies and variable mapping.

    Creates a high-performance index mapping system for nodes in the workflow,
    supporting zero-copy variable sharing and lazy index allocation.
    """

    __slots__ = ["_indices", "_values", "_name", "_nodes"]

    def __init__(self, name: str) -> None:
        """Initialize WorkflowIndexer for a specific workflow."""
        self._name = name
        self._indices: Dict[Tuple[str, str], Optional[int]] = {}
        self._values: List[Any] = []
        self._nodes: Dict[str, 'BaseNode'] = {}

    def __iter__(self):
        """Iterator over all (node, variable) mappings."""
        for (node, var) in self._indices:
            yield (node, var)

    def __getitem__(self, key: Tuple[str, str]) -> int:
        """Get workflow index of a node variable."""
        if key in self:
            node, var = key
            index = self._indices[node, var]
            return index

        raise ValueError(f"{key} not mapped in workflow indexer: {self._name}")

    def __contains__(self, key: Tuple[str, str]) -> bool:
        """Check if node variable is mapped in workflow."""
        return key in self._indices

    def __len__(self) -> int:
        """Number of variable slots allocated."""
        return len(self._values)

    def __repr__(self) -> str:
        return f"WorkflowIndexer(workflow='{self._name}', variables={len(self._values)})"

    def _register(self, node: str, var: str) -> None:
        """Register node variable without allocating index (lazy registration)."""
        if (node, var) not in self:
            self._indices[node, var] = None

    def _allocate(self, node: str, var: str) -> int:
        """Allocate workflow index for node variable."""
        self._register(node, var)

        current_index = self._indices[node, var]

        if current_index is None:
            index = len(self._values)
            self._indices[node, var] = index
            self._values.append(None)
            return index

        return current_index

    def _link(self,
              node: str, var: str,
              ref_node: str, ref_var: Optional[str] = None) -> None:
        """Create dependency link between nodes (zero-copy variable sharing)."""
        if ref_var is None:
            self._indices[node, var] = self._allocate(ref_node, var)
        else:
            self._indices[node, var] = self._allocate(ref_node, ref_var)

    def put(self, node: str, var: str, value: Any) -> None:
        """Set value for node variable."""
        if (node, var) not in self:
            raise ValueError(f"{(node, var)} not registered in workflow: {self._name}")

        verify_data(value)

        index = self._allocate(node, var)
        if value is None and self._values[index] is not None:
            return
        self._values[index] = value

    def get(self, node: str, var: str) -> Any:
        """Get value of node variable."""
        index = self._indices.get((node, var), None)
        if index is None:
            return None
        return self._values[index]

    def resolve(self, node: str, var: str, value: Any, default_value: Any) -> None:
        """Resolve dependency of node variable in workflow context."""
        from hush.core.nodes.base import INPUT, OUTPUT

        base_node = self._nodes[node]

        if hasattr(value, "is_base_node"):
            ref_name = value.full_name
            if value in [INPUT, OUTPUT]:
                if base_node.father:
                    ref_name = base_node.father.full_name
                else:
                    ref_name = self._name
            self._link(node, var, ref_name)

        elif isinstance(value, dict) and value:
            ref, ref_var = next(iter(value.items()))
            if hasattr(ref, "is_base_node"):
                ref_name = ref.full_name
                if ref in [INPUT, OUTPUT]:
                    if base_node.father:
                        ref_name = base_node.father.full_name
                    else:
                        ref_name = self._name
                self._link(node, var, ref_name, ref_var)
            else:
                self.put(node, var, value)
        else:
            self.put(node, var, value)

        _idx = self[node, var]
        if self._values[_idx] is None:
            self._values[_idx] = default_value

    def add_node(self, node: 'BaseNode') -> 'WorkflowIndexer':
        """Add node to workflow and register variable mappings."""
        node_name = node.full_name
        self._nodes[node_name] = node

        if node.input_schema:
            for var_name in node.input_schema:
                self._register(node_name, var_name)

        if node.output_schema:
            for var_name in node.output_schema:
                self._register(node_name, var_name)

        metrics = ["start_time", "end_time", "error"]
        if node.type == "llm":
            metrics.extend(["completion_start_time", "usage", "token_count", "model"])

        for metric in metrics:
            self._allocate(node_name, metric)

        if hasattr(node, "_nodes"):
            for _, sub_node in node._nodes.items():
                self.add_node(sub_node)

        if hasattr(node, "_graph"):
            self.add_node(node._graph)

        if hasattr(node, "_tools"):
            for _, tool_node in node._tools.items():
                self.add_node(tool_node)

        return self

    def build(self) -> 'WorkflowIndexer':
        """Build workflow index mappings and resolve all node dependencies."""
        for node_name, node in self._nodes.items():
            for schema, connections in [
                (node.input_schema, node.inputs),
                (node.output_schema, node.outputs)
            ]:
                if schema:
                    for var_name in schema:
                        default_value = schema[var_name].default
                        value = connections.get(var_name)
                        self.resolve(node_name, var_name, value, default_value)

            if node.input_schema:
                for var_name in node.input_schema:
                    node.input_indexes[var_name] = self[node_name, var_name]

            if node.output_schema:
                for var_name in node.output_schema:
                    node.output_indexes[var_name] = self[node_name, var_name]

            common_metrics = ["start_time", "end_time", "error"]
            for metric in common_metrics:
                node.metrics[metric] = self[node_name, metric]

            if node.type == "llm":
                llm_metrics = ["completion_start_time", "usage"]
                for metric in llm_metrics:
                    node.metrics[metric] = self[node_name, metric]

        return self

    def show(self):
        """Debug display of workflow index mappings."""
        print(f"\n=== Workflow Indexer: {self._name} ===")
        for (node, var) in self:
            try:
                idx = self[node, var]
                value = self.get(node, var)
                print(f"{node}.{var} → index[{idx}] = {value}")
            except ValueError as e:
                print(f"{node}.{var} → {e}")
        print(f"Total workflow variables: {len(self._values)}\n")
