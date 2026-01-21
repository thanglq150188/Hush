"""Base class for iteration nodes with common infrastructure."""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
import asyncio

from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.states.ref import Ref
from hush.core.utils.common import Param
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states import MemoryState


class Each:
    """Marker wrapper to mark iteration source in unified inputs.

    Used to mark which input will be iterated (each iteration receives one value),
    distinguishing from broadcast inputs (same value for all iterations).

    Example:
        with ForLoopNode(
            inputs={
                "item": Each(items_node["items"]),  # iterate through this list
                "multiplier": PARENT["multiplier"]   # broadcast to all iterations
            }
        ) as loop:
            ...
    """

    __slots__ = ['source']

    def __init__(self, source: Any):
        self.source = source

    def __repr__(self) -> str:
        return f"Each({self.source!r})"


class BaseIterationNode(GraphNode):
    """Base class for iteration nodes (ForLoop, Map, While, AsyncIter).

    Extracts common boilerplate:
    - _run_graph() for executing child nodes
    - _resolve_values() for resolving Ref objects
    - _build_iteration_data() for building iteration data
    - _normalize_iteration_io() for normalizing inputs/outputs

    Inherits build() from GraphNode which calls _post_build() hook.

    Performance optimization:
    - _var_indices: Pre-computed indices for iteration variables to avoid
      repeated schema.get_index() calls in hot loops
    """

    __slots__ = ['_each', '_broadcast_inputs', '_raw_inputs', '_raw_outputs', '_var_indices']

    def __init__(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize BaseIterationNode.

        Args:
            inputs: Dict mapping variable names to values or Each(source).
        """
        self._raw_outputs = kwargs.get('outputs')
        self._raw_inputs = inputs or {}

        super().__init__(**kwargs)

        # Separate Each() sources from broadcast inputs
        self._each: Dict[str, Any] = {}
        self._broadcast_inputs: Dict[str, Any] = {}

        for var_name, value in self._raw_inputs.items():
            if isinstance(value, Each):
                self._each[var_name] = value.source
            else:
                self._broadcast_inputs[var_name] = value

    def _post_build(self):
        """Setup inputs/outputs after graph is built. Override in subclasses."""
        pass

    async def _run_graph(
        self,
        state: 'MemoryState',
        context_id: str,
        parent_context: str
    ) -> Dict[str, Any]:
        """Run child nodes with parent_context."""
        active_tasks: Dict[str, asyncio.Task] = {}
        ready_count = self.ready_count.copy()
        soft_satisfied: set = set()

        for entry in self.entries:
            task = asyncio.create_task(
                name=entry,
                coro=self._nodes[entry].run(state, context_id, parent_context)
            )
            active_tasks[entry] = task

        while active_tasks:
            done_tasks, _ = await asyncio.wait(
                active_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED
            )

            nodes = self._nodes
            nexts = self.nexts
            edges_lookup = self._edges_lookup

            for task in done_tasks:
                node_name = task.get_name()
                active_tasks.pop(node_name)
                node = nodes[node_name]

                if node.type == "branch":
                    branch_target = node.get_target(state, context_id)
                    from hush.core.nodes.base import END
                    if branch_target != END.name:
                        next_nodes = [branch_target]
                    else:
                        next_nodes = []
                else:
                    next_nodes = nexts[node_name]

                for next_node in next_nodes:
                    edge = edges_lookup.get((node_name, next_node))
                    is_soft = edge and edge.soft

                    if is_soft:
                        if next_node in soft_satisfied:
                            continue
                        soft_satisfied.add(next_node)

                    ready_count[next_node] -= 1

                    if ready_count[next_node] == 0:
                        task = asyncio.create_task(
                            name=next_node,
                            coro=nodes[next_node].run(state, context_id, parent_context)
                        )
                        active_tasks[next_node] = task

        return self.get_outputs(state, context_id, parent_context)

    def _resolve_values(
        self,
        values: Dict[str, Any],
        state: 'MemoryState',
        context_id: Optional[str]
    ) -> Dict[str, Any]:
        """Resolve values, dereference Ref objects."""
        result = {}
        for var_name, value in values.items():
            if isinstance(value, Ref):
                raw = state[value.node, value.var, context_id]
                result[var_name] = value._fn(raw)
            else:
                result[var_name] = value
        return result

    def _build_iteration_data(
        self,
        each_values: Dict[str, List],
        broadcast_values: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build iteration data by zipping `each` values and adding broadcast."""
        if not each_values:
            return [broadcast_values.copy()] if broadcast_values else []

        lengths = {var: len(lst) for var, lst in each_values.items()}
        if len(set(lengths.values())) > 1:
            LOGGER.error(
                "%s [highlight]%s[/highlight]: 'each' variables have different lengths: %s",
                self.__class__.__name__, self.full_name, lengths
            )
            raise ValueError(f"All 'each' variables must have the same length. Got: {lengths}")

        # Pre-allocate result list and avoid repeated broadcast_values.copy()
        keys = list(each_values.keys())
        n = next(iter(lengths.values()))
        result = [{**broadcast_values} for _ in range(n)]

        # Fill in each values
        for i, vals in enumerate(zip(*each_values.values())):
            item = result[i]
            for j, k in enumerate(keys):
                item[k] = vals[j]
        return result

    def _normalize_iteration_io(self):
        """Common pattern for normalizing inputs/outputs in iteration nodes."""
        # Normalize each and broadcast inputs (resolve PARENT refs)
        normalized_each = self._normalize_params(self._each)
        normalized_broadcast = self._normalize_params(self._broadcast_inputs)

        self._each = {k: v.value for k, v in normalized_each.items()}
        self._broadcast_inputs = {k: v.value for k, v in normalized_broadcast.items()}

        # Build inputs
        parsed_inputs = {
            var_name: Param(type=List, required=isinstance(value, Ref), value=value)
            for var_name, value in self._each.items()
        }
        parsed_inputs.update({
            var_name: Param(type=Any, required=isinstance(value, Ref), value=value)
            for var_name, value in self._broadcast_inputs.items()
        })

        # Build outputs from graph outputs
        graph_outputs = self.outputs or {}
        parsed_outputs = {
            key: Param(type=List, required=param.required)
            for key, param in graph_outputs.items()
        }
        parsed_outputs["iteration_metrics"] = Param(type=Dict, required=False)

        # Preserve output mappings set via >> syntax
        existing_outputs = self.outputs or {}

        if self._raw_outputs is not None:
            self.outputs = self._merge_params(parsed_outputs, self._raw_outputs)
        else:
            self.outputs = parsed_outputs

        for key, existing_param in existing_outputs.items():
            if key in self.outputs and existing_param.value is not None:
                self.outputs[key].value = existing_param.value

        self.inputs = parsed_inputs