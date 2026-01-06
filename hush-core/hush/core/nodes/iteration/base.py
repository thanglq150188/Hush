"""Base class for iteration nodes that contain an inner graph."""

from typing import Dict, Any, List, TYPE_CHECKING

from hush.core.nodes.base import BaseNode
from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.utils.context import _current_graph

if TYPE_CHECKING:
    from hush.core.states import BaseState


class IterationNode(BaseNode):
    """
    Base class for iteration nodes that contain an inner graph.

    Provides common functionality for:
    - ForLoopNode: Iterates over a collection concurrently
    - WhileLoopNode: Iterates while a condition is true
    - StreamNode: Processes streaming data with ordered output

    All iteration nodes share:
    - An inner GraphNode that holds the loop body
    - Context manager pattern for building the inner graph
    - Delegation methods for add_node/add_edge
    - Utility methods for metrics calculation and input injection

    Subclasses must implement their own run() method.
    """

    __slots__ = ['_graph', '_token']

    def __init__(self, **kwargs):
        """Initialize an IterationNode with an inner graph."""
        super().__init__(**kwargs)
        self._graph: GraphNode = GraphNode(name=BaseNode.INNER_PROCESS)
        self._graph.father = self
        self._token = None

    def __enter__(self):
        """Enter context manager mode - set inner graph as current context."""
        self._token = _current_graph.set(self._graph)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and reset graph context."""
        _current_graph.reset(self._token)

    def add_node(self, node: BaseNode) -> BaseNode:
        """Delegate node addition to inner graph."""
        return self._graph.add_node(node)

    def add_edge(self, source: str, target: str, type=None):
        """Delegate edge addition to inner graph."""
        return self._graph.add_edge(source, target, type)

    def build(self):
        """Build the inner graph and perform post-initialization."""
        self._graph.build()
        self._post_build()

    def _post_build(self):
        """
        Hook for subclasses to perform additional setup after inner graph is built.
        Override this method in subclasses if needed.
        """
        pass

    def _calculate_iteration_metrics(self, latencies_ms: List[float]) -> Dict[str, Any]:
        """Calculate aggregate metrics from iteration latencies.

        Args:
            latencies_ms: List of latency values in milliseconds

        Returns:
            Dictionary with latency statistics
        """
        if not latencies_ms:
            return {
                "latency_avg_ms": 0.0,
                "latency_min_ms": 0.0,
                "latency_max_ms": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
            }

        sorted_latencies = sorted(latencies_ms)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            """Calculate percentile value."""
            if n == 1:
                return sorted_latencies[0]
            k = (n - 1) * p
            f = int(k)
            c = f + 1 if f + 1 < n else f
            return sorted_latencies[f] + (k - f) * (sorted_latencies[c] - sorted_latencies[f])

        return {
            "latency_avg_ms": round(sum(latencies_ms) / n, 2),
            "latency_min_ms": round(sorted_latencies[0], 2),
            "latency_max_ms": round(sorted_latencies[-1], 2),
            "latency_p50_ms": round(percentile(0.50), 2),
            "latency_p95_ms": round(percentile(0.95), 2),
            "latency_p99_ms": round(percentile(0.99), 2),
        }

    def inject_inputs(
        self,
        state: 'BaseState',
        inputs: Dict[str, Any],
        context_id: str
    ) -> None:
        """Inject input values into state for inner graph execution.

        Args:
            state: The workflow state
            inputs: Dict of {var_name: value} to inject
            context_id: The context ID for this iteration

        Note: Stores at BOTH the iteration node's location (for PARENT refs)
        AND the inner graph's location (for direct graph input refs).
        This allows inner nodes using PARENT["var"] to find the values.
        """
        for var_name, value in inputs.items():
            # Store at iteration node's location (for PARENT["var"] access)
            state._set_value(self.full_name, var_name, context_id, value)
            # Also store at inner graph's location (for backward compatibility)
            state._set_value(self._graph.full_name, var_name, context_id, value)
