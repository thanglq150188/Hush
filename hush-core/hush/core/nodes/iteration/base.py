"""Base class for iteration nodes that contain an inner graph."""

from abc import abstractmethod
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime
import traceback

from hush.core.nodes.base import BaseNode
from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.utils.context import _current_graph
from hush.core.loggings import LOGGER

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
        """
        for var_name, value in inputs.items():
            state[self._graph.full_name, var_name, context_id] = value

    @abstractmethod
    async def _execute_loop(
        self,
        state: 'BaseState',
        inputs: Dict[str, Any],
        request_id: str,
        context_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Execute the loop logic. Must be implemented by subclasses.

        Args:
            state: The workflow state
            inputs: The input values for this node
            request_id: The request ID for logging
            context_id: The context ID

        Returns:
            The outputs from the loop execution
        """
        pass

    async def run(
        self,
        state: 'BaseState',
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the iteration node with common setup/teardown.

        This method handles:
        - Recording execution
        - Getting inputs
        - Logging
        - Error handling
        - Metrics recording
        """
        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        _outputs = {}

        try:
            _inputs = self.get_inputs(state, context_id)

            if self.verbose:
                LOGGER.info(
                    "request[%s] - NODE: %s[%s] (%s) inputs=%s",
                    request_id, self.name, context_id,
                    str(self.type).upper(), str(_inputs)[:200]
                )

            _outputs = await self._execute_loop(state, _inputs, request_id, context_id)

            if self.verbose:
                LOGGER.info(
                    "request[%s] - NODE: %s[%s] (%s) outputs=%s",
                    request_id, self.name, context_id,
                    str(self.type).upper(), str(_outputs)[:200]
                )

            self.store_result(state, _outputs, context_id)

        except Exception as e:
            error_msg = traceback.format_exc()
            LOGGER.error(f"Error in node {self.full_name}: {str(e)}")
            LOGGER.error(error_msg)
            state[self.full_name, "error", context_id] = error_msg

        finally:
            end_time = datetime.now()
            state[self.full_name, "start_time", context_id] = start_time
            state[self.full_name, "end_time", context_id] = end_time
            return _outputs
