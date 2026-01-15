"""Base class cho các iteration node chứa inner graph."""

from typing import Dict, Any, List, TYPE_CHECKING

from hush.core.nodes.base import BaseNode
from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.utils.context import _current_graph

if TYPE_CHECKING:
    from hush.core.states import MemoryState


class Each:
    """Marker wrapper đánh dấu nguồn iteration trong unified inputs.

    Dùng để đánh dấu input nào sẽ được iterate (mỗi iteration nhận một value),
    phân biệt với broadcast inputs (cùng value cho tất cả iterations).

    Example:
        with ForLoopNode(
            inputs={
                "item": Each(items_node["items"]),  # iterate qua list này
                "multiplier": PARENT["multiplier"]   # broadcast cho tất cả iterations
            }
        ) as loop:
            ...

        with AsyncIterNode(
            inputs={
                "chunk": Each(stream_source["stream"]),  # iterate qua async iterable
                "prefix": config_node["prefix"]          # broadcast
            }
        ) as stream:
            ...
    """

    __slots__ = ['source']

    def __init__(self, source: Any):
        """Khởi tạo Each wrapper.

        Args:
            source: Nguồn iterable (list, Ref đến list, async iterable, etc.)
        """
        self.source = source

    def __repr__(self) -> str:
        return f"Each({self.source!r})"


class BaseIterationNode(BaseNode):
    """Abstract base class cho các iteration node chứa inner graph.

    Cung cấp functionality chung cho:
    - ForLoopNode: Iterate qua collection song song
    - WhileLoopNode: Iterate khi condition còn true
    - AsyncIterNode: Xử lý streaming data với output theo thứ tự

    Tất cả iteration nodes chia sẻ:
    - Inner GraphNode chứa loop body
    - Context manager pattern để build inner graph
    - Delegation methods cho add_node/add_edge
    - Utility methods cho tính metrics và inject input

    Subclasses phải implement run() method riêng.
    """

    __slots__ = ['_graph', '_token']

    def __init__(self, **kwargs):
        """Khởi tạo BaseIterationNode với inner graph."""
        super().__init__(**kwargs)
        self._graph: GraphNode = GraphNode(name=BaseNode.INNER_PROCESS)
        self._graph.father = self
        self._token = None

    def __enter__(self):
        """Vào context manager mode - set inner graph làm context hiện tại."""
        self._token = _current_graph.set(self._graph)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Thoát context manager và reset graph context."""
        _current_graph.reset(self._token)

    def add_node(self, node: BaseNode) -> BaseNode:
        """Delegate việc thêm node vào inner graph."""
        return self._graph.add_node(node)

    def add_edge(self, source: str, target: str, type=None):
        """Delegate việc thêm edge vào inner graph."""
        return self._graph.add_edge(source, target, type)

    def build(self):
        """Build inner graph và thực hiện post-initialization."""
        self._graph.build()
        self._post_build()

    def _post_build(self):
        """Hook cho subclasses thực hiện setup bổ sung sau khi inner graph được build.

        Override method này trong subclasses nếu cần.
        """
        pass

    def _calculate_iteration_metrics(self, latencies_ms: List[float]) -> Dict[str, Any]:
        """Tính aggregate metrics từ iteration latencies.

        Args:
            latencies_ms: List các giá trị latency tính bằng milliseconds

        Returns:
            Dictionary chứa các thống kê latency
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
            """Tính giá trị percentile."""
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
        state: 'MemoryState',
        inputs: Dict[str, Any],
        context_id: str
    ) -> None:
        """Inject các giá trị input vào state để inner graph thực thi.

        Args:
            state: Workflow state
            inputs: Dict {var_name: value} cần inject
            context_id: Context ID cho iteration này

        Note: Values được lưu tại cả iteration node và inner graph.
        Điều này cho phép nested iteration nodes truy cập PARENT values.
        """
        inner_graph_name = self._graph.full_name
        for var_name, value in inputs.items():
            state[self.full_name, var_name, context_id] = value
            state[inner_graph_name, var_name, context_id] = value
