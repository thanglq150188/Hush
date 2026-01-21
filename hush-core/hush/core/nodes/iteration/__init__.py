"""Các node iteration cho loop và streaming.

Bao gồm:
- ForLoopNode: Lặp qua collection tuần tự (sequential)
- MapNode: Lặp qua collection song song (parallel)
- WhileLoopNode: Lặp khi điều kiện còn đúng
- AsyncIterNode: Xử lý streaming data theo thứ tự
- Each: Marker wrapper để đánh dấu nguồn iteration
- BaseIterationNode: Base class cho các iteration nodes
"""

from hush.core.nodes.iteration.base import Each, calculate_iteration_metrics, BaseIterationNode
from hush.core.nodes.iteration.for_loop_node import ForLoopNode
from hush.core.nodes.iteration.map_node import MapNode
from hush.core.nodes.iteration.while_loop_node import WhileLoopNode
from hush.core.nodes.iteration.async_iter_node import AsyncIterNode

__all__ = [
    "Each",
    "calculate_iteration_metrics",
    "BaseIterationNode",
    "ForLoopNode",
    "MapNode",
    "WhileLoopNode",
    "AsyncIterNode",
]
