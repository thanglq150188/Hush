"""Core nodes cho hush workflow engine.

Bao gồm:
- BaseNode: Base class cho tất cả workflow nodes
- DummyNode: Node placeholder cho START/END markers
- GraphNode: Container quản lý subgraph với thực thi song song
- BranchNode: Conditional routing với precompiled conditions
- ForLoopNode: Iterate qua collection song song
- WhileLoopNode: Iterate khi condition còn true
- AsyncIterNode: Xử lý streaming data với ordered output
- CodeNode: Thực thi Python functions
- ParserNode: Extract structured data từ text
"""

from .base import (
    BaseNode,
    DummyNode,
    NodeType,
    START,
    END,
    PARENT,
)
from .graph.graph_node import GraphNode
from .flow.branch_node import BranchNode
from .iteration.for_loop_node import ForLoopNode
from .iteration.while_loop_node import WhileLoopNode
from .iteration.async_iter_node import AsyncIterNode
from .transform.code_node import CodeNode, code_node
from .transform.parser_node import ParserNode, ParserType

__all__ = [
    # Base
    "BaseNode",
    "DummyNode",
    "NodeType",
    # Markers
    "START",
    "END",
    "PARENT",
    # Graph
    "GraphNode",
    # Flow control
    "BranchNode",
    "ForLoopNode",
    "WhileLoopNode",
    "AsyncIterNode",
    # Transform
    "CodeNode",
    "code_node",
    "ParserNode",
    "ParserType",
]
