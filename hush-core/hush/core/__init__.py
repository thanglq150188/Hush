"""
Hush Core - Workflow Engine

A powerful async workflow orchestration framework.

Example:
    ```python
    from hush.core import WorkflowEngine, START, END

    with WorkflowEngine(name="my-workflow") as workflow:
        # Define your nodes and flow here
        START >> your_node >> END

    workflow.compile()
    result = await workflow.run(inputs={"key": "value"})
    ```
"""

from hush.core.workflow import WorkflowEngine
from hush.core.nodes import (
    BaseNode,
    DummyNode,
    GraphNode,
    BranchNode,
    ForLoopNode,
    WhileLoopNode,
    StreamNode,
    CodeNode,
    code_node,
    LambdaNode,
    ParserNode,
    ParserType,
    START,
    END,
    CONTINUE,
    INPUT,
    OUTPUT,
)
from hush.core.states import (
    WorkflowState,
    WorkflowIndexer,
    StateRegistry,
    STATE_REGISTRY,
)
from hush.core.configs import (
    NodeConfig,
    NodeType,
    EdgeConfig,
    EdgeType,
)
from hush.core.schema import (
    Param,
    ParamSet,
)
from hush.core.streams import (
    create_streaming_service,
    STREAM_SERVICE,
)
from hush.core.loggings import LOGGER
from hush.core.registry import (
    ResourceHub,
    ResourcePlugin,
    ResourceConfig,
    ConfigStorage,
    FileConfigStorage,
)

__version__ = "0.1.0"

__all__ = [
    # Main engine
    "WorkflowEngine",
    # Nodes
    "BaseNode",
    "DummyNode",
    "GraphNode",
    "BranchNode",
    "ForLoopNode",
    "WhileLoopNode",
    "StreamNode",
    "CodeNode",
    "code_node",
    "LambdaNode",
    "ParserNode",
    "ParserType",
    # Markers
    "START",
    "END",
    "CONTINUE",
    "INPUT",
    "OUTPUT",
    # State
    "WorkflowState",
    "WorkflowIndexer",
    "StateRegistry",
    "STATE_REGISTRY",
    # Config
    "NodeConfig",
    "NodeType",
    "EdgeConfig",
    "EdgeType",
    # Schema
    "Param",
    "ParamSet",
    # Streams
    "create_streaming_service",
    "STREAM_SERVICE",
    # Logging
    "LOGGER",
    # Registry
    "ResourceHub",
    "ResourcePlugin",
    "ResourceConfig",
    "ConfigStorage",
    "FileConfigStorage",
]
