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

# TODO: Uncomment when workflow.py is refactored
# from hush.core.workflow import WorkflowEngine
from hush.core.nodes import (
    BaseNode,
    DummyNode,
    GraphNode,
    BranchNode,
    ForLoopNode,
    WhileLoopNode,
    AsyncIterNode,
    CodeNode,
    code_node,
    ParserNode,
    ParserType,
    START,
    END,
    PARENT,
)
from hush.core.states import (
    StateSchema,
    BaseState,
    MemoryState,
    RedisState,
)
from hush.core.configs import (
    NodeConfig,
    NodeType,
    EdgeConfig,
    EdgeType,
)
from hush.core.utils import Param
from hush.core.streams import (
    STREAM_SERVICE,
)
from hush.core.loggings import LOGGER
from hush.core.registry import (
    ResourceHub,
    RESOURCE_HUB,
    get_hub,
    set_global_hub,
    ResourceFactory,
    register_config_class,
    register_config_classes,
    register_factory_handler,
    ConfigStorage,
    YamlConfigStorage,
    JsonConfigStorage,
)

__version__ = "0.1.0"

__all__ = [
    # Main engine
    # "WorkflowEngine",  # TODO: Uncomment when refactored
    # Nodes
    "BaseNode",
    "DummyNode",
    "GraphNode",
    "BranchNode",
    "ForLoopNode",
    "WhileLoopNode",
    "AsyncIterNode",
    "CodeNode",
    "code_node",
    "ParserNode",
    "ParserType",
    # Markers
    "START",
    "END",
    "PARENT",
    # State
    "StateSchema",
    "BaseState",
    "MemoryState",
    "RedisState",
    # Config
    "NodeConfig",
    "NodeType",
    "EdgeConfig",
    "EdgeType",
    # Schema
    "Param",
    # Streams
    "STREAM_SERVICE",
    # Logging
    "LOGGER",
    # Registry
    "ResourceHub",
    "RESOURCE_HUB",
    "get_hub",
    "set_global_hub",
    "ResourceFactory",
    "register_config_class",
    "register_config_classes",
    "register_factory_handler",
    "ConfigStorage",
    "YamlConfigStorage",
    "JsonConfigStorage",
]
