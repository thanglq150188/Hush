"""
Hush Core - Workflow Engine

A powerful async workflow orchestration framework.

Example:
    ```python
    from hush.core import Hush, START, END

    with Hush("my-workflow") as flow:
        # Define your nodes and flow here
        START >> your_node >> END

    flow.compile()
    result = await flow.run(inputs={"key": "value"})
    ```
"""

from hush.core.workflow import Hush
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
    MemoryState,
    Ref,
    Cell,
)
from hush.core.configs import (
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
from hush.core.tracers import (
    BaseTracer,
    register_tracer,
    get_registered_tracers,
    MEDIA_KEY,
    MediaAttachment,
    serialize_media_attachments,
)

__version__ = "0.1.0"

__all__ = [
    # Main engine
    "Hush",
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
    # Tracers
    "BaseTracer",
    "register_tracer",
    "get_registered_tracers",
    "MEDIA_KEY",
    "MediaAttachment",
    "serialize_media_attachments",
]
