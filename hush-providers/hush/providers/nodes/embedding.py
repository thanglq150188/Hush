"""Embedding Node for hush-providers.

This module provides EmbeddingNode that uses ResourceHub to access embedding resources.
It matches the original beeflow implementation using resource_key.
"""

from typing import Dict, Any, Optional, List, Union

from hush.core import BaseNode, WorkflowState
from hush.core.schema import ParamSet
from hush.core.configs import NodeType
from hush.core.registry import ResourceHub


class EmbeddingNode(BaseNode):
    """Embedding node for converting text to vector embeddings in workflows.

    Uses ResourceHub to access embedding resources by resource_key.

    Example:
        ```python
        from hush.core import WorkflowEngine, START, END, INPUT, OUTPUT
        from hush.core.registry import ResourceHub
        from hush.providers import EmbeddingNode, EmbeddingPlugin

        # Setup ResourceHub
        hub = ResourceHub.from_yaml("configs/resources.yaml")
        hub.register_plugin(EmbeddingPlugin)
        ResourceHub.set_instance(hub)

        # Create workflow
        with WorkflowEngine(name="embed") as workflow:
            embed = EmbeddingNode(
                name="embed",
                resource_key="bge-m3",  # References embedding:bge-m3 in resources.yaml
                inputs={"texts": INPUT},
                outputs={"embeddings": OUTPUT}
            )
            START >> embed >> END

        workflow.compile()
        result = await workflow.run(inputs={"texts": ["Hello world", "How are you?"]})
        ```
    """

    __slots__ = ['resource_key']

    type: NodeType = "embedding"

    input_schema: ParamSet = (
        ParamSet.new()
            .var("texts: Union[str, List[str]]", required=True)
            .build()
    )

    output_schema: ParamSet = (
        ParamSet.new()
            .var("embeddings: List[List[float]]", required=True)
            .build()
    )

    def __init__(
        self,
        resource_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize EmbeddingNode.

        Args:
            resource_key: Resource key for embedding model in ResourceHub (e.g., "bge-m3")
            **kwargs: Additional keyword arguments for BaseNode
        """
        super().__init__(**kwargs)

        self.resource_key = resource_key

        # Get embedder from ResourceHub
        hub = ResourceHub.instance()
        embedder = hub.embedding(self.resource_key)
        self.core = embedder.run

    def specific_metadata(self) -> Dict[str, Any]:
        """Return embedding-specific metadata dictionary."""
        return {
            "model": self.resource_key
        }
