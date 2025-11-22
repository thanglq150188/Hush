"""Embedding Node for hush-providers.

This module provides EmbeddingNode that uses ResourceHub to access embedding resources.
It matches the original beeflow implementation using resource_key.
"""

from typing import Dict, Any, Optional, List, Union

from hush.core import BaseNode, WorkflowState
from hush.core.schema import ParamSet
from hush.core.configs import NodeType
from hush.core.registry import ResourceHub, get_hub


class EmbeddingNode(BaseNode):
    """Embedding node for converting text to vector embeddings in workflows.

    Uses ResourceHub to access embedding resources by resource_key.

    Example:
        ```python
        from hush.core import WorkflowEngine, START, END, INPUT, OUTPUT, RESOURCE_HUB
        from hush.providers import EmbeddingNode  # Plugin auto-registers!
        from hush.providers.embeddings.config import EmbeddingConfig

        # Register config (optional - can also use resources.yaml)
        config = EmbeddingConfig(api_type="hf", model="BAAI/bge-m3", dimensions=1024)
        RESOURCE_HUB.register(config, registry_key="embedding:bge-m3", persist=False)

        # Create workflow
        with WorkflowEngine(name="embed") as workflow:
            embed = EmbeddingNode(
                name="embed",
                resource_key="bge-m3",  # Uses global RESOURCE_HUB
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

        # Try to get hub (prefers singleton for backwards compatibility, then global)
        try:
            hub = ResourceHub.instance()
        except RuntimeError:
            # Fall back to global hub if no singleton set
            hub = get_hub()

        embedder = hub.embedding(self.resource_key)
        self.core = embedder.run

    def specific_metadata(self) -> Dict[str, Any]:
        """Return embedding-specific metadata dictionary."""
        return {
            "model": self.resource_key
        }
