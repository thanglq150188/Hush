"""Rerank Node for hush-providers.

This module provides RerankNode that uses ResourceHub to access reranker resources.
It matches the original beeflow implementation using resource_key.
"""

from typing import Dict, Any, Optional, List

from hush.core import BaseNode, WorkflowState
from hush.core.schema import ParamSet
from hush.core.configs import NodeType
from hush.core.registry import ResourceHub, get_hub


class RerankNode(BaseNode):
    """Reranking node for scoring and re-ordering documents in workflows.

    Uses ResourceHub to access reranker resources by resource_key.

    Example:
        ```python
        from hush.core import WorkflowEngine, START, END, INPUT, OUTPUT, RESOURCE_HUB
        from hush.providers import RerankNode  # Plugin auto-registers!
        from hush.providers.rerankers.config import RerankingConfig

        # Register config (optional - can also use resources.yaml)
        config = RerankingConfig(api_type="hf", model="BAAI/bge-reranker-v2-m3")
        RESOURCE_HUB.register(config, registry_key="reranking:bge-m3", persist=False)

        # Create workflow
        with WorkflowEngine(name="rerank") as workflow:
            rerank = RerankNode(
                name="rerank",
                resource_key="bge-m3",  # Uses global RESOURCE_HUB
                inputs={"query": INPUT, "documents": INPUT},
                outputs={"reranks": OUTPUT}
            )
            START >> rerank >> END

        workflow.compile()
        result = await workflow.run(inputs={
            "query": "search query",
            "documents": ["doc1", "doc2", "doc3"],
            "top_k": 2
        })
        ```
    """

    __slots__ = ['resource_key', 'backend']

    type: NodeType = "rerank"

    input_schema: ParamSet = (
        ParamSet.new()
            .var("query: str", required=True)
            .var("documents: List", required=True)
            .var("top_k: int = -1")
            .var("threshold: float = 0.0")
            .build()
    )

    output_schema: ParamSet = (
        ParamSet.new()
            .var("reranks: List[Dict]")
            .build()
    )

    def __init__(
        self,
        resource_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize RerankNode.

        Args:
            resource_key: Resource key for reranker in ResourceHub (e.g., "bge-m3")
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

        self.backend = hub.reranker(self.resource_key)
        self.core = self._process

    async def _process(
        self,
        query: str,
        documents: List,
        top_k: int,
        threshold: float
    ) -> Dict[str, Any]:
        """Process reranking request.

        Args:
            query: Search query
            documents: List of documents (str or dict)
            top_k: Number of top results to return (-1 for all)
            threshold: Minimum score threshold

        Returns:
            Dictionary with reranked documents
        """
        is_dict = False

        if not documents:  # Check if the list is empty
            # Handle empty list case
            return {"reranks": []}

        elif isinstance(documents[0], str):
            # documents is a List[str]
            texts = documents
        elif isinstance(documents[0], dict):
            is_dict = True
            # documents is a List[Dict]
            # Extract text field from each dictionary
            texts = [doc.get('content', '') for doc in documents]
        else:
            # Handle unexpected type
            raise TypeError(f"Expected documents to be List[str] or List[Dict], got list of {type(documents[0]).__name__}")

        results = await self.backend.run(
            query=query,
            texts=texts,
            top_k=top_k if top_k > 0 else len(texts),
            threshold=threshold
        )

        reranked_docs = []

        for r in results:
            if is_dict:
                reranked_docs.append({
                    **documents[r['index']],
                    "score": r['score']
                })
            else:
                reranked_docs.append({
                    "content": documents[r['index']],
                    "score": r['score']
                })

        return {
            "reranks": reranked_docs
        }

    def specific_metadata(self) -> Dict[str, Any]:
        """Return rerank-specific metadata dictionary."""
        return {
            "model": self.resource_key
        }
