"""Hush - Main workflow orchestrator for node execution.

This module provides the Hush class, the primary interface for defining
and executing workflows in the hush framework.

Example:
    ```python
    from hush.core import Hush, START, END
    from hush.core.nodes import PromptNode, LLMNode

    with Hush("my-workflow") as flow:
        prompt = PromptNode(name="prompt", ...)
        llm = LLMNode(name="llm", ...)
        START >> prompt >> llm >> END

    flow.compile()
    result = await flow.run(inputs={"query": "hello"})
    ```
"""

from typing import TYPE_CHECKING, Any, Dict, Optional
import uuid

from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.states import StateSchema
from hush.core.streams import STREAM_SERVICE
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.tracers import BaseTracer
    from hush.core.states import MemoryState


class Hush:
    """Main workflow orchestrator for defining and executing workflows.

    Hush wraps a GraphNode and provides a clean interface for:
    - Defining workflow structure using context manager
    - Compiling the workflow DAG
    - Executing workflows with state management
    - Integrating with tracers for observability

    Attributes:
        name: Workflow name
        description: Workflow description
        graph: The underlying GraphNode

    Example:
        ```python
        # Define workflow
        with Hush("chatbot") as flow:
            prompt = PromptNode(name="prompt", user_prompt="{query}")
            llm = LLMNode(name="llm", resource_key="gpt-4o")
            START >> prompt >> llm >> END

        # Compile and run
        flow.compile()
        result = await flow.run(
            inputs={"query": "Hello!"},
            tracer=langfuse_tracer
        )
        ```
    """

    __slots__ = [
        "name",
        "description",
        "_graph",
        "_schema",
        "_compiled",
    ]

    def __init__(
        self,
        name: str,
        description: str = "",
        **kwargs
    ):
        """Initialize Hush workflow.

        Args:
            name: Workflow name (used for logging and tracing)
            description: Optional workflow description
            **kwargs: Additional arguments passed to GraphNode
        """
        self.name = name
        self.description = description
        self._graph = GraphNode(
            name=name,
            description=description,
            **kwargs
        )
        self._schema: Optional[StateSchema] = None
        self._compiled = False

    def __enter__(self) -> "Hush":
        """Enter context manager for workflow definition."""
        self._graph.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self._graph.__exit__(exc_type, exc_val, exc_tb)

    @property
    def graph(self) -> GraphNode:
        """Access the underlying GraphNode."""
        return self._graph

    @property
    def schema(self) -> Optional[StateSchema]:
        """Access the workflow schema (available after compile)."""
        return self._schema

    @property
    def compiled(self) -> bool:
        """Check if workflow has been compiled."""
        return self._compiled

    def compile(self) -> "Hush":
        """Compile the workflow, building the DAG and state schema.

        Must be called after defining the workflow structure and before
        calling run().

        Returns:
            Self for method chaining

        Raises:
            ValueError: If workflow structure is invalid
        """
        LOGGER.debug(f"Compiling workflow: {self.name}")

        # Build the graph (validates structure, sets up flow types)
        self._graph.build()

        # Create state schema from the graph
        self._schema = StateSchema(
            self._graph,
            name=f"{self.name}:workflow"
        )

        self._compiled = True
        LOGGER.info(f"Workflow '{self.name}' compiled successfully")

        return self

    async def run(
        self,
        inputs: Dict[str, Any],
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        tracer: Optional["BaseTracer"] = None,
    ) -> Dict[str, Any]:
        """Execute the workflow with given inputs.

        Args:
            inputs: Input data for the workflow
            user_id: Optional user identifier (auto-generated if not provided)
            session_id: Optional session identifier (auto-generated if not provided)
            request_id: Optional request identifier (auto-generated if not provided)
            tracer: Optional tracer for observability (e.g., LangfuseTracer)

        Returns:
            Dictionary containing workflow outputs

        Raises:
            RuntimeError: If workflow has not been compiled
        """
        if not self._compiled:
            raise RuntimeError(
                f"Workflow '{self.name}' has not been compiled. "
                "Call compile() before run()."
            )

        # Generate IDs if not provided
        if user_id is None:
            user_id = str(uuid.uuid4())
        if session_id is None:
            session_id = str(uuid.uuid4())
        if request_id is None:
            request_id = str(uuid.uuid4())

        LOGGER.info(
            f"Starting workflow '{self.name}', request_id={request_id}"
        )

        # Create state from schema
        state = self._schema.create_state(
            inputs=inputs,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
        )

        # Execute the graph
        result = await self._graph.run(state)

        # End stream for this request
        await STREAM_SERVICE.end_request(request_id, session_id=session_id)

        # Fire-and-forget flush to tracer in separate process (non-blocking)
        if tracer is not None:
            tracer.flush_in_background(self.name, state)

        LOGGER.info(
            f"Workflow '{self.name}' completed, request_id={request_id}"
        )

        return result

    def get_state(self, request_id: str) -> Optional["MemoryState"]:
        """Get workflow state by request ID.

        Args:
            request_id: The request ID to look up

        Returns:
            MemoryState if found, None otherwise

        Note:
            This requires a state registry implementation.
            Currently returns None as placeholder.
        """
        # TODO: Implement state registry lookup
        return None

    def show(self) -> None:
        """Display workflow structure for debugging."""
        if not self._compiled:
            print(f"Workflow '{self.name}' (not compiled)")
            print("Call compile() first to see full structure.")
            return

        print(f"\n=== Workflow: {self.name} ===")
        if self.description:
            print(f"Description: {self.description}")
        print()
        self._graph.show()
        print()
        self._schema.show()

    def __repr__(self) -> str:
        status = "compiled" if self._compiled else "not compiled"
        return f"<Hush name='{self.name}' {status}>"