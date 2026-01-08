"""Workflow engine for orchestrating node execution."""

# TODO: Refactor to use new StateSchema design
# from typing import Dict, Any
# import uuid
# import asyncio

# from hush.core.nodes.graph.graph_node import GraphNode
# from hush.core.nodes.base import BaseNode
# from hush.core.states import MemoryState, MemoryState
# from hush.core.loggings import LOGGER



# class WorkflowEngine:
#     """
#     Main workflow engine for defining and executing workflows.

#     Example:
#         ```python
#         with WorkflowEngine(name="my-workflow") as workflow:
#             node1 = SomeNode(name="step1", ...)
#             node2 = SomeNode(name="step2", ...)
#             START >> node1 >> node2 >> END

#         workflow.compile()
#         result = await workflow.run(inputs={"query": "hello"})
#         ```
#     """

#     def __init__(
#         self,
#         name: str,
#         description: str = "",
#         **kwargs
#     ):
#         """
#         Initialize WorkflowEngine.

#         Args:
#             name: Workflow name
#             description: Workflow description
#             **kwargs: Additional arguments passed to GraphNode
#         """
#         self.name = f"{name}:workflow"
#         self.graph_name = name
#         self._graph = GraphNode(
#             name=self.graph_name,
#             description=description,
#             **kwargs
#         )
#         self.indexer = None

#     def __enter__(self):
#         self._graph.__enter__()
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self._graph.__exit__(exc_type, exc_val, exc_tb)

#     def compile(self):
#         """Compile the workflow, building the DAG and indexer."""
#         from hush.core.nodes.base import INPUT

#         self._graph.build()

#         for var in self._graph.get_input_variables():
#             self._graph.inputs[var] = {INPUT: var}

#         self.indexer = WorkflowIndexer(name=self.name).add_node(self._graph).build()

#     async def run(
#         self,
#         inputs: Dict[str, Any],
#         user_id: str = None,
#         session_id: str = None,
#         request_id: str = None
#     ) -> Dict[str, Any]:
#         """
#         Execute the workflow with given inputs.

#         Args:
#             inputs: Input data for the workflow
#             user_id: Optional user identifier
#             session_id: Optional session identifier
#             request_id: Optional request identifier

#         Returns:
#             Dictionary containing workflow outputs
#         """
#         if user_id is None:
#             user_id = str(uuid.uuid4())
#         if session_id is None:
#             session_id = str(uuid.uuid4())
#         if request_id is None:
#             request_id = str(uuid.uuid4())

#         state = schema.create_state(
#             inputs=inputs,
#             user_id=user_id,
#             request_id=request_id,
#             session_id=session_id
#         )

#         result = await self._graph.run(state)

#         # Note: Langfuse/observability flush is handled externally
#         # You can add hooks here for custom observability

#         return result

#     def get_state(self, request_id: str) -> MemoryState:
#         """Get workflow state by request ID."""
#         pass
