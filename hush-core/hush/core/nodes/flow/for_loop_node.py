"""For loop node for iterating over collections."""

from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import traceback

from hush.core.configs.node_config import NodeType
from hush.core.nodes.base import BaseNode
from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.states.workflow_state import WorkflowState
from hush.core.utils.context import _current_graph
from hush.core.schema import ParamSet
from hush.core.loggings import LOGGER



class ForLoopNode(BaseNode):
    """A node that iterates over a collection and executes an inner graph flow for each item."""

    type: NodeType = "for"

    input_schema: ParamSet = (
        ParamSet.new()
            .var("batch_data: List", required=True)
            .build()
    )

    output_schema: ParamSet = (
        ParamSet.new()
            .var("batch_result: List", required=True)
            .build()
    )

    def __init__(self, **kwargs):
        """Initialize ForLoopNode."""
        super().__init__(**kwargs)

        self._graph: GraphNode = GraphNode(name=BaseNode.INNER_PROCESS)
        self._graph.father = self
        self._token = None

    def __enter__(self):
        """Enter context manager mode."""
        self._token = _current_graph.set(self._graph)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        _current_graph.reset(self._token)

    def add_node(self, node: BaseNode) -> BaseNode:
        """Delegate node addition to inner graph."""
        return self._graph.add_node(node)

    def add_edge(self, source: str, target: str, type=None):
        """Delegate edge addition to inner graph."""
        return self._graph.add_edge(source, target, type)

    def build(self):
        self._graph.build()
        self._post_init()

    async def run(
        self,
        state: WorkflowState,
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:

        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id

        start_time = datetime.now()

        _outputs = {}

        try:
            _inputs = self.get_inputs(state, context_id=context_id)

            batch_data = _inputs.get("batch_data", [])

            LOGGER.info("request[%s] - running NODE: %s[%s] (%s), inputs = %s",
                    request_id, self.name, context_id, str(self.type).upper(), str(_inputs)[:200])

            active_tasks = []
            for (i, loop_data) in enumerate(batch_data):
                task_id = f"for[{i}]"
                state.inject_inputs(
                    node=self._graph.full_name,
                    inputs=loop_data,
                    context_id=task_id
                )
                active_tasks.append(self._graph.run(state, task_id))

            _outputs = {"batch_result": await asyncio.gather(*active_tasks)}

            LOGGER.info("request[%s] - running NODE: %s[%s] (%s), outputs = %s",
                    request_id, self.name, context_id, str(self.type).upper(), str(_outputs)[:200])

            self.store_result(state, _outputs, context_id=context_id)

        except Exception as e:
            error_msg = traceback.format_exc()
            LOGGER.error(f"Error in node {self.full_name}: {str(e)}")
            LOGGER.error(error_msg)
            state[self.full_name, "error", context_id] = error_msg

        finally:
            await asyncio.sleep(0.000001)
            end_time = datetime.now()
            state.set_by_index(self.metrics['start_time'], start_time, context_id=context_id)
            state.set_by_index(self.metrics['end_time'], end_time, context_id=context_id)
            return _outputs
