"""While loop node for conditional iteration."""

from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import traceback

from hush.core.configs.node_config import NodeType
from hush.core.nodes.base import BaseNode
from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.states.workflow_state import WorkflowState
from hush.core.utils.context import _current_graph
from hush.core.schema import Param
from hush.core.loggings import LOGGER



class WhileLoopNode(BaseNode):
    """A node that iterates over a condition."""

    type: NodeType = "while"

    def __init__(self, **kwargs):
        """Initialize WhileLoopNode."""
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

    def build(self):
        self._graph.build()

        self.input_schema = self._graph.input_schema
        self.output_schema = self._graph.output_schema

        for key in self.inputs:
            if key not in self.input_schema:
                self.input_schema.params[key] = Param(type="Any", required=True)

        self._post_init()

    def add_node(self, node: BaseNode) -> BaseNode:
        """Delegate node addition to inner graph."""
        return self._graph.add_node(node)

    def add_edge(self, source: str, target: str, type=None):
        """Delegate edge addition to inner graph."""
        return self._graph.add_edge(source, target, type)

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

            LOGGER.info("request[%s] - running NODE: %s[%s] (%s), inputs = %s",
                    request_id, self.name, context_id, str(self.type).upper(), str(_inputs)[:200])

            _continue = True
            step_count = 0

            step_inputs = _inputs

            while _continue:
                step_name = f"while-{step_count}"

                state.inject_inputs(
                    node=self._graph.full_name,
                    inputs=step_inputs,
                    context_id=step_name
                )

                _outputs = await self._graph.run(state, context_id=step_name)

                step_inputs = self._graph.get_inputs(state, context_id=step_name)
                step_count += 1

                _continue = _outputs.get("continue_loop", False)

        except Exception as e:
            error_msg = traceback.format_exc()
            LOGGER.error(f"Error in node {self.name}: {str(e)}")
            LOGGER.error(error_msg)
            state[self.full_name, "error", context_id] = error_msg

        finally:
            await asyncio.sleep(0.000001)
            end_time = datetime.now()
            state.set_by_index(self.metrics['start_time'], start_time, context_id=context_id)
            state.set_by_index(self.metrics['end_time'], end_time, context_id=context_id)
            self.store_result(state, _outputs, context_id=context_id)

            LOGGER.info("request[%s] - running NODE: %s[%s] (%s), _outputs = %s",
                    request_id, self.name, context_id, str(self.type).upper(), str(_outputs)[:200])

            return _outputs
