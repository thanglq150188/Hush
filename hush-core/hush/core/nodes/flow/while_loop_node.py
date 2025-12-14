"""While loop node for conditional iteration."""

from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
import traceback

from hush.core.configs.node_config import NodeType
from hush.core.nodes.base import BaseNode
from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.states import BaseState
from hush.core.utils.context import _current_graph
from hush.core.utils.common import Param
from hush.core.loggings import LOGGER


class WhileLoopNode(BaseNode):
    """A node that iterates over a condition."""

    type: NodeType = "while"

    __slots__ = ['_graph', '_token']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._graph: GraphNode = GraphNode(name=BaseNode.INNER_PROCESS)
        self._graph.father = self
        self._token = None

    def __enter__(self):
        self._token = _current_graph.set(self._graph)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _current_graph.reset(self._token)

    def build(self):
        self._graph.build()

        # Copy schemas from inner graph
        self.input_schema = dict(self._graph.input_schema)
        self.output_schema = dict(self._graph.output_schema)

        # Add any extra input keys
        for key in self.inputs:
            if key not in self.input_schema:
                self.input_schema[key] = Param(type=Any, required=True)

        # Re-normalize connections with updated schema
        self.inputs = self._normalize_connections(self.inputs, self.input_schema)

    def add_node(self, node: BaseNode) -> BaseNode:
        return self._graph.add_node(node)

    def add_edge(self, source: str, target: str, type=None):
        return self._graph.add_edge(source, target, type)

    async def run(
        self,
        state: BaseState,
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:

        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        _outputs = {}

        try:
            _inputs = self.get_inputs(state, context_id=context_id)

            LOGGER.info("request[%s] - NODE: %s[%s] (%s) inputs=%s",
                request_id, self.name, context_id, str(self.type).upper(), str(_inputs)[:200])

            _continue = True
            step_count = 0
            step_inputs = _inputs

            while _continue:
                step_name = f"while-{step_count}"

                # Inject inputs into inner graph
                for var_name, value in step_inputs.items():
                    state[self._graph.full_name, var_name, step_name] = value

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
            end_time = datetime.now()
            state[self.full_name, "start_time", context_id] = start_time
            state[self.full_name, "end_time", context_id] = end_time
            self.store_result(state, _outputs, context_id=context_id)

            LOGGER.info("request[%s] - NODE: %s[%s] (%s) outputs=%s",
                request_id, self.name, context_id, str(self.type).upper(), str(_outputs)[:200])

            return _outputs
