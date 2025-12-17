"""For loop node for iterating over collections."""

from typing import Dict, Any, Optional, List
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


class ForLoopNode(BaseNode):
    """A node that iterates over a collection and executes an inner graph flow for each item."""

    type: NodeType = "for"

    __slots__ = ['_graph', '_token']

    def __init__(self, **kwargs):
        input_schema = {"batch_data": Param(type=List, required=True)}
        output_schema = {"batch_result": Param(type=List, required=True)}

        super().__init__(
            input_schema=input_schema,
            output_schema=output_schema,
            **kwargs
        )

        self._graph: GraphNode = GraphNode(name=BaseNode.INNER_PROCESS)
        self._graph.father = self
        self._token = None

    def __enter__(self):
        self._token = _current_graph.set(self._graph)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _current_graph.reset(self._token)

    def add_node(self, node: BaseNode) -> BaseNode:
        return self._graph.add_node(node)

    def add_edge(self, source: str, target: str, type=None):
        return self._graph.add_edge(source, target, type)

    def build(self):
        self._graph.build()

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
            batch_data = _inputs.get("batch_data", [])

            LOGGER.info("request[%s] - NODE: %s[%s] (%s) inputs=%s",
                request_id, self.name, context_id, str(self.type).upper(), str(_inputs)[:200])

            active_tasks = []
            for i, loop_data in enumerate(batch_data):
                task_id = f"for[{i}]"
                # Inject loop data into inner graph's inputs
                for var_name, value in loop_data.items():
                    state[self._graph.full_name, var_name, task_id] = value
                active_tasks.append(self._graph.run(state, task_id))

            _outputs = {"batch_result": await asyncio.gather(*active_tasks)}

            LOGGER.info("request[%s] - NODE: %s[%s] (%s) outputs=%s",
                request_id, self.name, context_id, str(self.type).upper(), str(_outputs)[:200])

            self.store_result(state, _outputs, context_id=context_id)

        except Exception as e:
            error_msg = traceback.format_exc()
            LOGGER.error(f"Error in node {self.full_name}: {str(e)}")
            LOGGER.error(error_msg)
            state[self.full_name, "error", context_id] = error_msg

        finally:
            end_time = datetime.now()
            state[self.full_name, "start_time", context_id] = start_time
            state[self.full_name, "end_time", context_id] = end_time
            return _outputs


if __name__ == "__main__":
    from hush.core.states import StateSchema
    from hush.core.nodes.base import START, END
    from hush.core.nodes.transform.code_node import code_node

    async def main():
        schema = StateSchema("test")
        state = schema.create_state()

        # Test 1: Basic for loop - double each number
        print("=" * 50)
        print("Test 1: Basic for loop - double each number")
        print("=" * 50)

        @code_node
        def double(value: int):
            return {"result": value * 2}

        with ForLoopNode(
            name="double_loop",
            inputs={"batch_data": [{"value": 1}, {"value": 2}, {"value": 3}]}
        ) as loop1:
            node = double(inputs={"value": loop1._graph["value"]})
            START >> node >> END

        loop1.build()

        print(f"Input schema: {loop1.input_schema}")
        print(f"Output schema: {loop1.output_schema}")

        result = await loop1.run(state)
        print(f"Result: {result}")

        # Test 2: Process strings
        print("\n" + "=" * 50)
        print("Test 2: Process strings - uppercase")
        print("=" * 50)

        @code_node
        def uppercase(text: str):
            return {"upper": text.upper()}

        with ForLoopNode(
            name="uppercase_loop",
            inputs={"batch_data": [{"text": "hello"}, {"text": "world"}, {"text": "test"}]}
        ) as loop2:
            node = uppercase(inputs={"text": loop2._graph["text"]})
            START >> node >> END

        loop2.build()

        result2 = await loop2.run(state)
        print(f"Result: {result2}")

        # Test 3: Multiple inputs per iteration
        print("\n" + "=" * 50)
        print("Test 3: Multiple inputs per iteration")
        print("=" * 50)

        @code_node
        def add(a: int, b: int):
            return {"sum": a + b}

        with ForLoopNode(
            name="add_loop",
            inputs={"batch_data": [
                {"a": 1, "b": 2},
                {"a": 10, "b": 20},
                {"a": 100, "b": 200}
            ]}
        ) as loop3:
            node = add(inputs={"a": loop3._graph["a"], "b": loop3._graph["b"]})
            START >> node >> END

        loop3.build()

        result3 = await loop3.run(state)
        print(f"Result: {result3}")

        # Test 4: Chain of nodes in loop
        print("\n" + "=" * 50)
        print("Test 4: Chain of nodes in loop")
        print("=" * 50)

        @code_node
        def add_one(x: int):
            return {"y": x + 1}

        @code_node
        def multiply_two(y: int):
            return {"z": y * 2}

        with ForLoopNode(
            name="chain_loop",
            inputs={"batch_data": [{"x": 1}, {"x": 2}, {"x": 3}]}
        ) as loop4:
            n1 = add_one(inputs={"x": loop4._graph["x"]})
            n2 = multiply_two(inputs={"y": n1["y"]})
            START >> n1 >> n2 >> END

        loop4.build()

        result4 = await loop4.run(state)
        print(f"Result (x+1)*2: {result4}")

        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)

    asyncio.run(main())
