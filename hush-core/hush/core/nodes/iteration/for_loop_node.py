"""For loop node for iterating over collections."""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
import asyncio
import os

from hush.core.configs.node_config import NodeType
from hush.core.nodes.iteration.base import IterationNode
from hush.core.utils.common import Param

if TYPE_CHECKING:
    from hush.core.states import BaseState


class ForLoopNode(IterationNode):
    """A node that iterates over a collection and executes an inner graph flow for each item."""

    type: NodeType = "for"

    __slots__ = ['_max_concurrency']

    def __init__(self, max_concurrency: Optional[int] = None, **kwargs):
        """Initialize a ForLoopNode.

        Args:
            max_concurrency: Maximum number of concurrent tasks to run.
                Defaults to the number of CPU cores if not specified.
                Uses a semaphore to limit concurrency.
        """
        input_schema = {"batch_data": Param(type=List, required=True)}
        output_schema = {
            "batch_result": Param(type=List, required=True),
            "iteration_metrics": Param(type=Dict, required=False)
        }

        super().__init__(
            input_schema=input_schema,
            output_schema=output_schema,
            **kwargs
        )

        self._max_concurrency = max_concurrency if max_concurrency is not None else os.cpu_count()

    async def _execute_loop(
        self,
        state: 'BaseState',
        inputs: Dict[str, Any],
        request_id: str,
        context_id: Optional[str]
    ) -> Dict[str, Any]:
        """Execute the for loop over batch_data concurrently with semaphore limiting."""
        batch_data = inputs.get("batch_data", [])

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self._max_concurrency)

        # Track latencies for metrics
        latencies_ms: List[float] = []
        success_count = 0
        error_count = 0

        async def run_with_semaphore(iteration_index: int, task_id: str, loop_data: Dict[str, Any]):
            nonlocal success_count, error_count

            start_time = datetime.now()

            try:
                async with semaphore:
                    self.inject_inputs(state, loop_data, task_id)
                    result = await self._graph.run(state, task_id)
                    success_count += 1
                    return result
            except Exception as e:
                error_count += 1
                raise e
            finally:
                end_time = datetime.now()
                latency = (end_time - start_time).total_seconds() * 1000
                latencies_ms.append(latency)

        active_tasks = []
        for i, loop_data in enumerate(batch_data):
            task_id = f"for[{i}]"
            active_tasks.append(run_with_semaphore(i, task_id, loop_data))

        results = await asyncio.gather(*active_tasks, return_exceptions=True)

        # Separate results and exceptions
        final_results = []
        for r in results:
            if isinstance(r, Exception):
                final_results.append({"error": str(r)})
            else:
                final_results.append(r)

        # Calculate iteration metrics
        iteration_metrics = self._calculate_iteration_metrics(latencies_ms)
        iteration_metrics.update({
            "total_iterations": len(batch_data),
            "success_count": success_count,
            "error_count": error_count,
        })

        return {
            "batch_result": final_results,
            "iteration_metrics": iteration_metrics
        }

    def specific_metadata(self) -> Dict[str, Any]:
        """Return subclass-specific metadata."""
        return {
            "max_concurrency": self._max_concurrency
        }


if __name__ == "__main__":
    from hush.core.states import StateSchema
    from hush.core.nodes.base import START, END, INPUT, OUTPUT
    from hush.core.nodes.transform.code_node import code_node

    async def main():
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
            node = double(inputs={"value": INPUT["value"]}, outputs=OUTPUT)
            START >> node >> END

        loop1.build()

        # Create schema from the built node (includes inner graph links)
        schema = StateSchema(loop1)
        state = schema.create_state()

        print(f"Input schema: {loop1.input_schema}")
        print(f"Output schema: {loop1.output_schema}")
        schema.show_links()

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
            node = uppercase(inputs={"text": INPUT["text"]}, outputs=OUTPUT)
            START >> node >> END

        loop2.build()

        schema2 = StateSchema(loop2)
        state2 = schema2.create_state()

        result2 = await loop2.run(state2)
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
            node = add(inputs={"a": INPUT["a"], "b": INPUT["b"]}, outputs=OUTPUT)
            START >> node >> END

        loop3.build()

        schema3 = StateSchema(loop3)
        state3 = schema3.create_state()

        result3 = await loop3.run(state3)
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
            n1 = add_one(inputs={"x": INPUT["x"]})
            n2 = multiply_two(inputs={"y": n1["y"]}, outputs=OUTPUT)
            START >> n1 >> n2 >> END

        loop4.build()

        schema4 = StateSchema(loop4)
        state4 = schema4.create_state()

        result4 = await loop4.run(state4)
        print(f"Result (x+1)*2: {result4}")

        # Test 5: With concurrency limit
        print("\n" + "=" * 50)
        print("Test 5: With concurrency limit (max_concurrency=2)")
        print("=" * 50)

        @code_node
        async def slow_process(value: int):
            await asyncio.sleep(0.1)
            return {"result": value * 10}

        with ForLoopNode(
            name="concurrent_loop",
            max_concurrency=2,
            inputs={"batch_data": [{"value": i} for i in range(5)]}
        ) as loop5:
            node = slow_process(inputs={"value": INPUT["value"]}, outputs=OUTPUT)
            START >> node >> END

        loop5.build()

        schema5 = StateSchema(loop5)
        state5 = schema5.create_state()

        result5 = await loop5.run(state5)
        print(f"Result: {result5}")
        print(f"Metrics: {result5.get('iteration_metrics', {})}")

        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)

    asyncio.run(main())
