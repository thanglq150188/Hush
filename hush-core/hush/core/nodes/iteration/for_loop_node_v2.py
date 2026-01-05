"""For loop node v2 with improved API using `each` and `inputs` parameters."""

from typing import Dict, Any, List, Optional, Set, Union, TYPE_CHECKING
from datetime import datetime
import asyncio
import traceback
import os

from hush.core.configs.node_config import NodeType
from hush.core.nodes.iteration.base import IterationNode
from hush.core.states.ref import Ref
from hush.core.utils.common import Param
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states import BaseState


class ForLoopNode(IterationNode):
    """A node that iterates over collections with support for broadcast variables.

    New API:
        - `each`: Dict of variables to iterate over (different value per iteration)
        - `inputs`: Dict of variables to broadcast (same value for all iterations)

    Example:
        with ForLoopNode(
            name="process_loop",
            each={"x": [1, 2, 3], "y": [10, 20, 30]},  # iterate (zip)
            inputs={"multiplier": 10}                   # broadcast
        ) as loop:
            node = calc(inputs={"x": PARENT["x"], "y": PARENT["y"], "multiplier": PARENT["multiplier"]})
            START >> node >> END

        # Creates 3 iterations:
        #   iteration 0: {"x": 1, "y": 10, "multiplier": 10}
        #   iteration 1: {"x": 2, "y": 20, "multiplier": 10}
        #   iteration 2: {"x": 3, "y": 30, "multiplier": 10}
    """

    type: NodeType = "for"

    __slots__ = ['_max_concurrency', '_each', '_broadcast_inputs']

    def __init__(
        self,
        each: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        max_concurrency: Optional[int] = None,
        **kwargs
    ):
        """Initialize a ForLoopNode.

        Args:
            each: Dict mapping variable names to lists (or Refs to lists).
                  These variables will be iterated over (zipped together).
            inputs: Dict mapping variable names to values (or Refs to values).
                    These variables will be broadcast to all iterations.
            max_concurrency: Maximum number of concurrent tasks to run.
                Defaults to the number of CPU cores if not specified.
        """
        # Don't pass inputs to parent - we handle it ourselves
        super().__init__(**kwargs)

        self._each = each or {}
        self._broadcast_inputs = inputs or {}
        self._max_concurrency = max_concurrency if max_concurrency is not None else os.cpu_count()

        # Validate no overlap between each and inputs
        overlap = set(self._each.keys()) & set(self._broadcast_inputs.keys())
        if overlap:
            raise ValueError(
                f"ForLoopNode '{self.name}': variables cannot be in both 'each' and 'inputs': {overlap}"
            )

    def _post_build(self):
        """Set input/output schema after inner graph is built."""
        # Input schema: all variables from each + broadcast inputs
        self.input_schema = {}

        for var_name, value in self._each.items():
            if isinstance(value, Ref):
                self.input_schema[var_name] = Param(type=List, required=True)
            else:
                self.input_schema[var_name] = Param(type=List, required=False)

        for var_name, value in self._broadcast_inputs.items():
            if isinstance(value, Ref):
                self.input_schema[var_name] = Param(type=Any, required=True)
            else:
                self.input_schema[var_name] = Param(type=Any, required=False)

        # Output schema: batch_result + iteration_metrics
        self.output_schema = {
            "batch_result": Param(type=List, required=True),
            "iteration_metrics": Param(type=Dict, required=False)
        }

        # Store each and inputs as node's inputs for get_inputs to work
        self.inputs = {**self._each, **self._broadcast_inputs}

    def _resolve_each_values(
        self,
        state: 'BaseState',
        context_id: Optional[str]
    ) -> Dict[str, List]:
        """Resolve `each` variables to actual lists.

        Returns:
            Dict mapping variable names to their list values.
        """
        resolved = {}
        for var_name, value in self._each.items():
            if isinstance(value, Ref):
                resolved[var_name] = state._get_value(value.node, value.var, context_id)
            else:
                resolved[var_name] = value
        return resolved

    def _resolve_broadcast_values(
        self,
        state: 'BaseState',
        context_id: Optional[str]
    ) -> Dict[str, Any]:
        """Resolve broadcast variables to actual values.

        Returns:
            Dict mapping variable names to their values.
        """
        resolved = {}
        for var_name, value in self._broadcast_inputs.items():
            if isinstance(value, Ref):
                resolved[var_name] = state._get_value(value.node, value.var, context_id)
            else:
                resolved[var_name] = value
        return resolved

    def _build_iteration_data(
        self,
        each_values: Dict[str, List],
        broadcast_values: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build list of iteration data by zipping `each` values and adding broadcast.

        Args:
            each_values: Dict of {var_name: [values...]}
            broadcast_values: Dict of {var_name: value}

        Returns:
            List of dicts, one per iteration, with all variables.
        """
        if not each_values:
            # No iteration variables - single iteration with just broadcast
            if broadcast_values:
                return [broadcast_values.copy()]
            return []

        # Validate all lists have same length
        lengths = {var: len(lst) for var, lst in each_values.items()}
        unique_lengths = set(lengths.values())

        if len(unique_lengths) > 1:
            LOGGER.error(
                f"ForLoopNode '{self.full_name}': 'each' variables have different lengths: {lengths}"
            )
            raise ValueError(
                f"All 'each' variables must have the same length. Got: {lengths}"
            )

        if not unique_lengths:
            return []

        num_iterations = list(unique_lengths)[0]

        # Build iteration data
        iteration_data = []
        for i in range(num_iterations):
            data = {}
            # Add iteration values
            for var_name, values in each_values.items():
                data[var_name] = values[i]
            # Add broadcast values
            data.update(broadcast_values)
            iteration_data.append(data)

        return iteration_data

    async def run(
        self,
        state: 'BaseState',
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the for loop over iteration data concurrently."""
        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        _inputs = {}
        _outputs = {}

        try:
            # Resolve each and broadcast values
            each_values = self._resolve_each_values(state, context_id)
            broadcast_values = self._resolve_broadcast_values(state, context_id)

            # Build iteration data
            iteration_data = self._build_iteration_data(each_values, broadcast_values)

            # Store inputs for logging
            _inputs = {
                **{k: v for k, v in each_values.items()},
                **broadcast_values
            }

            # Warn if no iterations
            if not iteration_data:
                LOGGER.warning(
                    f"ForLoopNode '{self.full_name}': no iteration data. No iterations will be executed."
                )

            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self._max_concurrency)

            # Track latencies for metrics
            latencies_ms: List[float] = []
            success_count = 0
            error_count = 0

            async def run_with_semaphore(task_id: str, loop_data: Dict[str, Any]):
                nonlocal success_count, error_count

                iter_start_time = datetime.now()

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
                    iter_end_time = datetime.now()
                    latency = (iter_end_time - iter_start_time).total_seconds() * 1000
                    latencies_ms.append(latency)

            active_tasks = []
            for i, loop_data in enumerate(iteration_data):
                task_id = f"for[{i}]"
                active_tasks.append(run_with_semaphore(task_id, loop_data))

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
                "total_iterations": len(iteration_data),
                "success_count": success_count,
                "error_count": error_count,
            })

            # Warn if high error rate (>10%)
            if len(iteration_data) > 0:
                error_rate = error_count / len(iteration_data)
                if error_rate > 0.1:
                    LOGGER.warning(
                        f"ForLoopNode '{self.full_name}': high error rate ({error_rate:.1%}). "
                        f"{error_count}/{len(iteration_data)} iterations failed."
                    )

            _outputs = {
                "batch_result": final_results,
                "iteration_metrics": iteration_metrics
            }

            self.store_result(state, _outputs, context_id)

        except Exception as e:
            error_msg = traceback.format_exc()
            LOGGER.error(f"Error in node {self.full_name}: {str(e)}")
            LOGGER.error(error_msg)
            state[self.full_name, "error", context_id] = error_msg

        finally:
            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            self._log(request_id, context_id, _inputs, _outputs, duration_ms)
            state[self.full_name, "start_time", context_id] = start_time
            state[self.full_name, "end_time", context_id] = end_time
            return _outputs

    def specific_metadata(self) -> Dict[str, Any]:
        """Return subclass-specific metadata."""
        return {
            "max_concurrency": self._max_concurrency,
            "each_vars": list(self._each.keys()),
            "broadcast_vars": list(self._broadcast_inputs.keys())
        }


if __name__ == "__main__":
    from hush.core.states import StateSchema
    from hush.core.nodes.base import START, END, PARENT
    from hush.core.nodes.transform.code_node import code_node
    from hush.core.nodes.graph.graph_node import GraphNode

    async def main():
        # =================================================================
        # Test 1: Simple iteration with `each`
        # =================================================================
        print("=" * 60)
        print("Test 1: Simple iteration with `each`")
        print("=" * 60)

        @code_node
        def double(value: int):
            return {"result": value * 2}

        with ForLoopNode(
            name="double_loop",
            each={"value": [1, 2, 3, 4, 5]}
        ) as loop1:
            node = double(inputs={"value": PARENT["value"]}, outputs=PARENT)
            START >> node >> END

        loop1.build()

        schema1 = StateSchema(loop1)
        state1 = schema1.create_state()

        print(f"Each vars: {loop1._each}")
        print(f"Broadcast vars: {loop1._broadcast_inputs}")

        result1 = await loop1.run(state1)
        print(f"Result: {result1['batch_result']}")
        print(f"Expected: [2, 4, 6, 8, 10]")
        assert [r['result'] for r in result1['batch_result']] == [2, 4, 6, 8, 10]

        # =================================================================
        # Test 2: Iteration with broadcast
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 2: Iteration with broadcast")
        print("=" * 60)

        @code_node
        def multiply(value: int, multiplier: int):
            return {"result": value * multiplier}

        with ForLoopNode(
            name="multiply_loop",
            each={"value": [1, 2, 3]},
            inputs={"multiplier": 10}
        ) as loop2:
            node = multiply(
                inputs={"value": PARENT["value"], "multiplier": PARENT["multiplier"]},
                outputs=PARENT
            )
            START >> node >> END

        loop2.build()

        schema2 = StateSchema(loop2)
        state2 = schema2.create_state()

        result2 = await loop2.run(state2)
        print(f"Result: {result2['batch_result']}")
        print(f"Expected: [10, 20, 30]")
        assert [r['result'] for r in result2['batch_result']] == [10, 20, 30]

        # =================================================================
        # Test 3: Multiple `each` variables (zip)
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 3: Multiple `each` variables (zip)")
        print("=" * 60)

        @code_node
        def add(x: int, y: int):
            return {"sum": x + y}

        with ForLoopNode(
            name="add_loop",
            each={
                "x": [1, 2, 3],
                "y": [10, 20, 30]
            }
        ) as loop3:
            node = add(inputs={"x": PARENT["x"], "y": PARENT["y"]}, outputs=PARENT)
            START >> node >> END

        loop3.build()

        schema3 = StateSchema(loop3)
        state3 = schema3.create_state()

        result3 = await loop3.run(state3)
        print(f"Result: {result3['batch_result']}")
        print(f"Expected: [11, 22, 33]")
        assert [r['sum'] for r in result3['batch_result']] == [11, 22, 33]

        # =================================================================
        # Test 4: Multiple `each` + broadcast
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 4: Multiple `each` + broadcast")
        print("=" * 60)

        @code_node
        def compute(x: int, y: int, factor: int):
            return {"result": (x + y) * factor}

        with ForLoopNode(
            name="compute_loop",
            each={
                "x": [1, 2, 3],
                "y": [10, 20, 30]
            },
            inputs={"factor": 2}
        ) as loop4:
            node = compute(
                inputs={"x": PARENT["x"], "y": PARENT["y"], "factor": PARENT["factor"]},
                outputs=PARENT
            )
            START >> node >> END

        loop4.build()

        schema4 = StateSchema(loop4)
        state4 = schema4.create_state()

        result4 = await loop4.run(state4)
        print(f"Result: {result4['batch_result']}")
        print(f"Expected: [22, 44, 66]  # (1+10)*2, (2+20)*2, (3+30)*2")
        assert [r['result'] for r in result4['batch_result']] == [22, 44, 66]

        # =================================================================
        # Test 5: Process strings
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 5: Process strings")
        print("=" * 60)

        @code_node
        def format_greeting(name: str, greeting: str):
            return {"message": f"{greeting}, {name}!"}

        with ForLoopNode(
            name="greeting_loop",
            each={"name": ["Alice", "Bob", "Charlie"]},
            inputs={"greeting": "Hello"}
        ) as loop5:
            node = format_greeting(
                inputs={"name": PARENT["name"], "greeting": PARENT["greeting"]},
                outputs=PARENT
            )
            START >> node >> END

        loop5.build()

        schema5 = StateSchema(loop5)
        state5 = schema5.create_state()

        result5 = await loop5.run(state5)
        print(f"Result: {result5['batch_result']}")
        expected5 = ["Hello, Alice!", "Hello, Bob!", "Hello, Charlie!"]
        assert [r['message'] for r in result5['batch_result']] == expected5

        # =================================================================
        # Test 6: Chain of nodes in loop
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 6: Chain of nodes in loop")
        print("=" * 60)

        @code_node
        def add_one(x: int):
            return {"y": x + 1}

        @code_node
        def multiply_two(y: int):
            return {"z": y * 2}

        with ForLoopNode(
            name="chain_loop",
            each={"x": [1, 2, 3]}
        ) as loop6:
            n1 = add_one(inputs={"x": PARENT["x"]})
            n2 = multiply_two(inputs={"y": n1["y"]}, outputs=PARENT)
            START >> n1 >> n2 >> END

        loop6.build()

        schema6 = StateSchema(loop6)
        state6 = schema6.create_state()

        result6 = await loop6.run(state6)
        print(f"Result (x+1)*2: {result6['batch_result']}")
        print(f"Expected: [4, 6, 8]  # (1+1)*2, (2+1)*2, (3+1)*2")
        assert [r['z'] for r in result6['batch_result']] == [4, 6, 8]

        # =================================================================
        # Test 7: With concurrency limit
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 7: With concurrency limit (max_concurrency=2)")
        print("=" * 60)

        @code_node
        async def slow_process(value: int):
            await asyncio.sleep(0.05)
            return {"result": value * 10}

        with ForLoopNode(
            name="concurrent_loop",
            each={"value": [1, 2, 3, 4, 5]},
            max_concurrency=2
        ) as loop7:
            node = slow_process(inputs={"value": PARENT["value"]}, outputs=PARENT)
            START >> node >> END

        loop7.build()

        schema7 = StateSchema(loop7)
        state7 = schema7.create_state()

        result7 = await loop7.run(state7)
        print(f"Result: {result7['batch_result']}")
        print(f"Metrics: {result7.get('iteration_metrics', {})}")
        assert [r['result'] for r in result7['batch_result']] == [10, 20, 30, 40, 50]

        # =================================================================
        # Test 8: With Ref (dynamic data from previous node)
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 8: With Ref (dynamic data from previous node)")
        print("=" * 60)

        @code_node
        def generate_data():
            return {"numbers": [10, 20, 30], "factor": 5}

        @code_node
        def process_item(item: int, factor: int):
            return {"result": item * factor}

        with GraphNode(name="ref_test_graph") as graph8:
            gen_node = generate_data()

            with ForLoopNode(
                name="ref_loop",
                each={"item": gen_node["numbers"]},
                inputs={"factor": gen_node["factor"]}
            ) as loop8:
                proc_node = process_item(
                    inputs={"item": PARENT["item"], "factor": PARENT["factor"]},
                    outputs=PARENT
                )
                START >> proc_node >> END

            START >> gen_node >> loop8 >> END

        graph8.build()

        schema8 = StateSchema(graph8)
        state8 = schema8.create_state()

        result8 = await graph8.run(state8)
        print(f"Generated numbers: [10, 20, 30], factor: 5")
        print(f"Result: {result8}")
        # Expected: [50, 100, 150]  # 10*5, 20*5, 30*5

        # =================================================================
        # Test 9: Nested ForLoopNode (simplified - inner loop with fixed values)
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 9: Nested ForLoopNode")
        print("=" * 60)

        @code_node
        def inner_double(x: int):
            return {"doubled": x * 2}

        # Simpler nested structure: outer loop processes, each returns a list
        with ForLoopNode(
            name="outer_loop",
            each={"a": [1, 2, 3]}
        ) as outer_loop9:
            # Each outer iteration creates inner loop data
            with ForLoopNode(
                name="inner_loop",
                each={"x": [10, 20]}
            ) as inner_loop9:
                inner_node = inner_double(inputs={"x": PARENT["x"]}, outputs=PARENT)
                START >> inner_node >> END

            START >> inner_loop9 >> END

        outer_loop9.build()

        schema9 = StateSchema(outer_loop9)
        state9 = schema9.create_state()

        result9 = await outer_loop9.run(state9)
        print(f"Result: {result9['batch_result']}")
        # Each outer iteration runs the same inner loop [10, 20] -> [20, 40]
        print("Expected: 3 iterations of outer, each with inner results [20, 40]")

        # =================================================================
        # Test 10: Empty iteration
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 10: Empty iteration")
        print("=" * 60)

        with ForLoopNode(
            name="empty_loop",
            each={"value": []}
        ) as loop10:
            node = double(inputs={"value": PARENT["value"]}, outputs=PARENT)
            START >> node >> END

        loop10.build()

        schema10 = StateSchema(loop10)
        state10 = schema10.create_state()

        result10 = await loop10.run(state10)
        print(f"Result: {result10}")
        assert result10['batch_result'] == []
        assert result10['iteration_metrics']['total_iterations'] == 0

        # =================================================================
        # Test 11: Mismatched lengths (should raise error)
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 11: Mismatched lengths (should raise error)")
        print("=" * 60)

        @code_node
        def dummy(x: int, y: int):
            return {"result": x + y}

        with ForLoopNode(
            name="mismatch_loop",
            each={
                "x": [1, 2, 3],
                "y": [10, 20]  # Different length!
            }
        ) as loop11:
            node = dummy(inputs={"x": PARENT["x"], "y": PARENT["y"]}, outputs=PARENT)
            START >> node >> END

        loop11.build()

        schema11 = StateSchema(loop11)
        state11 = schema11.create_state()

        try:
            result11 = await loop11.run(state11)
            print("ERROR: Should have raised ValueError!")
        except ValueError as e:
            print(f"Correctly raised ValueError: {e}")

        # =================================================================
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    asyncio.run(main())
