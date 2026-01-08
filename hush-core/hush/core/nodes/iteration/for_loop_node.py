"""For loop node with unified inputs API using Each wrapper for iteration sources."""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
from time import perf_counter
import asyncio
import traceback
import os

from hush.core.configs.node_config import NodeType
from hush.core.nodes.iteration.base import BaseIterationNode, Each
from hush.core.states.ref import Ref
from hush.core.utils.common import Param
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states import MemoryState


class ForLoopNode(BaseIterationNode):
    """A node that iterates over collections with support for broadcast variables.

    API (unified inputs with Each wrapper):
        - `inputs`: Dict of all variables. Use Each() wrapper for iteration sources.
          - Each(source): Variable iterated over (different value per iteration)
          - Regular values: Broadcast to all iterations (same value)

    Example:
        with ForLoopNode(
            name="process_loop",
            inputs={
                "x": Each([1, 2, 3]),           # iterate
                "y": Each([10, 20, 30]),        # iterate (zipped with x)
                "multiplier": 10                 # broadcast
            }
        ) as loop:
            node = calc(inputs={"x": PARENT["x"], "y": PARENT["y"], "multiplier": PARENT["multiplier"]})
            START >> node >> END

        # Creates 3 iterations:
        #   iteration 0: {"x": 1, "y": 10, "multiplier": 10}
        #   iteration 1: {"x": 2, "y": 20, "multiplier": 10}
        #   iteration 2: {"x": 3, "y": 30, "multiplier": 10}

        # Output (column-oriented, transposed from inner graph outputs):
        #   {"result": [r1, r2, r3], "iteration_metrics": {...}}
    """

    type: NodeType = "for"

    __slots__ = ['_max_concurrency', '_each', '_broadcast_inputs', '_raw_outputs', '_raw_inputs']

    def __init__(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        max_concurrency: Optional[int] = None,
        **kwargs
    ):
        """Initialize a ForLoopNode.

        Args:
            inputs: Dict mapping variable names to values or Each(source).
                    - Each(source): Iterated over (zipped if multiple)
                    - Other values: Broadcast to all iterations
                    Values can be literals or Refs to upstream nodes.
            max_concurrency: Maximum number of concurrent tasks to run.
                Defaults to the number of CPU cores if not specified.
        """
        # Store raw outputs and inputs before super().__init__ normalizes them
        self._raw_outputs = kwargs.get('outputs')
        self._raw_inputs = inputs or {}

        # Don't pass inputs to parent - we handle it ourselves
        super().__init__(**kwargs)

        # Separate Each() sources from broadcast inputs
        self._each = {}
        self._broadcast_inputs = {}

        for var_name, value in self._raw_inputs.items():
            if isinstance(value, Each):
                self._each[var_name] = value.source
            else:
                self._broadcast_inputs[var_name] = value

        self._max_concurrency = max_concurrency if max_concurrency is not None else os.cpu_count()

    def _post_build(self):
        """Set input/output schema after inner graph is built.

        IMPORTANT: This method normalizes _each and _broadcast_inputs using
        _normalize_connections to resolve PARENT refs to self.father.
        """
        # Normalize each and broadcast inputs (resolves PARENT refs)
        self._each = self._normalize_connections(self._each, {})
        self._broadcast_inputs = self._normalize_connections(self._broadcast_inputs, {})

        # Input schema: all variables from each (List type) + broadcast inputs (Any type)
        self.input_schema = {
            var_name: Param(type=List, required=isinstance(value, Ref))
            for var_name, value in self._each.items()
        }
        self.input_schema.update({
            var_name: Param(type=Any, required=isinstance(value, Ref))
            for var_name, value in self._broadcast_inputs.items()
        })

        # Output schema: derived from inner graph's output schema (column-oriented)
        self.output_schema = {
            key: Param(type=List, required=param.required)
            for key, param in self._graph.output_schema.items()
        }
        self.output_schema["iteration_metrics"] = Param(type=Dict, required=False)

        # Re-normalize outputs now that output_schema is set
        if self._raw_outputs is not None:
            self.outputs = self._normalize_connections(self._raw_outputs, self.output_schema)

        # Store each and inputs as node's inputs for get_inputs to work
        self.inputs = {**self._each, **self._broadcast_inputs}

    def _resolve_values(
        self,
        values: Dict[str, Any],
        state: 'MemoryState',
        context_id: Optional[str]
    ) -> Dict[str, Any]:
        """Resolve values, dereferencing any Ref objects.

        Args:
            values: Dict of {var_name: value_or_ref}
            state: The workflow state
            context_id: The context ID for resolution

        Returns:
            Dict mapping variable names to their resolved values.

        If the Ref has operations recorded (e.g., ref['key'].apply(len)),
        those operations are executed on the retrieved value.
        """
        result = {}
        for var_name, value in values.items():
            if isinstance(value, Ref):
                resolved = state._get_value(value.node, value.var, context_id)
                # Execute any recorded operations on the value
                if value.has_ops:
                    resolved = value.execute(resolved)
                result[var_name] = resolved
            else:
                result[var_name] = value
        return result

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
            return [broadcast_values.copy()] if broadcast_values else []

        # Validate all lists have same length
        lengths = {var: len(lst) for var, lst in each_values.items()}
        if len(set(lengths.values())) > 1:
            LOGGER.error(
                f"ForLoopNode '{self.full_name}': 'each' variables have different lengths: {lengths}"
            )
            raise ValueError(
                f"All 'each' variables must have the same length. Got: {lengths}"
            )

        # Zip and merge with broadcast
        keys = list(each_values.keys())
        return [
            {**dict(zip(keys, vals)), **broadcast_values}
            for vals in zip(*each_values.values())
        ]

    async def run(
        self,
        state: 'MemoryState',
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
            each_values = self._resolve_values(self._each, state, context_id)
            broadcast_values = self._resolve_values(self._broadcast_inputs, state, context_id)

            # Build iteration data
            iteration_data = self._build_iteration_data(each_values, broadcast_values)

            # Store inputs for logging
            _inputs = {**each_values, **broadcast_values}

            # Warn if no iterations
            if not iteration_data:
                LOGGER.warning(
                    f"ForLoopNode '{self.full_name}': no iteration data. No iterations will be executed."
                )

            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self._max_concurrency)

            async def execute_iteration(task_id: str, loop_data: Dict[str, Any]) -> Dict[str, Any]:
                """Execute a single iteration, returning result with metadata."""
                start = perf_counter()
                try:
                    async with semaphore:
                        self.inject_inputs(state, loop_data, task_id)
                        result = await self._graph.run(state, task_id)
                    return {"result": result, "latency_ms": (perf_counter() - start) * 1000, "success": True}
                except Exception as e:
                    return {"result": {"error": str(e), "error_type": type(e).__name__}, "latency_ms": (perf_counter() - start) * 1000, "success": False}

            # Run all iterations concurrently
            raw_results = await asyncio.gather(*[
                execute_iteration(f"for[{i}]", data)
                for i, data in enumerate(iteration_data)
            ])

            # Extract metrics and results in single pass
            latencies_ms = []
            final_results = []
            success_count = 0
            for r in raw_results:
                latencies_ms.append(r["latency_ms"])
                final_results.append(r["result"])
                success_count += r["success"]
            error_count = len(raw_results) - success_count

            # Calculate iteration metrics
            iteration_metrics = self._calculate_iteration_metrics(latencies_ms)
            iteration_metrics.update({
                "total_iterations": len(iteration_data),
                "success_count": success_count,
                "error_count": error_count,
            })

            # Warn if high error rate (>10%)
            if iteration_data and error_count / len(iteration_data) > 0.1:
                LOGGER.warning(
                    f"ForLoopNode '{self.full_name}': high error rate ({error_count / len(iteration_data):.1%}). "
                    f"{error_count}/{len(iteration_data)} iterations failed."
                )

            # Transpose results to column-oriented format
            output_keys = list(self._graph.output_schema.keys())
            _outputs = {
                key: [r.get(key) for r in final_results]
                for key in output_keys
            }
            _outputs["iteration_metrics"] = iteration_metrics

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
            "each": list(self._each.keys()),
            "inputs": list(self._broadcast_inputs.keys())
        }


if __name__ == "__main__":
    from hush.core.states import StateSchema
    from hush.core.nodes.base import START, END, PARENT
    from hush.core.nodes.transform.code_node import code_node
    from hush.core.nodes.graph.graph_node import GraphNode

    async def main():
        # =================================================================
        # Test 1: Simple iteration with Each()
        # =================================================================
        print("=" * 60)
        print("Test 1: Simple iteration with Each()")
        print("=" * 60)

        @code_node
        def double(value: int):
            return {"result": value * 2}

        with ForLoopNode(
            name="double_loop",
            inputs={"value": Each([1, 2, 3, 4, 5])}
        ) as loop1:
            node = double(inputs={"value": PARENT["value"]}, outputs=PARENT)
            START >> node >> END

        loop1.build()

        schema1 = StateSchema(loop1)
        state1 = schema1.create_state()

        print(f"Each vars: {loop1._each}")
        print(f"Broadcast vars: {loop1._broadcast_inputs}")

        result1 = await loop1.run(state1)
        print(f"Result: {result1['result']}")
        print(f"Expected: [2, 4, 6, 8, 10]")
        assert result1['result'] == [2, 4, 6, 8, 10]

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
            inputs={
                "value": Each([1, 2, 3]),
                "multiplier": 10  # broadcast
            }
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
        print(f"Result: {result2['result']}")
        print(f"Expected: [10, 20, 30]")
        assert result2['result'] == [10, 20, 30]

        # =================================================================
        # Test 3: Multiple Each() variables (zip)
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 3: Multiple Each() variables (zip)")
        print("=" * 60)

        @code_node
        def add(x: int, y: int):
            return {"sum": x + y}

        with ForLoopNode(
            name="add_loop",
            inputs={
                "x": Each([1, 2, 3]),
                "y": Each([10, 20, 30])
            }
        ) as loop3:
            node = add(inputs={"x": PARENT["x"], "y": PARENT["y"]}, outputs=PARENT)
            START >> node >> END

        loop3.build()

        schema3 = StateSchema(loop3)
        state3 = schema3.create_state()

        result3 = await loop3.run(state3)
        print(f"Result: {result3['sum']}")
        print(f"Expected: [11, 22, 33]")
        assert result3['sum'] == [11, 22, 33]

        # =================================================================
        # Test 4: Multiple Each() + broadcast
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 4: Multiple Each() + broadcast")
        print("=" * 60)

        @code_node
        def compute(x: int, y: int, factor: int):
            return {"result": (x + y) * factor}

        with ForLoopNode(
            name="compute_loop",
            inputs={
                "x": Each([1, 2, 3]),
                "y": Each([10, 20, 30]),
                "factor": 2  # broadcast
            }
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
        print(f"Result: {result4['result']}")
        print(f"Expected: [22, 44, 66]  # (1+10)*2, (2+20)*2, (3+30)*2")
        assert result4['result'] == [22, 44, 66]

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
            inputs={
                "name": Each(["Alice", "Bob", "Charlie"]),
                "greeting": "Hello"  # broadcast
            }
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
        print(f"Result: {result5['message']}")
        expected5 = ["Hello, Alice!", "Hello, Bob!", "Hello, Charlie!"]
        assert result5['message'] == expected5

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
            inputs={"x": Each([1, 2, 3])}
        ) as loop6:
            n1 = add_one(inputs={"x": PARENT["x"]})
            n2 = multiply_two(inputs={"y": n1["y"]}, outputs=PARENT)
            START >> n1 >> n2 >> END

        loop6.build()

        schema6 = StateSchema(loop6)
        state6 = schema6.create_state()

        result6 = await loop6.run(state6)
        print(f"Result (x+1)*2: {result6['z']}")
        print(f"Expected: [4, 6, 8]  # (1+1)*2, (2+1)*2, (3+1)*2")
        assert result6['z'] == [4, 6, 8]

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
            inputs={"value": Each([1, 2, 3, 4, 5])},
            max_concurrency=2
        ) as loop7:
            node = slow_process(inputs={"value": PARENT["value"]}, outputs=PARENT)
            START >> node >> END

        loop7.build()

        schema7 = StateSchema(loop7)
        state7 = schema7.create_state()

        result7 = await loop7.run(state7)
        print(f"Result: {result7['result']}")
        print(f"Metrics: {result7.get('iteration_metrics', {})}")
        assert result7['result'] == [10, 20, 30, 40, 50]

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
                inputs={
                    "item": Each(gen_node["numbers"]),
                    "factor": gen_node["factor"]  # broadcast Ref
                },
                outputs=PARENT
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
        # Expected: {"result": [50, 100, 150]}  # 10*5, 20*5, 30*5

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
            inputs={"a": Each([1, 2, 3])}
        ) as outer_loop9:
            # Each outer iteration creates inner loop data
            with ForLoopNode(
                name="inner_loop",
                inputs={"x": Each([10, 20])},
                outputs=PARENT
            ) as inner_loop9:
                inner_node = inner_double(inputs={"x": PARENT["x"]}, outputs=PARENT)
                START >> inner_node >> END

            START >> inner_loop9 >> END

        outer_loop9.build()

        schema9 = StateSchema(outer_loop9)
        state9 = schema9.create_state()

        result9 = await outer_loop9.run(state9)
        print(f"Result: {result9}")
        # Each outer iteration runs the same inner loop [10, 20] -> {"doubled": [20, 40]}
        print("Expected: {'doubled': [[20, 40], [20, 40], [20, 40]]}")

        # =================================================================
        # Test 10: Empty iteration
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 10: Empty iteration")
        print("=" * 60)

        with ForLoopNode(
            name="empty_loop",
            inputs={"value": Each([])}
        ) as loop10:
            node = double(inputs={"value": PARENT["value"]}, outputs=PARENT)
            START >> node >> END

        loop10.build()

        schema10 = StateSchema(loop10)
        state10 = schema10.create_state()

        result10 = await loop10.run(state10)
        print(f"Result: {result10}")
        assert result10['result'] == []
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
            inputs={
                "x": Each([1, 2, 3]),
                "y": Each([10, 20])  # Different length!
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
        # Test 12: Ref with operations in ForLoopNode
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 12: Ref with operations (Each uses Ref['key'])")
        print("=" * 60)

        @code_node
        def get_complex_data():
            return {
                "dataset": {
                    "items": [10, 20, 30],
                    "multiplier": 2
                }
            }

        @code_node
        def process_with_factor(item: int, factor: int):
            return {"result": item * factor}

        with GraphNode(name="ref_ops_for_graph") as graph12:
            data_node = get_complex_data()

            # Use Ref operations: Each uses data["dataset"]["items"]
            # broadcast uses data["dataset"]["multiplier"]
            with ForLoopNode(
                name="ref_ops_loop",
                inputs={
                    "item": Each(data_node["dataset"]["items"]),
                    "factor": data_node["dataset"]["multiplier"]  # broadcast
                },
                outputs=PARENT
            ) as loop12:
                proc = process_with_factor(
                    inputs={"item": PARENT["item"], "factor": PARENT["factor"]},
                    outputs=PARENT
                )
                START >> proc >> END

            START >> data_node >> loop12 >> END

        graph12.build()

        schema12 = StateSchema(graph12)
        state12 = schema12.create_state()

        result12 = await graph12.run(state12)
        print(f"Result: {result12}")
        print(f"Expected: [20, 40, 60]  # [10*2, 20*2, 30*2]")
        assert result12["result"] == [20, 40, 60]

        # =================================================================
        # Test 13: Ref with apply() in ForLoopNode
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 13: Ref with apply() in ForLoopNode")
        print("=" * 60)

        @code_node
        def get_nested_lists():
            return {"data": {"lists": [[1, 2], [3, 4, 5], [6]]}}

        @code_node
        def sum_list(numbers: list):
            return {"total": sum(numbers)}

        with GraphNode(name="ref_apply_for_graph") as graph13:
            data_node = get_nested_lists()

            # Each uses Ref with nested access
            with ForLoopNode(
                name="apply_loop",
                inputs={"numbers": Each(data_node["data"]["lists"])},
                outputs=PARENT
            ) as loop13:
                sum_node = sum_list(inputs={"numbers": PARENT["numbers"]}, outputs=PARENT)
                START >> sum_node >> END

            START >> data_node >> loop13 >> END

        graph13.build()

        schema13 = StateSchema(graph13)
        state13 = schema13.create_state()

        result13 = await graph13.run(state13)
        print(f"Result: {result13}")
        print(f"Expected totals: [3, 12, 6]  # sum([1,2]), sum([3,4,5]), sum([6])")
        assert result13["total"] == [3, 12, 6]

        # =================================================================
        # Test 14: Ref with apply(len) to get iteration count dynamically
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 14: Ref with chained ops + apply")
        print("=" * 60)

        @code_node
        def get_users():
            return {"response": {"users": [{"score": 10}, {"score": 20}, {"score": 30}]}}

        @code_node
        def double_score(score: int):
            return {"doubled": score * 2}

        with GraphNode(name="ref_chain_for_graph") as graph14:
            users_node = get_users()

            # Extract scores using Ref operations and apply
            # users_node["response"]["users"] -> list of dicts
            # Then we iterate over it
            with ForLoopNode(
                name="score_loop",
                inputs={"user": Each(users_node["response"]["users"])},
                outputs=PARENT
            ) as loop14:
                # Inside the loop, we access user["score"]
                @code_node
                def extract_and_double(user: dict):
                    return {"result": user["score"] * 2}

                score_node = extract_and_double(inputs={"user": PARENT["user"]}, outputs=PARENT)
                START >> score_node >> END

            START >> users_node >> loop14 >> END

        graph14.build()

        schema14 = StateSchema(graph14)
        state14 = schema14.create_state()

        result14 = await graph14.run(state14)
        print(f"Result: {result14}")
        print(f"Expected: [20, 40, 60]")
        assert result14["result"] == [20, 40, 60]

        # =================================================================
        # Test 15: ForLoopNode with PARENT broadcast inside GraphNode
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 15: ForLoopNode with PARENT broadcast inside GraphNode")
        print("=" * 60)

        @code_node
        def get_items():
            return {"items": [1, 2, 3, 4, 5]}

        @code_node
        def multiply_item(item: int, multiplier: int):
            return {"result": item * multiplier}

        with GraphNode(name="parent_broadcast_graph") as graph15:
            items_node = get_items()

            # ForLoopNode uses PARENT["multiplier"] as broadcast - should resolve to graph15
            with ForLoopNode(
                name="multiply_loop",
                inputs={
                    "item": Each(items_node["items"]),
                    "multiplier": PARENT["multiplier"]  # <-- PARENT = graph15
                },
                outputs=PARENT
            ) as loop15:
                proc = multiply_item(
                    inputs={"item": PARENT["item"], "multiplier": PARENT["multiplier"]},
                    outputs=PARENT
                )
                START >> proc >> END

            START >> items_node >> loop15 >> END

        graph15.build()

        schema15 = StateSchema(graph15)
        state15 = schema15.create_state(inputs={"multiplier": 10})

        result15 = await graph15.run(state15)
        print(f"Result: {result15}")
        print(f"Expected: [10, 20, 30, 40, 50]  # [1*10, 2*10, 3*10, 4*10, 5*10]")
        assert result15["result"] == [10, 20, 30, 40, 50]

        # =================================================================
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    asyncio.run(main())
