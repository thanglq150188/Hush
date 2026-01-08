"""While loop node for iterating while a condition is true."""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
from time import perf_counter
import traceback

from hush.core.configs.node_config import NodeType
from hush.core.nodes.iteration.base import BaseIterationNode
from hush.core.utils.common import Param, extract_condition_variables
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states import MemoryState


class WhileLoopNode(BaseIterationNode):
    """A node that iterates while a condition is true.

    The loop continues while `stop_condition` evaluates to False.
    When `stop_condition` becomes True, the loop stops.

    Example:
        with WhileLoopNode(
            name="loop",
            inputs={"counter": 0},
            stop_condition="counter >= 5"
        ) as loop:
            node = process(
                inputs={"counter": PARENT["counter"]},
                outputs={"new_counter": PARENT["counter"]}
            )
            START >> node >> END
    """

    type: NodeType = "while"

    __slots__ = ['_max_iterations', '_stop_condition', '_compiled_condition']

    def __init__(
        self,
        stop_condition: Optional[str] = None,
        max_iterations: int = 100,
        **kwargs
    ):
        """Initialize a WhileLoopNode.

        Args:
            stop_condition: A string expression that is evaluated each iteration.
                When it evaluates to True, the loop stops.
                Example: "counter >= 5" or "total > 100 and done"
            max_iterations: Maximum iterations to prevent infinite loops.
                Defaults to 100.
        """
        super().__init__(**kwargs)
        self._max_iterations = max_iterations
        self._stop_condition = stop_condition
        self._compiled_condition = self._compile_condition(stop_condition) if stop_condition else None

    def _compile_condition(self, condition: str):
        """Compile the stop condition for performance."""
        try:
            return compile(condition, f'<stop_condition: {condition}>', 'eval')
        except SyntaxError as e:
            LOGGER.error(f"Invalid stop_condition syntax '{condition}': {e}")
            raise ValueError(f"Invalid stop_condition syntax: {condition}") from e

    def _evaluate_stop_condition(self, inputs: Dict[str, Any]) -> bool:
        """Evaluate the stop condition against current inputs.

        Returns:
            True if the loop should stop, False to continue.
        """
        if self._compiled_condition is None:
            return False  # No condition = never stop (rely on max_iterations)

        try:
            result = eval(self._compiled_condition, {"__builtins__": {}}, inputs)
            return bool(result)
        except Exception as e:
            LOGGER.error(f"Error evaluating stop_condition '{self._stop_condition}': {e}")
            return False  # On error, continue (let max_iterations be the safety)

    def _post_build(self):
        """Set input/output schema from inner graph after build."""
        self.input_schema = (self._graph.input_schema or {}).copy()
        self.output_schema = (self._graph.output_schema or {}).copy()

        # Ensure inputs in self.inputs are in schema
        for key in self.inputs:
            if key not in self.input_schema:
                self.input_schema[key] = Param(type=Any, required=True)

        # Add variables from stop_condition to input schema
        if self._stop_condition:
            for var_name in extract_condition_variables(self._stop_condition):
                if var_name not in self.input_schema:
                    self.input_schema[var_name] = Param(type=Any, required=False)

        # Add iteration_metrics to output schema
        self.output_schema.setdefault("iteration_metrics", Param(type=Dict, required=False))

    async def run(
        self,
        state: 'MemoryState',
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the while loop until stop_condition is True or max_iterations reached."""
        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        _inputs = {}
        _outputs = {}

        try:
            _inputs = self.get_inputs(state, context_id)
            step_inputs = _inputs
            latencies_ms: List[float] = []
            step_count = 0

            # Check stop condition before first iteration
            should_stop = self._evaluate_stop_condition(step_inputs)

            while not should_stop and step_count < self._max_iterations:
                step_name = f"while-{step_count}"
                iter_start = perf_counter()

                self.inject_inputs(state, step_inputs, step_name)
                _outputs = await self._graph.run(state, context_id=step_name)

                latencies_ms.append((perf_counter() - iter_start) * 1000)

                # Merge outputs with previous inputs to preserve unchanged variables
                step_inputs = {**step_inputs, **_outputs}
                step_count += 1

                # Check stop condition after iteration
                should_stop = self._evaluate_stop_condition(step_inputs)

            # Warn if max_iterations was reached (potential infinite loop)
            if step_count >= self._max_iterations and not should_stop:
                LOGGER.warning(
                    f"WhileLoopNode '{self.full_name}': max_iterations ({self._max_iterations}) reached. "
                    f"Condition '{self._stop_condition}' never evaluated to True. "
                    "This may indicate an infinite loop or incorrect stop condition."
                )

            # Calculate iteration metrics (all completed iterations succeeded, errors propagate)
            iteration_metrics = self._calculate_iteration_metrics(latencies_ms)
            iteration_metrics.update({
                "total_iterations": step_count,
                "success_count": step_count,
                "error_count": 0,
                "max_iterations_reached": step_count >= self._max_iterations,
                "stopped_by_condition": should_stop,
            })

            # Add iteration_metrics to outputs
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
            "max_iterations": self._max_iterations,
            "stop_condition": self._stop_condition
        }


if __name__ == "__main__":
    import asyncio
    from hush.core.states import StateSchema, MemoryState
    from hush.core.nodes.base import START, END, PARENT
    from hush.core.nodes.transform.code_node import code_node

    def test(name: str, condition: bool):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        if not condition:
            raise AssertionError(f"Test failed: {name}")

    async def main():
        # =====================================================================
        # Test 1: Simple counter loop (count to 5)
        # =====================================================================
        print("\n" + "=" * 50)
        print("Test 1: Simple counter loop (stop when counter >= 5)")
        print("=" * 50)

        @code_node
        def increment(counter: int):
            new_counter = counter + 1
            return {"new_counter": new_counter}

        with WhileLoopNode(
            name="counter_loop",
            inputs={"counter": 0},
            stop_condition="counter >= 5",
            max_iterations=10
        ) as loop1:
            node = increment(
                inputs={"counter": PARENT["counter"]},
                outputs={"new_counter": PARENT["counter"]}
            )
            START >> node >> END

        loop1.build()

        schema1 = StateSchema(loop1)

        schema1.show()

        state1 = MemoryState(schema1)

        result = await loop1.run(state1)

        test("iterations = 5", result.get('iteration_metrics', {}).get('total_iterations') == 5)
        test("stopped by condition", result.get('iteration_metrics', {}).get('stopped_by_condition') == True)

        # # =====================================================================
        # # Test 2: Accumulator loop (sum until >= 100)
        # # =====================================================================
        # print("\n" + "=" * 50)
        # print("Test 2: Accumulator loop (stop when total >= 100)")
        # print("=" * 50)

        # @code_node
        # def accumulate(total: int, step: int):
        #     new_total = total + step
        #     return {"new_total": new_total}

        # with WhileLoopNode(
        #     name="accumulator_loop",
        #     inputs={"total": 0, "step": 15},
        #     max_iterations=10,
        #     stop_condition="total >= 100"
        # ) as loop2:
        #     node = accumulate(
        #         inputs={"total": PARENT["total"], "step": PARENT["step"]},
        #         outputs={"new_total": PARENT["total"]}
        #     )
        #     START >> node >> END

        # loop2.build()

        # schema2 = StateSchema(loop2)
        # state2 = MemoryState(schema2)

        # result2 = await loop2.run(state2)
        # test("iterations = 7 (0+15*7=105)", result2.get('iteration_metrics', {}).get('total_iterations') == 7)

        # # =====================================================================
        # # Test 3: Max iterations safety (no stop condition)
        # # =====================================================================
        # print("\n" + "=" * 50)
        # print("Test 3: Max iterations safety (max_iterations=5, no stop_condition)")
        # print("=" * 50)

        # @code_node
        # def infinite_loop(value: int):
        #     new_value = value + 1
        #     return {"new_value": new_value}

        # with WhileLoopNode(
        #     name="safe_loop",
        #     max_iterations=5,
        #     inputs={"value": 0}
        # ) as loop3:
        #     node = infinite_loop(
        #         inputs={"value": PARENT["value"]},
        #         outputs={"new_value": PARENT["value"]}
        #     )
        #     START >> node >> END

        # loop3.build()

        # schema3 = StateSchema(loop3)
        # state3 = MemoryState(schema3)

        # result3 = await loop3.run(state3)
        # test("max iterations reached", result3.get('iteration_metrics', {}).get('max_iterations_reached') == True)

        # # =====================================================================
        # # Test 4: Complex condition with multiple variables
        # # =====================================================================
        # print("\n" + "=" * 50)
        # print("Test 4: Complex condition (stop when x >= 10 or done == True)")
        # print("=" * 50)

        # @code_node
        # def complex_step(x: int, done: bool):
        #     new_x = x + 3
        #     new_done = new_x > 8
        #     return {"new_x": new_x, "new_done": new_done}

        # with WhileLoopNode(
        #     name="complex_loop",
        #     inputs={"x": 0, "done": False},
        #     stop_condition="x >= 10 or done"
        # ) as loop4:
        #     node = complex_step(
        #         inputs={"x": PARENT["x"], "done": PARENT["done"]},
        #         outputs={"new_x": PARENT["x"], "new_done": PARENT["done"]}
        #     )
        #     START >> node >> END

        # loop4.build()

        # schema4 = StateSchema(loop4)
        # state4 = MemoryState(schema4)

        # result4 = await loop4.run(state4)
        # # 0->3->6->9 (done=True at 9>8), so 3 iterations before condition triggers
        # test("complex condition works", result4.get('iteration_metrics', {}).get('total_iterations') == 3)

        # # =====================================================================
        # # Test 5: WhileLoop in GraphNode with Ref inputs
        # # =====================================================================
        # print("\n" + "=" * 50)
        # print("Test 5: WhileLoop in GraphNode with Ref inputs")
        # print("=" * 50)

        # from hush.core.nodes.graph.graph_node import GraphNode

        # @code_node
        # def get_config():
        #     return {
        #         "config": {
        #             "settings": {
        #                 "start_value": 0,
        #                 "increment": 3,
        #                 "limit": 10
        #             }
        #         }
        #     }

        # @code_node
        # def increment_by(counter: int, step: int):
        #     new_counter = counter + step
        #     return {"new_counter": new_counter}

        # with GraphNode(name="ref_while_graph") as graph5:
        #     config_node = get_config()

        #     with WhileLoopNode(
        #         name="ref_loop",
        #         inputs={
        #             "counter": config_node["config"]["settings"]["start_value"],
        #             "step": config_node["config"]["settings"]["increment"]
        #         },
        #         stop_condition="counter >= 10"
        #     ) as loop5:
        #         node = increment_by(
        #             inputs={"counter": PARENT["counter"], "step": PARENT["step"]},
        #             outputs={"new_counter": PARENT["counter"]}
        #         )
        #         START >> node >> END

        #     START >> config_node >> loop5 >> END

        # graph5.build()

        # schema5 = StateSchema(graph5)
        # state5 = MemoryState(schema5)

        # await graph5.run(state5)

        # loop_result = state5[loop5.full_name, "iteration_metrics", None]
        # iterations5 = loop_result.get('total_iterations', 0) if loop_result else 0
        # test("Ref inputs work (4 iterations)", iterations5 == 4)

        # # =====================================================================
        # # Test 6: Schema extraction
        # # =====================================================================
        # print("\n" + "=" * 50)
        # print("Test 6: Schema extraction")
        # print("=" * 50)

        # test("loop1 has 'counter' in input_schema", "counter" in loop1.input_schema)
        # test("loop1 has 'iteration_metrics' in output_schema", "iteration_metrics" in loop1.output_schema)

        # # =====================================================================
        # # Summary
        # # =====================================================================
        # print("\n" + "=" * 50)
        # print("All WhileLoopNode tests passed!")
        # print("=" * 50)

    asyncio.run(main())
