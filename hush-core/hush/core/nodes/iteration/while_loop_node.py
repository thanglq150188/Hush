"""While loop node for iterating while a condition is true."""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
import traceback

from hush.core.configs.node_config import NodeType
from hush.core.nodes.iteration.base import IterationNode
from hush.core.utils.common import Param
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states import BaseState


class WhileLoopNode(IterationNode):
    """A node that iterates while a condition is true.

    The loop continues while `continue_loop=True`.

    Example:
        with WhileLoopNode(name="loop", inputs={"counter": 0}) as loop:
            node = process(
                inputs={"counter": INPUT["counter"]},
                outputs={"new_counter": INPUT["counter"]}
            )
            check = BranchNode(
                name="check",
                cases={"new_counter < 5": CONTINUE},
                default=END,
                inputs={"new_counter": node["new_counter"]}
            )
            START >> node >> check >> [CONTINUE, END]
    """

    type: NodeType = "while"

    __slots__ = ['_max_iterations']

    def __init__(self, max_iterations: int = 100, **kwargs):
        """Initialize a WhileLoopNode.

        Args:
            max_iterations: Maximum iterations to prevent infinite loops.
                Defaults to 100.
        """
        super().__init__(**kwargs)
        self._max_iterations = max_iterations

    def _post_build(self):
        """Set input/output schema from inner graph after build."""
        self.input_schema = self._graph.input_schema.copy() if self._graph.input_schema else {}
        self.output_schema = self._graph.output_schema.copy() if self._graph.output_schema else {}
        # self.output_schema["continue_loop"] = Param(type=bool, required=False, default=True)


        # Ensure inputs in self.inputs are in schema
        for key in self.inputs:
            if key not in self.input_schema:
                self.input_schema[key] = Param(type=Any, required=True)

        # Add iteration_metrics to output schema
        if "iteration_metrics" not in self.output_schema:
            self.output_schema["iteration_metrics"] = Param(type=Dict, required=False)

    async def run(
        self,
        state: 'BaseState',
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute the while loop until continue_loop is False."""
        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        _outputs = {}

        try:
            _inputs = self.get_inputs(state, context_id)

            if self.verbose:
                LOGGER.info(
                    "request[%s] - NODE: %s[%s] (%s) inputs=%s",
                    request_id, self.name, context_id,
                    str(self.type).upper(), str(_inputs)[:200]
                )

            _continue = True
            step_count = 0
            step_inputs = _inputs

            # Track latencies for metrics
            latencies_ms: List[float] = []
            success_count = 0
            error_count = 0

            while _continue and step_count < self._max_iterations:
                step_name = f"while-{step_count}"
                iter_start_time = datetime.now()

                self.inject_inputs(state, step_inputs, step_name)

                try:
                    _outputs = await self._graph.run(state, context_id=step_name)
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    raise e
                finally:
                    iter_end_time = datetime.now()
                    latency = (iter_end_time - iter_start_time).total_seconds() * 1000
                    latencies_ms.append(latency)

                # Get updated inputs for next iteration
                step_inputs = self._graph.get_inputs(state, context_id=step_name)
                step_count += 1

                # Check continue_loop from graph outputs (set by CONTINUE node)
                _continue = _outputs.get("continue_loop", True)
                # if _continue is None:
                #     _continue = True  # Default to True if not set

            # Calculate iteration metrics
            iteration_metrics = self._calculate_iteration_metrics(latencies_ms)
            iteration_metrics.update({
                "total_iterations": step_count,
                "success_count": success_count,
                "error_count": error_count,
                "max_iterations_reached": step_count >= self._max_iterations,
            })

            # Add iteration_metrics to outputs
            _outputs["iteration_metrics"] = iteration_metrics

            if self.verbose:
                LOGGER.info(
                    "request[%s] - NODE: %s[%s] (%s) outputs=%s",
                    request_id, self.name, context_id,
                    str(self.type).upper(), str(_outputs)[:200]
                )

            self.store_result(state, _outputs, context_id)

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

    def specific_metadata(self) -> Dict[str, Any]:
        """Return subclass-specific metadata."""
        return {
            "max_iterations": self._max_iterations
        }


if __name__ == "__main__":
    import asyncio
    from hush.core.states import StateSchema
    from hush.core.nodes.base import START, END, INPUT, OUTPUT
    from hush.core.nodes.transform.code_node import code_node
    from hush.core.nodes.flow.branch_node import BranchNode

    async def main():
        # Test 1: Simple counter loop (count to 5)
        print("=" * 50)
        print("Test 1: Simple counter loop (count to 5)")
        print("=" * 50)

        @code_node
        def increment(counter: int):
            new_counter = counter + 1
            return {"new_counter": new_counter}
        
        @code_node
        def stop_loop():
            return {
                "continue_loop": False # (bool = True)
            }

        with WhileLoopNode(
            name="counter_loop",
            inputs={"counter": 0, "continue_loop": True}
        ) as loop1:
            node = increment(
                inputs={"counter": INPUT["counter"]},
                outputs={"new_counter": INPUT["counter"]}
            )
            check = BranchNode(
                name="check",
                cases={"new_counter < 5": END},
                default="stop_loop",
                inputs={"new_counter": node["new_counter"]}
            )

            _stop = stop_loop(
                outputs={"continue_loop": OUTPUT}
            )

            START >> node >> check >> [_stop, END]
            _stop >> END

        loop1.build()

        schema1 = StateSchema(loop1)

        state1 = schema1.create_state()

        print(f"Input schema: {loop1._graph.input_schema}")
        print(f"Output schema: {loop1._graph.output_schema}")
        schema1.show_links()
        schema1.show_defaults()

        result = await loop1.run(state1)

        state1.show()
        # print(f"Result: {result}")
        # print(f"Iterations: {result.get('iteration_metrics', {}).get('total_iterations')}")

        # # Test 2: Accumulator loop (sum until >= 100)
        # print("\n" + "=" * 50)
        # print("Test 2: Accumulator loop (sum until >= 100)")
        # print("=" * 50)

        # @code_node
        # def accumulate(total: int, step: int):
        #     new_total = total + step
        #     return {"new_total": new_total, "new_step": step}

        # with WhileLoopNode(
        #     name="accumulator_loop",
        #     inputs={"total": 0, "step": 15}
        # ) as loop2:
        #     node = accumulate(
        #         inputs={"total": INPUT["total"], "step": INPUT["step"]},
        #         outputs={"new_total": INPUT["total"], "new_step": INPUT["step"]}
        #     )
        #     check = BranchNode(
        #         name="check",
        #         cases={"new_total < 100": CONTINUE},
        #         default=END,
        #         inputs={"new_total": node["new_total"]}
        #     )
        #     START >> node >> check >> [CONTINUE, END]

        # loop2.build()

        # schema2 = StateSchema(loop2)
        # state2 = schema2.create_state()

        # result2 = await loop2.run(state2)
        # print(f"Result: {result2}")
        # print(f"Iterations: {result2.get('iteration_metrics', {}).get('total_iterations')}")

        # # Test 3: Max iterations safety (always continues, limited by max_iterations)
        # print("\n" + "=" * 50)
        # print("Test 3: Max iterations safety (max_iterations=5)")
        # print("=" * 50)

        # @code_node
        # def infinite_loop(value: int):
        #     return {"new_value": value + 1}

        # with WhileLoopNode(
        #     name="safe_loop",
        #     max_iterations=5,
        #     inputs={"value": 0}
        # ) as loop3:
        #     node = infinite_loop(
        #         inputs={"value": INPUT["value"]},
        #         outputs={"new_value": INPUT["value"]}
        #     )
        #     # Always continue - max_iterations will stop it
        #     START >> node >> CONTINUE

        # loop3.build()

        # schema3 = StateSchema(loop3)
        # state3 = schema3.create_state()

        # result3 = await loop3.run(state3)
        # print(f"Result: {result3}")
        # print(f"Max iterations reached: {result3.get('iteration_metrics', {}).get('max_iterations_reached')}")

        # # Test 4: Chain of nodes in loop (double until >= 1000)
        # print("\n" + "=" * 50)
        # print("Test 4: Chain of nodes in loop (double until >= 1000)")
        # print("=" * 50)

        # @code_node
        # def double_value(x: int):
        #     return {"doubled": x * 2}

        # @code_node
        # def update_x(doubled: int):
        #     return {"new_x": doubled}

        # with WhileLoopNode(
        #     name="chain_loop",
        #     inputs={"x": 1}
        # ) as loop4:
        #     n1 = double_value(inputs={"x": INPUT["x"]})
        #     n2 = update_x(
        #         inputs={"doubled": n1["doubled"]},
        #         outputs={"new_x": INPUT["x"]}
        #     )
        #     check = BranchNode(
        #         name="check",
        #         cases={"new_x < 1000": CONTINUE},
        #         default=END,
        #         inputs={"new_x": n2["new_x"]}
        #     )
        #     START >> n1 >> n2 >> check >> [CONTINUE, END]

        # loop4.build()

        # schema4 = StateSchema(loop4)
        # state4 = schema4.create_state()

        # result4 = await loop4.run(state4)
        # print(f"Result: {result4}")
        # print(f"Iterations: {result4.get('iteration_metrics', {}).get('total_iterations')}")

        # print("\n" + "=" * 50)
        # print("All tests passed!")
        # print("=" * 50)

    asyncio.run(main())
