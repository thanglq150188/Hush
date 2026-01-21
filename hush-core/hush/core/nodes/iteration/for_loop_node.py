"""ForLoopNode - sequential iteration node for processing items one at a time."""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
from time import perf_counter
import traceback

from hush.core.configs.node_config import NodeType
from hush.core.nodes.iteration.base import BaseIterationNode, calculate_iteration_metrics
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states import MemoryState


class ForLoopNode(BaseIterationNode):
    """Sequential iteration node - processes items one at a time in order.

    Use ForLoopNode when:
        - Iterations may depend on results from previous iterations
        - You need predictable sequential execution
        - Memory constraints require processing one item at a time

    Example:
        with ForLoopNode(
            name="process_loop",
            inputs={
                "x": Each([1, 2, 3]),           # iterate
                "multiplier": 10                 # broadcast
            }
        ) as loop:
            node = calc(inputs={"x": PARENT["x"], "multiplier": PARENT["multiplier"]})
            START >> node >> END
    """

    type: NodeType = "for"

    def _post_build(self):
        """Setup inputs/outputs after graph is built."""
        self._normalize_iteration_io()

    async def run(
        self,
        state: 'MemoryState',
        context_id: Optional[str] = None,
        parent_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute for loop through iteration data sequentially."""
        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        perf_start = perf_counter()
        _inputs = {}
        _outputs = {}

        try:
            # Resolve each and broadcast values using parent_context
            each_values = self._resolve_values(self._each, state, parent_context)
            broadcast_values = self._resolve_values(self._broadcast_inputs, state, parent_context)

            # Build iteration data
            iteration_data = self._build_iteration_data(each_values, broadcast_values)
            _inputs = {**each_values, **broadcast_values}

            if not iteration_data:
                LOGGER.warning(
                    "[title]\\[%s][/title] ForLoopNode [highlight]%s[/highlight]: no iteration data.",
                    request_id, self.full_name
                )

            # Execute iterations sequentially
            latencies_ms: List[float] = []
            final_results: List[Dict[str, Any]] = []
            success_count = 0

            for i, loop_data in enumerate(iteration_data):
                iter_context = f"[{i}]" if not context_id else f"{context_id}.[{i}]"
                iter_start = perf_counter()

                try:
                    for var_name, value in loop_data.items():
                        state[self.full_name, var_name, iter_context] = value

                    result = await self._run_graph(state, iter_context, iter_context)
                    final_results.append(result)
                    success_count += 1
                except Exception as e:
                    final_results.append({"error": str(e), "error_type": type(e).__name__})

                latencies_ms.append((perf_counter() - iter_start) * 1000)

            error_count = len(iteration_data) - success_count

            # Build metrics
            iteration_metrics = calculate_iteration_metrics(latencies_ms)
            iteration_metrics.update({
                "total_iterations": len(iteration_data),
                "success_count": success_count,
                "error_count": error_count,
            })

            if iteration_data and error_count / len(iteration_data) > 0.1:
                LOGGER.warning(
                    "[title]\\[%s][/title] ForLoopNode [highlight]%s[/highlight]: high error rate [muted](%s)[/muted].",
                    request_id, self.full_name, f"{error_count / len(iteration_data):.1%}"
                )

            # Transpose results to column-oriented format
            output_keys = [k for k in self.outputs.keys() if k != "iteration_metrics"]
            _outputs = {key: [r.get(key) for r in final_results] for key in output_keys}
            _outputs["iteration_metrics"] = iteration_metrics

            self.store_result(state, _outputs, context_id)

        except Exception as e:
            error_msg = traceback.format_exc()
            LOGGER.error("[title]\\[%s][/title] Error in node [highlight]%s[/highlight]: %s", request_id, self.full_name, str(e))
            LOGGER.error(error_msg)
            state[self.full_name, "error", context_id] = error_msg

        finally:
            end_time = datetime.now()
            duration_ms = (perf_counter() - perf_start) * 1000
            self._log(request_id, context_id, _inputs, _outputs, duration_ms)
            state[self.full_name, "start_time", context_id] = start_time
            state[self.full_name, "end_time", context_id] = end_time
            return _outputs

    def specific_metadata(self) -> Dict[str, Any]:
        """Return subclass-specific metadata."""
        return {
            "each": list(self._each.keys()),
            "inputs": list(self._broadcast_inputs.keys())
        }