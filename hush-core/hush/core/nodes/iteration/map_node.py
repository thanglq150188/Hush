"""MapNode - parallel iteration node applying function to each item in a collection."""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
from time import perf_counter
import asyncio
import traceback
import os

from hush.core.configs.node_config import NodeType
from hush.core.nodes.iteration.base import BaseIterationNode, get_iter_context
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states import MemoryState


class MapNode(BaseIterationNode):
    """Parallel iteration node - applies function to each item concurrently.

    Use MapNode when:
        - Items are independent and can be processed in parallel
        - Order of execution doesn't matter (results are collected in order)
        - You want maximum throughput for I/O-bound operations

    Example:
        with MapNode(
            name="process_map",
            inputs={
                "x": Each([1, 2, 3]),           # iterate
                "multiplier": 10                 # broadcast
            },
            max_concurrency=4
        ) as map_node:
            node = calc(inputs={"x": PARENT["x"], "multiplier": PARENT["multiplier"]})
            START >> node >> END
    """

    type: NodeType = "map"

    __slots__ = ['_max_concurrency']

    def __init__(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        max_concurrency: Optional[int] = None,
        **kwargs
    ):
        """Initialize MapNode.

        Args:
            inputs: Dict mapping variable names to values or Each(source).
            max_concurrency: Max concurrent tasks. Defaults to CPU count.
        """
        super().__init__(inputs=inputs, **kwargs)
        self._max_concurrency = max_concurrency if max_concurrency is not None else os.cpu_count()

    def _post_build(self):
        """Setup inputs/outputs after graph is built."""
        self._normalize_iteration_io()

    async def run(
        self,
        state: 'MemoryState',
        context_id: Optional[str] = None,
        parent_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute map loop through iteration data in parallel."""
        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        perf_start = perf_counter()
        _inputs = {}
        _outputs = {}

        try:
            each_values = self._resolve_values(self._each, state, parent_context)
            broadcast_values = self._resolve_values(self._broadcast_inputs, state, parent_context)

            iteration_data = self._build_iteration_data(each_values, broadcast_values)
            _inputs = {**each_values, **broadcast_values}

            if not iteration_data:
                LOGGER.warning(
                    "[title]\\[%s][/title] MapNode [highlight]%s[/highlight]: no iteration data.",
                    request_id, self.full_name
                )

            semaphore = asyncio.Semaphore(self._max_concurrency)

            async def execute_iteration(iter_context: str, loop_data: Dict[str, Any]) -> Dict[str, Any]:
                try:
                    async with semaphore:
                        for var_name, value in loop_data.items():
                            state[self.full_name, var_name, iter_context] = value
                        result = await self._run_graph(state, iter_context, iter_context)
                    return {"result": result, "success": True}
                except Exception as e:
                    return {"result": {"error": str(e), "error_type": type(e).__name__}, "success": False}

            ctx_prefix = (context_id + ".") if context_id else ""
            raw_results = await asyncio.gather(*[
                execute_iteration(get_iter_context(ctx_prefix, i), data)
                for i, data in enumerate(iteration_data)
            ])

            # Extract metrics and results
            final_results = []
            success_count = 0
            for r in raw_results:
                final_results.append(r["result"])
                success_count += r["success"]
            error_count = len(raw_results) - success_count

            iteration_metrics = {
                "total_iterations": len(iteration_data),
                "success_count": success_count,
                "error_count": error_count,
            }

            if iteration_data and error_count / len(iteration_data) > 0.1:
                LOGGER.warning(
                    "[title]\\[%s][/title] MapNode [highlight]%s[/highlight]: high error rate [muted](%s)[/muted].",
                    request_id, self.full_name, f"{error_count / len(iteration_data):.1%}"
                )

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

            # Record trace metadata for observability
            state.record_trace_metadata(
                node_name=self.full_name,
                context_id=context_id,
                name=self.name,
                input_vars=list(self.inputs.keys()) if self.inputs else [],
                output_vars=list(self.outputs.keys()) if self.outputs else [],
                parent_name=parent_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                contain_generation=False,
                metadata=self.metadata(),
            )
            return _outputs

    def specific_metadata(self) -> Dict[str, Any]:
        """Return subclass-specific metadata."""
        return {
            "max_concurrency": self._max_concurrency,
            "each": list(self._each.keys()),
            "inputs": list(self._broadcast_inputs.keys())
        }