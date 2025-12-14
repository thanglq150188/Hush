"""Stream node for processing streaming data asynchronously."""

from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import traceback
import time

from hush.core.configs.node_config import NodeType
from hush.core.nodes.base import BaseNode
from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.states import BaseState
from hush.core.utils.context import _current_graph
from hush.core.utils.common import Param
from hush.core.streams import STREAM_SERVICE
from hush.core.loggings import LOGGER


class StreamNode(BaseNode):
    """A node that processes streaming data asynchronously while maintaining order."""

    type: NodeType = "stream"

    __slots__ = ['_graph', '_token', 'enable_metrics', 'max_chunk_traces', 'log_interval']

    def __init__(
        self,
        enable_metrics: bool = True,
        max_chunk_traces: int = 100,
        log_interval: int = 100,
        **kwargs
    ):
        input_schema = {"input_channel": Param(type=str, required=True)}
        output_schema = {
            "output_channel": Param(type=str, required=True),
            "stream_metrics": Param(type=Dict, default={})
        }

        super().__init__(
            input_schema=input_schema,
            output_schema=output_schema,
            **kwargs
        )

        self._graph: GraphNode = GraphNode(name=BaseNode.INNER_PROCESS)
        self._graph.father = self
        self._token = None
        self.enable_metrics = enable_metrics
        self.max_chunk_traces = max_chunk_traces
        self.log_interval = log_interval

    def __enter__(self):
        self._token = _current_graph.set(self._graph)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _current_graph.reset(self._token)

    def build(self):
        self._graph.build()

    def add_node(self, node: BaseNode) -> BaseNode:
        return self._graph.add_node(node)

    def add_edge(self, source: str, target: str, type=None):
        return self._graph.add_edge(source, target, type)

    def _compute_metrics(self, metrics_data: List[Dict[str, Any]], total_chunks: int) -> Dict[str, Any]:
        """Compute performance metrics from collected data."""
        if not metrics_data:
            return {}

        latencies = [m['latency_ms'] for m in metrics_data]
        success_count = sum(1 for m in metrics_data if m['success'])
        error_count = total_chunks - success_count
        sorted_latencies = sorted(latencies)

        def percentile(data: list, p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = int(k)
            c = f + 1 if f < len(data) - 1 else f
            return data[f] + (k - f) * (data[c] - data[f])

        if metrics_data:
            time_span = metrics_data[-1]['timestamp'] - metrics_data[0]['timestamp']
            throughput = total_chunks / time_span if time_span > 0 else 0
        else:
            throughput = 0

        return {
            'total_chunks': total_chunks,
            'successful_chunks': success_count,
            'failed_chunks': error_count,
            'success_rate': (success_count / total_chunks * 100) if total_chunks > 0 else 0,
            'latency_min_ms': min(latencies) if latencies else 0,
            'latency_max_ms': max(latencies) if latencies else 0,
            'latency_mean_ms': sum(latencies) / len(latencies) if latencies else 0,
            'latency_p50_ms': percentile(sorted_latencies, 0.50),
            'latency_p95_ms': percentile(sorted_latencies, 0.95),
            'latency_p99_ms': percentile(sorted_latencies, 0.99),
            'throughput_chunks_per_sec': throughput,
        }

    async def _process_chunk(
        self,
        chunk: Dict[str, Any],
        chunk_id: int,
        state: BaseState,
        request_id: str,
        metrics_data: Optional[List] = None
    ) -> Dict[str, Any]:
        """Process a single chunk through the inner graph."""
        context_id = f"stream[{chunk_id}]"
        start_time = time.perf_counter() if self.enable_metrics else None

        should_log = (self.log_interval > 0 and chunk_id % self.log_interval == 0)

        original_verbose_flags = {}
        if not should_log:
            for node in self._graph._nodes.values():
                original_verbose_flags[node.name] = node.verbose
                node.verbose = False

        try:
            # Inject chunk as input to inner graph
            for var_name, value in chunk.items():
                state[self._graph.full_name, var_name, context_id] = value

            result = await self._graph.run(state, context_id)

            if self.enable_metrics and start_time is not None and metrics_data is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                metrics_data.append({
                    'chunk_id': chunk_id,
                    'latency_ms': latency_ms,
                    'success': True,
                    'timestamp': time.time()
                })

            return {"chunk_id": chunk_id, "result": result, "success": True, "error": None}

        except Exception as e:
            error_msg = traceback.format_exc()
            LOGGER.error(f"[{request_id}] Error processing chunk {chunk_id}: {str(e)}")

            if self.enable_metrics and start_time is not None and metrics_data is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                metrics_data.append({
                    'chunk_id': chunk_id,
                    'latency_ms': latency_ms,
                    'success': False,
                    'timestamp': time.time()
                })

            return {"chunk_id": chunk_id, "result": None, "success": False, "error": error_msg}

        finally:
            if original_verbose_flags:
                for node_name, was_verbose in original_verbose_flags.items():
                    if node_name in self._graph._nodes:
                        self._graph._nodes[node_name].verbose = was_verbose

    async def run(
        self,
        state: BaseState,
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:

        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        session_id = state.session_id
        start_time = datetime.now()

        _inputs = self.get_inputs(state, context_id=context_id)
        _outputs = self.get_outputs(state, context_id=context_id)

        metrics_data = [] if self.enable_metrics else None

        try:
            input_channel = _inputs.get("input_channel")
            output_channel = _outputs.get("output_channel")

            LOGGER.info("request[%s] - NODE: %s[%s] (%s) inputs=%s",
                request_id, self.name, context_id, str(self.type).upper(), str(_inputs)[:200])

            active_tasks = {}
            completed_results = {}
            next_result_id = 0
            chunk_id = 0
            stream_ended = False

            async def consume_stream():
                nonlocal chunk_id, stream_ended
                try:
                    async for chunk in STREAM_SERVICE.get(request_id, input_channel, session_id=session_id):
                        task = asyncio.create_task(
                            self._process_chunk(chunk, chunk_id, state, request_id, metrics_data)
                        )
                        active_tasks[chunk_id] = task
                        chunk_id += 1
                except Exception as e:
                    LOGGER.error(f"[{request_id}] Stream consumption error: {e}")
                finally:
                    stream_ended = True
                    LOGGER.info(f"[{request_id}] Stream ended, total chunks: {chunk_id}")

            consumer_task = asyncio.create_task(consume_stream())

            while True:
                if stream_ended and next_result_id >= chunk_id:
                    break

                if next_result_id in active_tasks:
                    if next_result_id in completed_results:
                        result = completed_results.pop(next_result_id)
                    else:
                        result = await active_tasks[next_result_id]

                    if self.stream and result.get("success"):
                        await STREAM_SERVICE.push(
                            request_id=request_id,
                            channel_name=output_channel,
                            data=result["result"],
                            session_id=session_id
                        )
                    next_result_id += 1
                else:
                    await asyncio.sleep(0.001)
                    for task_id, task in list(active_tasks.items()):
                        if task_id > next_result_id and task.done():
                            completed_results[task_id] = await task

            await consumer_task

            LOGGER.info(f"[{request_id}] StreamNode processed {chunk_id} chunks")

            if self.enable_metrics and metrics_data:
                computed_metrics = self._compute_metrics(metrics_data, chunk_id)
                state[self.full_name, "stream_metrics", context_id] = computed_metrics
                _outputs["stream_metrics"] = computed_metrics
                LOGGER.info(
                    f"[{request_id}] Metrics: {computed_metrics['total_chunks']} chunks, "
                    f"success_rate={computed_metrics['success_rate']:.1f}%, "
                    f"latency p50={computed_metrics['latency_p50_ms']:.2f}ms, "
                    f"p95={computed_metrics['latency_p95_ms']:.2f}ms, "
                    f"p99={computed_metrics['latency_p99_ms']:.2f}ms"
                )

        except Exception as e:
            error_msg = traceback.format_exc()
            LOGGER.error(f"Error in node {self.full_name}: {str(e)}")
            LOGGER.error(error_msg)
            state[self.full_name, "error", context_id] = error_msg

        finally:
            end_time = datetime.now()
            state[self.full_name, "start_time", context_id] = start_time
            state[self.full_name, "end_time", context_id] = end_time
            self.store_result(state, _outputs, context_id=context_id)

            if self.stream:
                await STREAM_SERVICE.end(request_id, output_channel, session_id=session_id)

            LOGGER.info("request[%s] - NODE: %s[%s] (%s) outputs=%s",
                request_id, self.name, context_id, str(self.type).upper(), str(_outputs)[:200])

            return _outputs
