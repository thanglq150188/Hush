from typing import Dict, Any, Optional, List
from hush.core.configs.node_config import NodeType
from hush.core.nodes.base import BaseNode
from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.states.workflow_state import WorkflowState
from hush.core.utils.context import _current_graph
from hush.core.streams import STREAM_SERVICE, STREAM_SERVICE
from hush.core.schema import ParamSet
from hush.core.loggings import LOGGER
import asyncio
import traceback
import time
from datetime import datetime



class StreamNode(BaseNode):
    """A node that processes streaming data asynchronously while maintaining order"""

    type: NodeType = "stream"

    input_schema: ParamSet = (
        ParamSet.new()
            .var("input_channel: str", required=True)
            .build()
    )

    output_schema: ParamSet = (
        ParamSet.new()
            .var("output_channel: str", required=True)
            .var("stream_metrics: Dict[str, Any] = {}")
            .build()
    )

    def __init__(self, enable_metrics: bool = True, max_chunk_traces: int = 100, log_interval: int = 100, **kwargs):
        """Initialize StreamNode

        Args:
            enable_metrics: Enable detailed performance metrics tracking (default: True)
            max_chunk_traces: Maximum number of chunk traces to send to Langfuse (default: 100)
            log_interval: Log every N chunks (default: 100). Set to 0 to disable chunk logging.
        """
        super().__init__(**kwargs)
        self._graph: GraphNode = GraphNode(name=BaseNode.INNER_PROCESS)
        self._graph.father = self
        self._token = None
        self.enable_metrics = enable_metrics
        self.max_chunk_traces = max_chunk_traces
        self.log_interval = log_interval

    def __enter__(self):
        """Enter context manager mode"""
        self._token = _current_graph.set(self._graph)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        _current_graph.reset(self._token)

    def build(self):
        self._graph.build()
        self._post_init()

    def add_node(self, node: BaseNode) -> BaseNode:
        """Delegate node addition to inner graph"""
        return self._graph.add_node(node)

    def add_edge(self, source: str, target: str, type=None):
        """Delegate edge addition to inner graph"""
        return self._graph.add_edge(source, target, type)

    def _compute_metrics(self, metrics_data: List[Dict[str, Any]], total_chunks: int) -> Dict[str, Any]:
        """Compute performance metrics from collected data (called once at end)"""
        if not metrics_data:
            return {}

        # Extract latencies
        latencies = [m['latency_ms'] for m in metrics_data]
        success_count = sum(1 for m in metrics_data if m['success'])
        error_count = total_chunks - success_count

        # Sort for percentile calculation (done once)
        sorted_latencies = sorted(latencies)

        # Calculate percentiles
        def percentile(data: list, p: float) -> float:
            if not data:
                return 0.0
            k = (len(data) - 1) * p
            f = int(k)
            c = f + 1 if f < len(data) - 1 else f
            return data[f] + (k - f) * (data[c] - data[f])

        # Calculate throughput
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
        state: WorkflowState,
        request_id: str,
        metrics_data: Optional[List] = None
    ) -> Dict[str, Any]:
        """Process a single chunk through the inner graph"""
        context_id = f"stream[{chunk_id}]"

        # Start timing (minimal overhead)
        start_time = time.perf_counter() if self.enable_metrics else None

        # Control logging for inner graph nodes based on log_interval
        should_log = (self.log_interval > 0 and chunk_id % self.log_interval == 0)

        # Temporarily set verbose flag for all nodes in inner graph
        original_verbose_flags = {}
        if not should_log:
            for node in self._graph._nodes.values():
                original_verbose_flags[node.name] = node.verbose
                node.verbose = False

        try:
            # Inject chunk as input to inner graph
            state.inject_inputs(
                node=self._graph.full_name,
                inputs=chunk,
                context_id=context_id
            )

            # Run inner graph
            result = await self._graph.run(state, context_id)

            # Record metrics (only if enabled)
            if self.enable_metrics and start_time is not None and metrics_data is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                metrics_data.append({
                    'chunk_id': chunk_id,
                    'latency_ms': latency_ms,
                    'success': True,
                    'timestamp': time.time()
                })

            return {
                "chunk_id": chunk_id,
                "result": result,
                "success": True,
                "error": None
            }

        except Exception as e:
            error_msg = traceback.format_exc()
            LOGGER.error(f"[{request_id}] Error processing chunk {chunk_id}: {str(e)}")

            # Record error metrics
            if self.enable_metrics and start_time is not None and metrics_data is not None:
                latency_ms = (time.perf_counter() - start_time) * 1000
                metrics_data.append({
                    'chunk_id': chunk_id,
                    'latency_ms': latency_ms,
                    'success': False,
                    'timestamp': time.time()
                })

            return {
                "chunk_id": chunk_id,
                "result": None,
                "success": False,
                "error": error_msg
            }

        finally:
            # Restore verbose flags
            if original_verbose_flags:
                for node_name, was_verbose in original_verbose_flags.items():
                    if node_name in self._graph._nodes:
                        self._graph._nodes[node_name].verbose = was_verbose

    async def run(
        self,
        state: WorkflowState,
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:

        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        session_id = state.session_id
        start_time = datetime.now()

        _inputs = self.get_inputs(state, context_id=context_id)
        _outputs = self.get_outputs(state, context_id=context_id)

        # Initialize metrics collection (lightweight list append)
        metrics_data = [] if self.enable_metrics else None

        try:

            input_channel = _inputs.get("input_channel")
            output_channel = _outputs.get("output_channel")

            LOGGER.info("request[%s] - running NODE: %s[%s] (%s), inputs = %s",
                    request_id, self.name, context_id, str(self.type).upper(), str(_inputs)[:200])

            # Track ongoing tasks and results
            active_tasks = {}
            completed_results = {}
            next_result_id = 0
            chunk_id = 0
            stream_ended = False

            async def consume_stream():
                """Consume chunks from STREAM_SERVICE and launch tasks"""
                nonlocal chunk_id, stream_ended

                try:
                    async for chunk in STREAM_SERVICE.get(request_id, input_channel, session_id=session_id):
                        # Launch async processing task
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

            # Start consuming stream
            consumer_task = asyncio.create_task(consume_stream())

            # Emit results in order as they complete
            while True:
                # Check if we're done
                if stream_ended and next_result_id >= chunk_id:
                    break

                # Wait for next result in sequence
                if next_result_id in active_tasks:
                    # If already completed, use cached result
                    if next_result_id in completed_results:
                        result = completed_results.pop(next_result_id)
                    else:
                        # Wait for this specific task
                        result = await active_tasks[next_result_id]

                    # Stream output to channel if enabled
                    if self.stream and result.get("success"):
                        await STREAM_SERVICE.push(
                            request_id=request_id,
                            channel_name=output_channel,
                            data=result["result"],
                            session_id=session_id
                        )
                    next_result_id += 1

                else:
                    # Wait for more chunks to arrive
                    await asyncio.sleep(0.001)

                    # Check completed tasks ahead of sequence
                    for task_id, task in list(active_tasks.items()):
                        if task_id > next_result_id and task.done():
                            completed_results[task_id] = await task

            # Wait for consumer to finish
            await consumer_task

            LOGGER.info(f"[{request_id}] StreamNode processed {chunk_id} chunks")

            # Compute and store metrics (only if enabled and we have data)
            if self.enable_metrics and metrics_data:
                computed_metrics = self._compute_metrics(metrics_data, chunk_id)
                state.set_by_index(self.output_indexes["stream_metrics"], computed_metrics, context_id)
                # Update _outputs with the computed metrics
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
            await asyncio.sleep(0.000001)
            end_time = datetime.now()
            state.set_by_index(self.metrics['start_time'], start_time, context_id=context_id)
            state.set_by_index(self.metrics['end_time'], end_time, context_id=context_id)
            self.store_result(state, _outputs, context_id=context_id)

            if self.stream:
                await STREAM_SERVICE.end(request_id, output_channel, session_id=session_id)

            LOGGER.info("request[%s] - running NODE: %s[%s] (%s), outputs = %s",
                    request_id, self.name, context_id, str(self.type).upper(), str(_outputs)[:200])

            return _outputs
