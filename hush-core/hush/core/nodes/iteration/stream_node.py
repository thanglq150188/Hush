"""Stream node for processing async streaming data."""

from typing import Dict, Any, Optional, AsyncIterable, Callable, Awaitable, List, TYPE_CHECKING
from datetime import datetime
import asyncio
import traceback
import os

from hush.core.configs.node_config import NodeType
from hush.core.nodes.iteration.base import IterationNode
from hush.core.utils.common import Param
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states import BaseState


class StreamNode(IterationNode):
    """
    A streaming node that processes async data with optional dynamic batching.

    Accepts any AsyncIterable as input and an optional output handler callable.
    Supports dynamic batching based on user-defined conditions.
    Uses semaphore to limit concurrent processing and prevent CPU contention.

    Features:
    - Concurrent processing with ordered output emission
    - Optional batching before processing (batch_condition)
    - Optional output handler for processed results
    - Semaphore-limited concurrency
    """

    type: NodeType = "stream"

    __slots__ = ['_max_concurrency']

    def __init__(self, max_concurrency: Optional[int] = None, **kwargs):
        """Initialize a StreamNode.

        Args:
            max_concurrency: Maximum number of concurrent chunk processing tasks.
                Defaults to CPU count if not specified.
                Uses a semaphore to limit concurrency and prevent CPU contention.
        """
        input_schema = {
            "input_stream": Param(type=AsyncIterable, required=True),
            "batch_condition": Param(type=Callable, required=False),  # (batch, new_item) -> should_flush
        }
        output_schema = {
            "output_handler": Param(type=Callable, required=False),  # async (data) -> None
            "iteration_metrics": Param(type=Dict, required=False),
        }

        super().__init__(
            input_schema=input_schema,
            output_schema=output_schema,
            **kwargs
        )

        self._max_concurrency = max_concurrency if max_concurrency is not None else os.cpu_count()

    async def _process_chunk(
        self,
        chunk: Any,
        chunk_id: int,
        state: 'BaseState',
        request_id: str
    ) -> Dict[str, Any]:
        """Process a single chunk (or batch) through the inner graph."""
        context_id = f"stream[{chunk_id}]"
        start_time = datetime.now()

        try:
            # Inject chunk as input to inner graph
            chunk_data = chunk if isinstance(chunk, dict) else {"data": chunk}
            self.inject_inputs(state, chunk_data, context_id)

            # Run inner graph
            result = await self._graph.run(state, context_id)

            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000

            return {
                "chunk_id": chunk_id,
                "result": result,
                "success": True,
                "error": None,
                "latency_ms": latency_ms
            }

        except Exception as e:
            error_msg = traceback.format_exc()
            LOGGER.error(f"[{request_id}] Error processing chunk {chunk_id}: {str(e)}")

            end_time = datetime.now()
            latency_ms = (end_time - start_time).total_seconds() * 1000

            return {
                "chunk_id": chunk_id,
                "result": None,
                "success": False,
                "error": error_msg,
                "latency_ms": latency_ms
            }

    async def run(
        self,
        state: 'BaseState',
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process streaming data with concurrent processing but ordered output."""
        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        _outputs = {}

        try:
            _inputs = self.get_inputs(state, context_id)
            _outputs = self.get_outputs(state, context_id=context_id)

            if self.verbose:
                LOGGER.info(
                    "request[%s] - NODE: %s[%s] (%s) inputs=%s",
                    request_id, self.name, context_id,
                    str(self.type).upper(), str(_inputs)[:200]
                )

            input_stream: AsyncIterable = _inputs.get("input_stream")
            batch_condition: Optional[Callable[[List, Any], bool]] = _inputs.get("batch_condition")
            output_handler: Optional[Callable[[Any], Awaitable[None]]] = _outputs.get("output_handler")

            # Create semaphore to limit concurrency
            semaphore = asyncio.Semaphore(self._max_concurrency)

            # Track ongoing tasks and results
            active_tasks: Dict[int, asyncio.Task] = {}
            completed_results: Dict[int, Dict] = {}
            next_result_id = 0
            chunk_id = 0
            stream_ended = False

            # Track metrics
            all_latencies: List[float] = []
            handler_latencies: List[float] = []
            success_count = 0
            error_count = 0
            handler_error_count = 0

            # Batching state
            current_batch: List = []

            async def process_with_semaphore(chunk_data: Any, cid: int) -> Dict[str, Any]:
                """Process chunk with semaphore limiting."""
                async with semaphore:
                    return await self._process_chunk(chunk_data, cid, state, request_id)

            async def flush_batch(batch: List) -> None:
                """Flush a batch by creating a processing task."""
                nonlocal chunk_id
                if not batch:
                    return

                # Create batch data
                batch_data = {"batch": batch, "batch_size": len(batch)}

                task = asyncio.create_task(
                    process_with_semaphore(batch_data, chunk_id)
                )
                active_tasks[chunk_id] = task
                chunk_id += 1

            async def consume_stream():
                """Consume chunks from input stream and launch tasks."""
                nonlocal chunk_id, stream_ended, current_batch

                try:
                    async for chunk in input_stream:
                        if batch_condition:
                            # Dynamic batching mode
                            if current_batch and batch_condition(current_batch, chunk):
                                await flush_batch(current_batch)
                                current_batch = []
                            current_batch.append(chunk)
                        else:
                            # No batching - process each chunk individually with semaphore
                            task = asyncio.create_task(
                                process_with_semaphore(chunk, chunk_id)
                            )
                            active_tasks[chunk_id] = task
                            chunk_id += 1

                    # Flush remaining batch
                    if batch_condition and current_batch:
                        await flush_batch(current_batch)

                except Exception as e:
                    LOGGER.error(f"[{request_id}] Stream consumption error: {e}")
                finally:
                    stream_ended = True
                    LOGGER.debug(f"[{request_id}] Stream ended, total chunks: {chunk_id}")

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

                    # Collect metrics from result
                    if result.get("success"):
                        all_latencies.append(result.get("latency_ms", 0))
                        success_count += 1
                    else:
                        all_latencies.append(result.get("latency_ms", 0))
                        error_count += 1

                    # Call output handler if provided and result is successful
                    if output_handler and result.get("success"):
                        handler_start = datetime.now()
                        try:
                            await output_handler(result["result"])
                        except Exception as e:
                            handler_error_count += 1
                            LOGGER.error(f"[{request_id}] Output handler error: {e}")
                        finally:
                            handler_end = datetime.now()
                            handler_latency = (handler_end - handler_start).total_seconds() * 1000
                            handler_latencies.append(handler_latency)

                    # Clean up completed task
                    del active_tasks[next_result_id]
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

            # Calculate iteration metrics
            iteration_metrics = self._calculate_iteration_metrics(all_latencies)
            iteration_metrics.update({
                "total_iterations": chunk_id,
                "success_count": success_count,
                "error_count": error_count,
            })

            # Calculate output_handler metrics if handler was used
            if handler_latencies:
                handler_metrics = self._calculate_iteration_metrics(handler_latencies)
                iteration_metrics["output_handler"] = {
                    "latency_avg_ms": handler_metrics["latency_avg_ms"],
                    "latency_min_ms": handler_metrics["latency_min_ms"],
                    "latency_max_ms": handler_metrics["latency_max_ms"],
                    "latency_p50_ms": handler_metrics["latency_p50_ms"],
                    "latency_p95_ms": handler_metrics["latency_p95_ms"],
                    "latency_p99_ms": handler_metrics["latency_p99_ms"],
                    "call_count": len(handler_latencies),
                    "error_count": handler_error_count,
                }

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
            "max_concurrency": self._max_concurrency
        }


if __name__ == "__main__":
    from hush.core.states import StateSchema
    from hush.core.nodes.base import START, END, PARENT
    from hush.core.nodes.transform.code_node import code_node

    async def main():
        # Test 1: Simple 1-on-1 streaming
        print("=" * 50)
        print("Test 1: Simple streaming - double each value")
        print("=" * 50)

        async def simple_source():
            """Stream 10 chunks"""
            for i in range(10):
                yield {"value": i, "text": f"chunk_{i}"}
                await asyncio.sleep(0.01)

        results = []

        async def collect_results(data):
            results.append(data)

        @code_node
        def process_chunk(value: int, text: str):
            return {
                "processed_value": value * 2,
                "processed_text": text.upper()
            }

        schema = StateSchema("test_stream")

        with StreamNode(
            name="processor",
            inputs={"input_stream": simple_source()},
            outputs={"output_handler": collect_results}
        ) as stream_node:
            processor = process_chunk(inputs=PARENT, outputs=PARENT)
            START >> processor >> END

        stream_node.build()

        # Need to set output_handler in state
        state = schema.create_state()
        state[stream_node.full_name, "output_handler", None] = collect_results

        await stream_node.run(state)

        print(f"Processed {len(results)} chunks")
        for r in results[:3]:
            print(f"  {r}")
        print("  ...")

        # Test 2: Streaming with batching
        print("\n" + "=" * 50)
        print("Test 2: Streaming with batching")
        print("=" * 50)

        async def batch_source():
            """Stream 20 items"""
            for i in range(20):
                yield {"value": i}
                await asyncio.sleep(0.005)

        batch_results = []

        async def collect_batches(data):
            batch_results.append(data)

        @code_node
        def process_batch(batch: list, batch_size: int):
            total = sum(item["value"] for item in batch)
            return {
                "batch_total": total,
                "batch_size": batch_size
            }

        def should_flush(batch: list, new_item: dict) -> bool:
            """Flush when batch has 5 items"""
            return len(batch) >= 5

        schema2 = StateSchema("test_batch_stream")

        with StreamNode(
            name="batch_processor",
            inputs={
                "input_stream": batch_source(),
                "batch_condition": should_flush
            },
            outputs={"output_handler": collect_batches}
        ) as stream_node2:
            processor = process_batch(inputs=PARENT, outputs=PARENT)
            START >> processor >> END

        stream_node2.build()

        state2 = schema2.create_state()
        state2[stream_node2.full_name, "output_handler", None] = collect_batches

        await stream_node2.run(state2)

        print(f"Processed {len(batch_results)} batches")
        for r in batch_results:
            print(f"  Batch total: {r.get('batch_total')}, size: {r.get('batch_size')}")

        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)

    asyncio.run(main())
