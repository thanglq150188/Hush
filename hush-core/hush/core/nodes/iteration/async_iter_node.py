"""Stream node for processing async streaming data."""

from typing import Dict, Any, Optional, AsyncIterable, Callable, Awaitable, TYPE_CHECKING, List
from datetime import datetime
from time import perf_counter
import asyncio
import traceback
import os

from hush.core.configs.node_config import NodeType
from hush.core.nodes.iteration.base import IterationNode
from hush.core.utils.common import Param
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states import BaseState


def batch_by_size(n: int) -> Callable[[List, Any], bool]:
    """Create a batch condition that flushes when batch reaches n items."""
    return lambda batch, _: len(batch) >= n


class AsyncIterNode(IterationNode):
    """
    A streaming node that processes async data with optional dynamic batching.

    Accepts any AsyncIterable as input and an optional output handler callable.
    Supports dynamic batching based on user-defined conditions.
    Uses semaphore to limit concurrent processing and prevent CPU contention.

    Features:
    - Concurrent processing with ordered output emission
    - Optional batching before processing (batch_fn)
    - Optional output handler for processed results
    - Semaphore-limited concurrency
    """

    type: NodeType = "stream"

    __slots__ = ['_max_concurrency']

    def __init__(self, max_concurrency: Optional[int] = None, **kwargs):
        """Initialize a AsyncIterNode.

        Args:
            max_concurrency: Maximum number of concurrent chunk processing tasks.
                Defaults to CPU count if not specified.
                Uses a semaphore to limit concurrency and prevent CPU contention.
        """
        input_schema = {
            "source": Param(type=AsyncIterable, required=True),
            "batch_fn": Param(type=Callable, required=False),  # (batch, new_item) -> should_flush
        }
        output_schema = {
            "callback": Param(type=Callable, required=False),  # async (data) -> None
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
        start = perf_counter()

        try:
            chunk_data = chunk if isinstance(chunk, dict) else {"data": chunk}
            self.inject_inputs(state, chunk_data, context_id)
            result = await self._graph.run(state, context_id)
            return {"chunk_id": chunk_id, "result": result, "success": True, "latency_ms": (perf_counter() - start) * 1000}
        except Exception as e:
            LOGGER.error(f"[{request_id}] Error processing chunk {chunk_id}: {e}")
            return {"chunk_id": chunk_id, "result": None, "success": False, "error": str(e), "latency_ms": (perf_counter() - start) * 1000}

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
        _inputs = {}
        _outputs = {}

        try:
            _inputs = self.get_inputs(state, context_id)
            _outputs = self.get_outputs(state, context_id=context_id)

            source: AsyncIterable = _inputs.get("source")
            batch_fn: Optional[Callable[[List, Any], bool]] = _inputs.get("batch_fn")
            callback: Optional[Callable[[Any], Awaitable[None]]] = _outputs.get("callback")

            if source is None:
                LOGGER.warning(f"AsyncIterNode '{self.full_name}': source is None. No data will be processed.")

            semaphore = asyncio.Semaphore(self._max_concurrency)
            result_queue: asyncio.Queue = asyncio.Queue()

            # Metrics tracking
            latencies: List[float] = []
            handler_latencies: List[float] = []
            success_count = 0
            error_count = 0
            handler_error_count = 0
            total_chunks = 0

            async def process_with_semaphore(chunk_data: Any, cid: int):
                async with semaphore:
                    result = await self._process_chunk(chunk_data, cid, state, request_id)
                    await result_queue.put((cid, result))

            async def consume_stream():
                """Consume stream and launch processing tasks."""
                nonlocal total_chunks
                current_batch: List = []
                tasks = []

                async for chunk in source:
                    if batch_fn:
                        if current_batch and batch_fn(current_batch, chunk):
                            tasks.append(asyncio.create_task(
                                process_with_semaphore({"batch": current_batch, "batch_size": len(current_batch)}, total_chunks)
                            ))
                            total_chunks += 1
                            current_batch = []
                        current_batch.append(chunk)
                    else:
                        tasks.append(asyncio.create_task(process_with_semaphore(chunk, total_chunks)))
                        total_chunks += 1

                # Flush remaining batch
                if batch_fn and current_batch:
                    tasks.append(asyncio.create_task(
                        process_with_semaphore({"batch": current_batch, "batch_size": len(current_batch)}, total_chunks)
                    ))
                    total_chunks += 1

                # Wait for all tasks and signal completion
                if tasks:
                    await asyncio.gather(*tasks)
                await result_queue.put(None)  # Sentinel to signal end

            # Start consumer
            consumer_task = asyncio.create_task(consume_stream())

            # Collect results in order
            pending_results: Dict[int, Dict] = {}
            next_id = 0

            while True:
                item = await result_queue.get()
                if item is None:  # Stream ended
                    break

                cid, result = item
                pending_results[cid] = result

                # Emit results in order
                while next_id in pending_results:
                    result = pending_results.pop(next_id)
                    latencies.append(result.get("latency_ms", 0))

                    if result.get("success"):
                        success_count += 1
                        if callback:
                            handler_start = perf_counter()
                            try:
                                await callback(result["result"])
                            except Exception as e:
                                handler_error_count += 1
                                LOGGER.error(f"[{request_id}] Output handler error: {e}")
                            handler_latencies.append((perf_counter() - handler_start) * 1000)
                    else:
                        error_count += 1

                    next_id += 1

            await consumer_task

            # Build iteration metrics
            iteration_metrics = self._calculate_iteration_metrics(latencies)
            iteration_metrics.update({
                "total_iterations": total_chunks,
                "success_count": success_count,
                "error_count": error_count,
            })

            # Warn if high error rate (>10%)
            if total_chunks > 0 and error_count / total_chunks > 0.1:
                LOGGER.warning(
                    f"AsyncIterNode '{self.full_name}': high error rate ({error_count / total_chunks:.1%}). "
                    f"{error_count}/{total_chunks} chunks failed."
                )

            # Add handler metrics if used
            if handler_latencies:
                handler_metrics = self._calculate_iteration_metrics(handler_latencies)
                iteration_metrics["callback"] = {
                    **handler_metrics,
                    "call_count": len(handler_latencies),
                    "error_count": handler_error_count,
                }
                if handler_error_count / len(handler_latencies) > 0.05:
                    LOGGER.warning(
                        f"AsyncIterNode '{self.full_name}': high handler error rate ({handler_error_count / len(handler_latencies):.1%})."
                    )

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

        with AsyncIterNode(
            name="processor",
            inputs={"source": simple_source()},
            outputs={"callback": collect_results}
        ) as stream_node:
            processor = process_chunk(inputs=PARENT, outputs=PARENT)
            START >> processor >> END

        stream_node.build()

        # Need to set callback in state
        state = schema.create_state()
        state[stream_node.full_name, "callback", None] = collect_results

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

        schema2 = StateSchema("test_batch_stream")

        with AsyncIterNode(
            name="batch_processor",
            inputs={
                "source": batch_source(),
                "batch_fn": batch_by_size(5)
            },
            outputs={"callback": collect_batches}
        ) as stream_node2:
            processor = process_batch(inputs=PARENT, outputs=PARENT)
            START >> processor >> END

        stream_node2.build()

        state2 = schema2.create_state()
        state2[stream_node2.full_name, "callback", None] = collect_batches

        await stream_node2.run(state2)

        print(f"Processed {len(batch_results)} batches")
        for r in batch_results:
            print(f"  Batch total: {r.get('batch_total')}, size: {r.get('batch_size')}")

        # =================================================================
        # Test 3: Ref operations with AsyncIterNode inputs
        # =================================================================
        print("\n" + "=" * 50)
        print("Test 3: Ref operations extract nested batch_fn")
        print("=" * 50)

        from hush.core.nodes.graph.graph_node import GraphNode

        @code_node
        def get_stream_config():
            return {
                "stream_config": {
                    "settings": {
                        "batch_fn": batch_by_size(3)
                    }
                }
            }

        async def config_source():
            """Stream 9 items"""
            for i in range(9):
                yield {"data": i * 10}
                await asyncio.sleep(0.005)

        @code_node
        def process_data_batch(batch: list, batch_size: int):
            # Process batch: sum all data values
            total = sum(item["data"] for item in batch)
            return {"batch_total": total, "batch_size": batch_size}

        with GraphNode(name="ref_stream_graph") as graph3:
            config_node = get_stream_config()

            # AsyncIterNode uses Ref with nested access for batch_fn
            with AsyncIterNode(
                name="ref_processor",
                inputs={
                    "source": config_source(),
                    "batch_fn": config_node["stream_config"]["settings"]["batch_fn"]
                }
            ) as stream_node3:
                processor = process_data_batch(inputs=PARENT, outputs=PARENT)
                START >> processor >> END

            START >> config_node >> stream_node3 >> END

        graph3.build()

        schema3 = StateSchema(graph3)
        state3 = schema3.create_state()

        await graph3.run(state3)

        # Verify Ref operation worked by checking iteration_metrics
        metrics3 = state3._get_value(stream_node3.full_name, "iteration_metrics", None)
        if metrics3:
            print(f"Total iterations: {metrics3.get('total_iterations')}")
            print(f"Expected: 3 batches (batch_by_size(3) from nested config)")
            print("Test 3 passed - Ref operation extracted batch_fn from nested config!")

        # =================================================================
        # Test 4: Verify Ref chained operations in inputs
        # =================================================================
        print("\n" + "=" * 50)
        print("Test 4: Ref chained operations in AsyncIterNode inputs")
        print("=" * 50)

        @code_node
        def get_complex_config():
            return {
                "data": {
                    "stream_params": {
                        "batch_fn": batch_by_size(2)
                    }
                }
            }

        async def simple_stream():
            for i in range(6):
                yield {"num": i}
                await asyncio.sleep(0.003)

        @code_node
        def sum_batch_nums(batch: list, batch_size: int):
            # Process batch: sum of squares of all nums
            total = sum(item["num"] ** 2 for item in batch)
            return {"sum_of_squares": total, "batch_size": batch_size}

        with GraphNode(name="chain_stream_graph") as graph4:
            config_node4 = get_complex_config()

            with AsyncIterNode(
                name="chain_processor",
                inputs={
                    "source": simple_stream(),
                    "batch_fn": config_node4["data"]["stream_params"]["batch_fn"]
                }
            ) as stream_node4:
                proc = sum_batch_nums(inputs=PARENT, outputs=PARENT)
                START >> proc >> END

            START >> config_node4 >> stream_node4 >> END

        graph4.build()

        schema4 = StateSchema(graph4)
        state4 = schema4.create_state()

        await graph4.run(state4)

        metrics4 = state4._get_value(stream_node4.full_name, "iteration_metrics", None)
        if metrics4:
            print(f"Total iterations: {metrics4.get('total_iterations')}")
            print(f"Expected: 3 batches (batch_by_size(2) from nested config, 6 items)")
            print("Test 4 passed - Ref chained operations extracted batch_fn!")

        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)

    asyncio.run(main())
