"""Async iteration node for processing async streaming data."""

from typing import Dict, Any, Optional, AsyncIterable, Callable, Awaitable, TYPE_CHECKING, List
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
    from hush.core.states import BaseState


def batch_by_size(n: int) -> Callable[[List, Any], bool]:
    """Create a batch condition that flushes when batch reaches n items."""
    return lambda batch, _: len(batch) >= n


class AsyncIterNode(BaseIterationNode):
    """
    A streaming node that processes async iterable data with optional batching.

    API (unified inputs with Each wrapper):
        - `inputs`: Dict of all variables. Use Each() wrapper for the async iterable source.
          - Each(source): The async iterable to iterate over
          - Regular values: Broadcast to all iterations (same value)
        - `callback`: Optional async handler for streaming results
        - `batch_fn`: Optional batching condition

    Features:
    - Concurrent processing with ordered output emission
    - Optional batching before processing
    - Optional callback for streaming results
    - Semaphore-limited concurrency
    - Collects results like ForLoopNode (unless callback-only mode)

    Example:
        async def my_stream():
            for i in range(10):
                yield {"value": i}
                await asyncio.sleep(0.01)

        with AsyncIterNode(
            inputs={
                "item": Each(my_stream()),    # async iterable source
                "multiplier": 10               # broadcast
            },
            callback=handle_result
        ) as stream_node:
            processor = process(
                inputs={"item": PARENT["item"], "multiplier": PARENT["multiplier"]},
                outputs=PARENT
            )
            START >> processor >> END
    """

    type: NodeType = "stream"

    __slots__ = [
        '_max_concurrency', '_each', '_broadcast_inputs',
        '_callback', '_batch_fn', '_raw_outputs', '_raw_inputs'
    ]

    def __init__(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[Any], Awaitable[None]]] = None,
        batch_fn: Optional[Callable[[List, Any], bool]] = None,
        max_concurrency: Optional[int] = None,
        **kwargs
    ):
        """Initialize an AsyncIterNode.

        Args:
            inputs: Dict mapping variable names to values or Each(source).
                    - Each(source): The async iterable to iterate over (exactly one required)
                    - Other values: Broadcast to all iterations
                    Values can be literals or Refs to upstream nodes.
            callback: Optional async function called with each result in order.
                      If provided, results are streamed; otherwise collected.
            batch_fn: Optional function (batch, new_item) -> should_flush.
                      When batching, inner graph receives {"batch": [...], "batch_size": n}.
            max_concurrency: Maximum concurrent processing tasks.
                            Defaults to CPU count.
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

        # Validate exactly one Each() source
        if len(self._each) != 1:
            raise ValueError(
                f"AsyncIterNode requires exactly one Each() source in inputs, got: {list(self._each.keys())}"
            )

        self._callback = callback
        self._batch_fn = batch_fn
        self._max_concurrency = max_concurrency if max_concurrency is not None else os.cpu_count()

    def _post_build(self):
        """Set input/output schema after inner graph is built.

        IMPORTANT: This method normalizes _each and _broadcast_inputs using
        _normalize_connections to resolve PARENT refs to self.father.
        """
        # Normalize each and broadcast inputs (resolves PARENT refs)
        self._each = self._normalize_connections(self._each, {})
        self._broadcast_inputs = self._normalize_connections(self._broadcast_inputs, {})

        # Get the iteration variable name
        iter_var_name = list(self._each.keys())[0]

        # Input schema: iteration var (AsyncIterable) + broadcast inputs (Any)
        self.input_schema = {
            iter_var_name: Param(type=AsyncIterable, required=True)
        }
        self.input_schema.update({
            var_name: Param(type=Any, required=isinstance(value, Ref))
            for var_name, value in self._broadcast_inputs.items()
        })

        # Output schema: from inner graph (as lists) + metrics
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

    def _resolve_broadcast_values(
        self,
        state: 'BaseState',
        context_id: Optional[str]
    ) -> Dict[str, Any]:
        """Resolve broadcast input values, dereferencing any Refs."""
        result = {}
        for var_name, value in self._broadcast_inputs.items():
            if isinstance(value, Ref):
                resolved = state._get_value(value.node, value.var, context_id)
                if value.has_ops:
                    resolved = value.execute(resolved)
                result[var_name] = resolved
            else:
                result[var_name] = value
        return result

    def _get_source(self, state: 'BaseState', context_id: Optional[str]) -> AsyncIterable:
        """Get the async iterable source, resolving Ref if needed."""
        iter_var_name = list(self._each.keys())[0]
        source = self._each[iter_var_name]

        if isinstance(source, Ref):
            resolved = state._get_value(source.node, source.var, context_id)
            if source.has_ops:
                resolved = source.execute(resolved)
            return resolved
        return source

    async def _process_chunk(
        self,
        chunk_data: Dict[str, Any],
        chunk_id: int,
        state: 'BaseState',
        request_id: str
    ) -> Dict[str, Any]:
        """Process a single chunk through the inner graph."""
        context_id = f"stream[{chunk_id}]"
        start = perf_counter()

        try:
            self.inject_inputs(state, chunk_data, context_id)
            result = await self._graph.run(state, context_id)
            return {
                "chunk_id": chunk_id,
                "result": result,
                "success": True,
                "latency_ms": (perf_counter() - start) * 1000
            }
        except Exception as e:
            LOGGER.error(f"[{request_id}] Error processing chunk {chunk_id}: {e}")
            return {
                "chunk_id": chunk_id,
                "result": None,
                "success": False,
                "error": str(e),
                "latency_ms": (perf_counter() - start) * 1000
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
        _inputs = {}
        _outputs = {}

        try:
            # Get source and broadcast values
            source = self._get_source(state, context_id)
            broadcast_values = self._resolve_broadcast_values(state, context_id)
            iter_var_name = list(self._each.keys())[0]

            # Store inputs for logging
            _inputs = {iter_var_name: "<AsyncIterable>", **broadcast_values}

            if source is None:
                LOGGER.warning(f"AsyncIterNode '{self.full_name}': source is None.")

            semaphore = asyncio.Semaphore(self._max_concurrency)
            result_queue: asyncio.Queue = asyncio.Queue()

            # Metrics tracking
            latencies: List[float] = []
            handler_latencies: List[float] = []
            success_count = 0
            error_count = 0
            handler_error_count = 0
            total_chunks = 0

            # Collected results (if not streaming via callback)
            collected_results: List[Dict[str, Any]] = []

            async def process_with_semaphore(chunk_data: Dict[str, Any], cid: int):
                async with semaphore:
                    result = await self._process_chunk(chunk_data, cid, state, request_id)
                    await result_queue.put((cid, result))

            async def consume_stream():
                """Consume stream and launch processing tasks."""
                nonlocal total_chunks
                current_batch: List = []
                tasks = []

                async for item in source:
                    if self._batch_fn:
                        # Batching mode
                        if current_batch and self._batch_fn(current_batch, item):
                            # Flush batch
                            batch_data = {
                                "batch": current_batch,
                                "batch_size": len(current_batch),
                                **broadcast_values
                            }
                            tasks.append(asyncio.create_task(
                                process_with_semaphore(batch_data, total_chunks)
                            ))
                            total_chunks += 1
                            current_batch = []
                        current_batch.append(item)
                    else:
                        # Normal mode: each item becomes iteration variable
                        chunk_data = {iter_var_name: item, **broadcast_values}
                        tasks.append(asyncio.create_task(
                            process_with_semaphore(chunk_data, total_chunks)
                        ))
                        total_chunks += 1

                # Flush remaining batch
                if self._batch_fn and current_batch:
                    batch_data = {
                        "batch": current_batch,
                        "batch_size": len(current_batch),
                        **broadcast_values
                    }
                    tasks.append(asyncio.create_task(
                        process_with_semaphore(batch_data, total_chunks)
                    ))
                    total_chunks += 1

                # Wait for all tasks and signal completion
                if tasks:
                    await asyncio.gather(*tasks)
                await result_queue.put(None)  # Sentinel

            # Start consumer
            consumer_task = asyncio.create_task(consume_stream())

            # Collect results in order
            pending_results: Dict[int, Dict] = {}
            next_id = 0

            while True:
                item = await result_queue.get()
                if item is None:
                    break

                cid, result = item
                pending_results[cid] = result

                # Emit results in order
                while next_id in pending_results:
                    result = pending_results.pop(next_id)
                    latencies.append(result.get("latency_ms", 0))

                    if result.get("success"):
                        success_count += 1

                        # Stream to callback if provided
                        if self._callback:
                            handler_start = perf_counter()
                            try:
                                await self._callback(result["result"])
                            except Exception as e:
                                handler_error_count += 1
                                LOGGER.error(f"[{request_id}] Callback error: {e}")
                            handler_latencies.append((perf_counter() - handler_start) * 1000)

                        # Collect result
                        collected_results.append(result["result"])
                    else:
                        error_count += 1
                        collected_results.append({"error": result.get("error")})

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
                    f"AsyncIterNode '{self.full_name}': high error rate "
                    f"({error_count / total_chunks:.1%}). {error_count}/{total_chunks} failed."
                )

            # Add callback metrics if used
            if handler_latencies:
                handler_metrics = self._calculate_iteration_metrics(handler_latencies)
                iteration_metrics["callback"] = {
                    **handler_metrics,
                    "call_count": len(handler_latencies),
                    "error_count": handler_error_count,
                }
                if handler_error_count / len(handler_latencies) > 0.05:
                    LOGGER.warning(
                        f"AsyncIterNode '{self.full_name}': high callback error rate "
                        f"({handler_error_count / len(handler_latencies):.1%})."
                    )

            # Transpose results to column-oriented format (like ForLoopNode)
            output_keys = list(self._graph.output_schema.keys())
            _outputs = {
                key: [r.get(key) for r in collected_results]
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
            "inputs": list(self._broadcast_inputs.keys()),
            "has_callback": self._callback is not None,
            "has_batch_fn": self._batch_fn is not None,
        }


if __name__ == "__main__":
    from hush.core.states import StateSchema
    from hush.core.nodes.base import START, END, PARENT
    from hush.core.nodes.transform.code_node import code_node
    from hush.core.nodes.graph.graph_node import GraphNode

    async def main():
        # =================================================================
        # Test 1: Simple streaming - double each value
        # =================================================================
        print("=" * 60)
        print("Test 1: Simple streaming - double each value")
        print("=" * 60)

        async def simple_source():
            for i in range(10):
                yield i
                await asyncio.sleep(0.01)

        results = []

        async def collect_results(data):
            results.append(data)

        @code_node
        def double_value(value: int):
            return {"result": value * 2}

        with AsyncIterNode(
            name="double_stream",
            inputs={"value": Each(simple_source())},
            callback=collect_results
        ) as stream_node:
            processor = double_value(
                inputs={"value": PARENT["value"]},
                outputs=PARENT
            )
            START >> processor >> END

        stream_node.build()

        schema = StateSchema(stream_node)
        state = schema.create_state()

        output = await stream_node.run(state)

        print(f"Callback received {len(results)} results: {results[:5]}...")
        print(f"Collected output: {output['result'][:5]}...")
        print(f"Metrics: {output['iteration_metrics']}")
        assert output['result'] == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

        # =================================================================
        # Test 2: Streaming with broadcast inputs
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 2: Streaming with broadcast inputs")
        print("=" * 60)

        async def number_source():
            for i in range(5):
                yield i + 1
                await asyncio.sleep(0.01)

        @code_node
        def multiply(value: int, multiplier: int):
            return {"result": value * multiplier}

        with AsyncIterNode(
            name="multiply_stream",
            inputs={
                "value": Each(number_source()),
                "multiplier": 10  # broadcast
            }
        ) as stream_node2:
            processor = multiply(
                inputs={"value": PARENT["value"], "multiplier": PARENT["multiplier"]},
                outputs=PARENT
            )
            START >> processor >> END

        stream_node2.build()

        schema2 = StateSchema(stream_node2)
        state2 = schema2.create_state()

        output2 = await stream_node2.run(state2)

        print(f"Result: {output2['result']}")
        print(f"Expected: [10, 20, 30, 40, 50]")
        assert output2['result'] == [10, 20, 30, 40, 50]

        # =================================================================
        # Test 3: Streaming with batching
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 3: Streaming with batching")
        print("=" * 60)

        async def batch_source():
            for i in range(12):
                yield i
                await asyncio.sleep(0.005)

        @code_node
        def sum_batch(batch: list, batch_size: int):
            return {"total": sum(batch), "size": batch_size}

        with AsyncIterNode(
            name="batch_stream",
            inputs={"item": Each(batch_source())},
            batch_fn=batch_by_size(4)
        ) as stream_node3:
            processor = sum_batch(
                inputs={"batch": PARENT["batch"], "batch_size": PARENT["batch_size"]},
                outputs=PARENT
            )
            START >> processor >> END

        stream_node3.build()

        schema3 = StateSchema(stream_node3)
        state3 = schema3.create_state()

        output3 = await stream_node3.run(state3)

        print(f"Totals: {output3['total']}")
        print(f"Sizes: {output3['size']}")
        print(f"Expected totals: [6, 22, 38] (0+1+2+3, 4+5+6+7, 8+9+10+11)")
        assert output3['total'] == [6, 22, 38]
        assert output3['size'] == [4, 4, 4]

        # =================================================================
        # Test 4: With Ref (dynamic config from upstream)
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 4: With Ref (dynamic config from upstream)")
        print("=" * 60)

        @code_node
        def get_config():
            return {"factor": 5, "offset": 100}

        async def ref_source():
            for i in range(4):
                yield i + 1
                await asyncio.sleep(0.01)

        @code_node
        def compute(value: int, factor: int, offset: int):
            return {"result": value * factor + offset}

        with GraphNode(name="ref_test") as graph4:
            config_node = get_config()

            with AsyncIterNode(
                name="compute_stream",
                inputs={
                    "value": Each(ref_source()),
                    "factor": config_node["factor"],   # broadcast Ref
                    "offset": config_node["offset"]    # broadcast Ref
                },
                outputs=PARENT
            ) as stream_node4:
                processor = compute(
                    inputs={
                        "value": PARENT["value"],
                        "factor": PARENT["factor"],
                        "offset": PARENT["offset"]
                    },
                    outputs=PARENT
                )
                START >> processor >> END

            START >> config_node >> stream_node4 >> END

        graph4.build()

        schema4 = StateSchema(graph4)
        state4 = schema4.create_state()

        output4 = await graph4.run(state4)

        print(f"Result: {output4['result']}")
        print(f"Expected: [105, 110, 115, 120] (1*5+100, 2*5+100, ...)")
        assert output4['result'] == [105, 110, 115, 120]

        # =================================================================
        # Test 5: Dict items in stream
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 5: Dict items in stream")
        print("=" * 60)

        async def dict_source():
            users = [
                {"name": "Alice", "score": 85},
                {"name": "Bob", "score": 92},
                {"name": "Charlie", "score": 78},
            ]
            for user in users:
                yield user
                await asyncio.sleep(0.01)

        @code_node
        def grade_user(user: dict):
            grade = "A" if user["score"] >= 90 else "B" if user["score"] >= 80 else "C"
            return {"name": user["name"], "grade": grade}

        with AsyncIterNode(
            name="grade_stream",
            inputs={"user": Each(dict_source())}
        ) as stream_node5:
            processor = grade_user(
                inputs={"user": PARENT["user"]},
                outputs=PARENT
            )
            START >> processor >> END

        stream_node5.build()

        schema5 = StateSchema(stream_node5)
        state5 = schema5.create_state()

        output5 = await stream_node5.run(state5)

        print(f"Names: {output5['name']}")
        print(f"Grades: {output5['grade']}")
        assert output5['name'] == ["Alice", "Bob", "Charlie"]
        assert output5['grade'] == ["B", "A", "C"]

        # =================================================================
        # Test 6: Concurrency limit
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 6: Concurrency limit (max_concurrency=2)")
        print("=" * 60)

        async def slow_source():
            for i in range(6):
                yield i
                await asyncio.sleep(0.01)

        @code_node
        async def slow_process(value: int):
            await asyncio.sleep(0.05)
            return {"result": value * 10}

        with AsyncIterNode(
            name="concurrent_stream",
            inputs={"value": Each(slow_source())},
            max_concurrency=2
        ) as stream_node6:
            processor = slow_process(
                inputs={"value": PARENT["value"]},
                outputs=PARENT
            )
            START >> processor >> END

        stream_node6.build()

        schema6 = StateSchema(stream_node6)
        state6 = schema6.create_state()

        output6 = await stream_node6.run(state6)

        print(f"Result: {output6['result']}")
        print(f"Metrics: {output6['iteration_metrics']}")
        assert output6['result'] == [0, 10, 20, 30, 40, 50]

        # =================================================================
        # Test 7: AsyncIterNode receives source from upstream node output
        # =================================================================
        print("\n" + "=" * 60)
        print("Test 7: AsyncIterNode receives source from upstream node")
        print("=" * 60)

        # This node creates and returns an async generator as its output
        @code_node
        def create_stream_source(start: int, end: int):
            """Node that produces an async iterable as output."""
            async def generated_stream():
                for i in range(start, end):
                    yield {"id": i, "value": i * 10}
                    await asyncio.sleep(0.01)
            return {"stream": generated_stream(), "metadata": {"count": end - start}}

        @code_node
        def process_item(item: dict, prefix: str):
            """Process each streamed item."""
            return {
                "processed_id": f"{prefix}_{item['id']}",
                "doubled_value": item["value"] * 2
            }

        with GraphNode(name="dynamic_stream_graph") as graph7:
            # Upstream node creates the async iterable
            source_creator = create_stream_source(
                inputs={"start": PARENT["start"], "end": PARENT["end"]}
            )

            # AsyncIterNode receives the stream via Ref from upstream node
            with AsyncIterNode(
                name="dynamic_processor",
                inputs={
                    "item": Each(source_creator["stream"]),  # <-- Ref to upstream output!
                    "prefix": PARENT["prefix"]               # broadcast from graph input
                },
                outputs=PARENT
            ) as stream_node7:
                processor = process_item(
                    inputs={
                        "item": PARENT["item"],
                        "prefix": PARENT["prefix"]
                    },
                    outputs=PARENT
                )
                START >> processor >> END

            START >> source_creator >> stream_node7 >> END

        graph7.build()

        schema7 = StateSchema(graph7)
        state7 = schema7.create_state(inputs={"start": 0, "end": 5, "prefix": "MSG"})

        output7 = await graph7.run(state7)

        print(f"Processed IDs: {output7['processed_id']}")
        print(f"Doubled values: {output7['doubled_value']}")
        print(f"Expected IDs: ['MSG_0', 'MSG_1', 'MSG_2', 'MSG_3', 'MSG_4']")
        print(f"Expected values: [0, 20, 40, 60, 80]")
        assert output7['processed_id'] == ['MSG_0', 'MSG_1', 'MSG_2', 'MSG_3', 'MSG_4']
        assert output7['doubled_value'] == [0, 20, 40, 60, 80]

        # =================================================================
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)

    asyncio.run(main())
