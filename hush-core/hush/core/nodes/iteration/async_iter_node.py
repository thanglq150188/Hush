"""Async iteration node để xử lý async streaming data."""

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
    from hush.core.states import MemoryState


def batch_by_size(n: int) -> Callable[[List, Any], bool]:
    """Tạo batch condition để flush khi batch đạt n items."""
    return lambda batch, _: len(batch) >= n


class AsyncIterNode(BaseIterationNode):
    """Streaming node xử lý async iterable data với optional batching.

    API (unified inputs với Each wrapper):
        - `inputs`: Dict tất cả variables. Dùng Each() wrapper cho async iterable source.
          - Each(source): Async iterable để iterate
          - Regular values: Broadcast cho tất cả iterations (cùng value)
        - `callback`: Optional async handler cho streaming results
        - `batch_fn`: Optional batching condition

    Features:
    - Concurrent processing với ordered output emission
    - Optional batching trước khi processing
    - Optional callback cho streaming results
    - Semaphore-limited concurrency
    - Collect results giống ForLoopNode (trừ khi callback-only mode)

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
        """Khởi tạo AsyncIterNode.

        Args:
            inputs: Dict mapping variable names đến values hoặc Each(source).
                    - Each(source): Async iterable để iterate (yêu cầu đúng một)
                    - Other values: Broadcast cho tất cả iterations
                    Values có thể là literals hoặc Refs đến upstream nodes.
            callback: Optional async function được gọi với mỗi result theo thứ tự.
                      Nếu có, results được stream; ngược lại được collect.
            batch_fn: Optional function (batch, new_item) -> should_flush.
                      Khi batching, inner graph nhận {"batch": [...], "batch_size": n}.
            max_concurrency: Số concurrent processing tasks tối đa.
                            Mặc định là CPU count.
        """
        # Lưu raw outputs và inputs trước khi super().__init__ normalize chúng
        self._raw_outputs = kwargs.get('outputs')
        self._raw_inputs = inputs or {}

        # Không pass inputs cho parent - tự xử lý
        super().__init__(**kwargs)

        # Tách Each() sources khỏi broadcast inputs
        self._each = {}
        self._broadcast_inputs = {}

        for var_name, value in self._raw_inputs.items():
            if isinstance(value, Each):
                self._each[var_name] = value.source
            else:
                self._broadcast_inputs[var_name] = value

        # Validate đúng một Each() source
        if len(self._each) != 1:
            raise ValueError(
                f"AsyncIterNode requires exactly one Each() source in inputs, got: {list(self._each.keys())}"
            )

        self._callback = callback
        self._batch_fn = batch_fn
        self._max_concurrency = max_concurrency if max_concurrency is not None else os.cpu_count()

    def _post_build(self):
        """Thiết lập inputs/outputs sau khi inner graph được build.

        QUAN TRỌNG: Method này normalize _each và _broadcast_inputs dùng
        _normalize_params để resolve PARENT refs thành self.father.
        """
        # Normalize each và broadcast inputs (resolve PARENT refs)
        normalized_each = self._normalize_params(self._each)
        normalized_broadcast = self._normalize_params(self._broadcast_inputs)

        # Extract value/Ref từ normalized Params
        self._each = {k: v.value for k, v in normalized_each.items()}
        self._broadcast_inputs = {k: v.value for k, v in normalized_broadcast.items()}

        # Lấy tên iteration variable
        iter_var_name = list(self._each.keys())[0]

        # Build inputs: iteration var (AsyncIterable) + broadcast inputs (Any)
        parsed_inputs = {
            iter_var_name: Param(type=AsyncIterable, required=True, value=self._each[iter_var_name])
        }

        # Nếu batching enabled, đăng ký thêm batch và batch_size variables
        if self._batch_fn is not None:
            parsed_inputs["batch"] = Param(type=List, required=False)
            parsed_inputs["batch_size"] = Param(type=int, required=False)

        parsed_inputs.update({
            var_name: Param(type=Any, required=isinstance(value, Ref), value=value)
            for var_name, value in self._broadcast_inputs.items()
        })

        # Build outputs: từ inner graph (dạng lists) + metrics
        parsed_outputs = {
            key: Param(type=List, required=param.required)
            for key, param in self._graph.outputs.items()
        }
        parsed_outputs["iteration_metrics"] = Param(type=Dict, required=False)

        # Merge với user-provided outputs nếu có
        if self._raw_outputs is not None:
            self.outputs = self._merge_params(parsed_outputs, self._raw_outputs)
        else:
            self.outputs = parsed_outputs

        # Set inputs
        self.inputs = parsed_inputs

    def _resolve_values(
        self,
        values: Dict[str, Any],
        state: 'MemoryState',
        context_id: Optional[str]
    ) -> Dict[str, Any]:
        """Resolve values, dereference các Ref objects.

        Args:
            values: Dict {var_name: value_or_ref}
            state: Workflow state
            context_id: Context ID để resolution

        Returns:
            Dict mapping variable names đến resolved values.

        Note: Phải apply value._fn ở đây vì Ref trong values có thể có
        operations (e.g., ref["key"]["subkey"]) không được đăng ký trong
        schema. Schema chỉ lưu base node/var, không lưu operations.
        """
        result = {}
        for var_name, value in values.items():
            if isinstance(value, Ref):
                # Lấy raw value từ state và apply Ref's operations
                raw = state[value.node, value.var, context_id]
                result[var_name] = value._fn(raw)
            else:
                result[var_name] = value
        return result

    def _get_source(self, state: 'MemoryState', context_id: Optional[str]) -> AsyncIterable:
        """Lấy async iterable source, resolve Ref nếu cần."""
        iter_var_name = list(self._each.keys())[0]
        source = self._each[iter_var_name]

        if isinstance(source, Ref):
            # Lấy raw value từ state và apply Ref's operations
            raw = state[source.node, source.var, context_id]
            return source._fn(raw)
        return source

    async def _process_chunk(
        self,
        chunk_data: Dict[str, Any],
        chunk_id: int,
        state: 'MemoryState',
        request_id: str
    ) -> Dict[str, Any]:
        """Xử lý single chunk qua inner graph."""
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
        state: 'MemoryState',
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Xử lý streaming data với concurrent processing nhưng ordered output."""
        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        perf_start = perf_counter()
        _inputs = {}
        _outputs = {}

        try:
            # Lấy source và broadcast values
            source = self._get_source(state, context_id)
            broadcast_values = self._resolve_values(self._broadcast_inputs, state, context_id)
            iter_var_name = list(self._each.keys())[0]

            # Lưu inputs để logging
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

            # Collected results (nếu không streaming qua callback)
            collected_results: List[Dict[str, Any]] = []

            async def process_with_semaphore(chunk_data: Dict[str, Any], cid: int):
                async with semaphore:
                    result = await self._process_chunk(chunk_data, cid, state, request_id)
                    await result_queue.put((cid, result))

            async def consume_stream():
                """Consume stream và launch processing tasks."""
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
                        # Normal mode: mỗi item thành iteration variable
                        chunk_data = {iter_var_name: item, **broadcast_values}
                        tasks.append(asyncio.create_task(
                            process_with_semaphore(chunk_data, total_chunks)
                        ))
                        total_chunks += 1

                # Flush batch còn lại
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

                # Chờ tất cả tasks và signal completion
                if tasks:
                    await asyncio.gather(*tasks)
                await result_queue.put(None)  # Sentinel

            # Start consumer
            consumer_task = asyncio.create_task(consume_stream())

            # Collect results theo thứ tự
            pending_results: Dict[int, Dict] = {}
            next_id = 0

            while True:
                item = await result_queue.get()
                if item is None:
                    break

                cid, result = item
                pending_results[cid] = result

                # Emit results theo thứ tự
                while next_id in pending_results:
                    result = pending_results.pop(next_id)
                    latencies.append(result.get("latency_ms", 0))

                    if result.get("success"):
                        success_count += 1

                        # Stream đến callback nếu có
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

            # Cảnh báo nếu error rate cao (>10%)
            if total_chunks > 0 and error_count / total_chunks > 0.1:
                LOGGER.warning(
                    f"AsyncIterNode '{self.full_name}': high error rate "
                    f"({error_count / total_chunks:.1%}). {error_count}/{total_chunks} failed."
                )

            # Thêm callback metrics nếu có sử dụng
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

            # Transpose results sang column-oriented format (giống ForLoopNode)
            output_keys = list(self._graph.outputs.keys())
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
            duration_ms = (perf_counter() - perf_start) * 1000
            self._log(request_id, context_id, _inputs, _outputs, duration_ms)
            state[self.full_name, "start_time", context_id] = start_time
            state[self.full_name, "end_time", context_id] = end_time
            return _outputs

    def specific_metadata(self) -> Dict[str, Any]:
        """Trả về metadata đặc thù của subclass."""
        return {
            "max_concurrency": self._max_concurrency,
            "each": list(self._each.keys()),
            "inputs": list(self._broadcast_inputs.keys()),
            "has_callback": self._callback is not None,
            "has_batch_fn": self._batch_fn is not None,
        }
