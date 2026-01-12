"""For loop node với unified inputs API sử dụng Each wrapper cho iteration sources."""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
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


class ForLoopNode(BaseIterationNode):
    """Node iterate qua collections với hỗ trợ broadcast variables.

    API (unified inputs với Each wrapper):
        - `inputs`: Dict tất cả variables. Dùng Each() wrapper cho iteration sources.
          - Each(source): Variable được iterate (khác value mỗi iteration)
          - Regular values: Broadcast cho tất cả iterations (cùng value)

    Example:
        with ForLoopNode(
            name="process_loop",
            inputs={
                "x": Each([1, 2, 3]),           # iterate
                "y": Each([10, 20, 30]),        # iterate (zipped với x)
                "multiplier": 10                 # broadcast
            }
        ) as loop:
            node = calc(inputs={"x": PARENT["x"], "y": PARENT["y"], "multiplier": PARENT["multiplier"]})
            START >> node >> END

        # Tạo 3 iterations:
        #   iteration 0: {"x": 1, "y": 10, "multiplier": 10}
        #   iteration 1: {"x": 2, "y": 20, "multiplier": 10}
        #   iteration 2: {"x": 3, "y": 30, "multiplier": 10}

        # Output (column-oriented, transposed từ inner graph outputs):
        #   {"result": [r1, r2, r3], "iteration_metrics": {...}}
    """

    type: NodeType = "for"

    __slots__ = ['_max_concurrency', '_each', '_broadcast_inputs', '_raw_outputs', '_raw_inputs']

    def __init__(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        max_concurrency: Optional[int] = None,
        **kwargs
    ):
        """Khởi tạo ForLoopNode.

        Args:
            inputs: Dict mapping variable names đến values hoặc Each(source).
                    - Each(source): Được iterate (zipped nếu nhiều)
                    - Other values: Broadcast cho tất cả iterations
                    Values có thể là literals hoặc Refs đến upstream nodes.
            max_concurrency: Số concurrent tasks tối đa được chạy.
                Mặc định là số CPU cores nếu không chỉ định.
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

        # Build inputs: tất cả variables từ each (List type) + broadcast inputs (Any type)
        parsed_inputs = {
            var_name: Param(type=List, required=isinstance(value, Ref), value=value)
            for var_name, value in self._each.items()
        }
        parsed_inputs.update({
            var_name: Param(type=Any, required=isinstance(value, Ref), value=value)
            for var_name, value in self._broadcast_inputs.items()
        })

        # Build outputs: dẫn xuất từ inner graph's outputs (column-oriented)
        parsed_outputs = {
            key: Param(type=List, required=param.required)
            for key, param in self._graph.outputs.items()
        }
        parsed_outputs["iteration_metrics"] = Param(type=Dict, required=False)

        # Preserve output mappings set via << syntax before _post_build
        # e.g., PARENT["final_result"] << loop["result"] sets loop.outputs["result"].value
        existing_outputs = self.outputs or {}

        # Merge với user-provided outputs nếu có
        if self._raw_outputs is not None:
            self.outputs = self._merge_params(parsed_outputs, self._raw_outputs)
        else:
            self.outputs = parsed_outputs

        # Restore .value references from existing outputs (set by << syntax)
        for key, existing_param in existing_outputs.items():
            if key in self.outputs and existing_param.value is not None:
                self.outputs[key].value = existing_param.value

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

    def _build_iteration_data(
        self,
        each_values: Dict[str, List],
        broadcast_values: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build list iteration data bằng cách zip `each` values và thêm broadcast.

        Args:
            each_values: Dict {var_name: [values...]}
            broadcast_values: Dict {var_name: value}

        Returns:
            List các dicts, mỗi iteration một dict, với tất cả variables.
        """
        if not each_values:
            # Không có iteration variables - single iteration chỉ với broadcast
            return [broadcast_values.copy()] if broadcast_values else []

        # Validate tất cả lists có cùng độ dài
        lengths = {var: len(lst) for var, lst in each_values.items()}
        if len(set(lengths.values())) > 1:
            LOGGER.error(
                f"ForLoopNode '{self.full_name}': 'each' variables have different lengths: {lengths}"
            )
            raise ValueError(
                f"All 'each' variables must have the same length. Got: {lengths}"
            )

        # Zip và merge với broadcast - optimized to avoid dict spread
        keys = list(each_values.keys())
        result = []
        for vals in zip(*each_values.values()):
            item = broadcast_values.copy()  # Shallow copy broadcast
            for i, k in enumerate(keys):
                item[k] = vals[i]
            result.append(item)
        return result

    async def run(
        self,
        state: 'MemoryState',
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Thực thi for loop qua iteration data song song."""
        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        perf_start = perf_counter()
        _inputs = {}
        _outputs = {}

        try:
            # Resolve each và broadcast values
            each_values = self._resolve_values(self._each, state, context_id)
            broadcast_values = self._resolve_values(self._broadcast_inputs, state, context_id)

            # Build iteration data
            iteration_data = self._build_iteration_data(each_values, broadcast_values)

            # Lưu inputs để logging
            _inputs = {**each_values, **broadcast_values}

            # Cảnh báo nếu không có iterations
            if not iteration_data:
                LOGGER.warning(
                    f"ForLoopNode '{self.full_name}': no iteration data. No iterations will be executed."
                )

            # Tạo semaphore để giới hạn concurrency
            semaphore = asyncio.Semaphore(self._max_concurrency)

            async def execute_iteration(task_id: str, loop_data: Dict[str, Any]) -> Dict[str, Any]:
                """Thực thi single iteration, trả về result với metadata."""
                start = perf_counter()
                try:
                    async with semaphore:
                        self.inject_inputs(state, loop_data, task_id)
                        result = await self._graph.run(state, task_id)
                    return {"result": result, "latency_ms": (perf_counter() - start) * 1000, "success": True}
                except Exception as e:
                    return {"result": {"error": str(e), "error_type": type(e).__name__}, "latency_ms": (perf_counter() - start) * 1000, "success": False}

            # Chạy tất cả iterations song song
            raw_results = await asyncio.gather(*[
                execute_iteration(f"for[{i}]", data)
                for i, data in enumerate(iteration_data)
            ])

            # Extract metrics và results trong single pass
            latencies_ms = []
            final_results = []
            success_count = 0
            for r in raw_results:
                latencies_ms.append(r["latency_ms"])
                final_results.append(r["result"])
                success_count += r["success"]
            error_count = len(raw_results) - success_count

            # Tính iteration metrics
            iteration_metrics = self._calculate_iteration_metrics(latencies_ms)
            iteration_metrics.update({
                "total_iterations": len(iteration_data),
                "success_count": success_count,
                "error_count": error_count,
            })

            # Cảnh báo nếu error rate cao (>10%)
            if iteration_data and error_count / len(iteration_data) > 0.1:
                LOGGER.warning(
                    f"ForLoopNode '{self.full_name}': high error rate ({error_count / len(iteration_data):.1%}). "
                    f"{error_count}/{len(iteration_data)} iterations failed."
                )

            # Transpose results sang column-oriented format
            output_keys = list(self._graph.outputs.keys())
            _outputs = {
                key: [r.get(key) for r in final_results]
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
            "inputs": list(self._broadcast_inputs.keys())
        }
