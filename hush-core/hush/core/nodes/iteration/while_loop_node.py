"""While loop node để iterate khi condition còn true."""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
from time import perf_counter
import traceback

from hush.core.configs.node_config import NodeType
from hush.core.nodes.iteration.base import BaseIterationNode
from hush.core.utils.common import Param, extract_condition_variables
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states import MemoryState


class WhileLoopNode(BaseIterationNode):
    """Node iterate khi condition còn true.

    Loop tiếp tục khi `stop_condition` evaluate thành False.
    Khi `stop_condition` thành True, loop dừng.

    Example:
        with WhileLoopNode(
            name="loop",
            inputs={"counter": 0},
            stop_condition="counter >= 5"
        ) as loop:
            node = process(
                inputs={"counter": PARENT["counter"]},
                outputs={"new_counter": PARENT["counter"]}
            )
            START >> node >> END
    """

    type: NodeType = "while"

    __slots__ = ['_max_iterations', '_stop_condition', '_compiled_condition']

    def __init__(
        self,
        stop_condition: Optional[str] = None,
        max_iterations: int = 100,
        **kwargs
    ):
        """Khởi tạo WhileLoopNode.

        Args:
            stop_condition: Biểu thức string được evaluate mỗi iteration.
                Khi evaluate thành True, loop dừng.
                Ví dụ: "counter >= 5" hoặc "total > 100 and done"
            max_iterations: Số iterations tối đa để ngăn infinite loops.
                Mặc định là 100.
        """
        super().__init__(**kwargs)
        self._max_iterations = max_iterations
        self._stop_condition = stop_condition
        self._compiled_condition = self._compile_condition(stop_condition) if stop_condition else None

    def _compile_condition(self, condition: str):
        """Compile stop condition để tăng performance."""
        try:
            return compile(condition, f'<stop_condition: {condition}>', 'eval')
        except SyntaxError as e:
            LOGGER.error(f"Invalid stop_condition syntax '{condition}': {e}")
            raise ValueError(f"Invalid stop_condition syntax: {condition}") from e

    def _evaluate_stop_condition(self, inputs: Dict[str, Any]) -> bool:
        """Evaluate stop condition với current inputs.

        Returns:
            True nếu loop nên dừng, False để tiếp tục.
        """
        if self._compiled_condition is None:
            return False  # Không có condition = không bao giờ dừng (dựa vào max_iterations)

        try:
            result = eval(self._compiled_condition, {"__builtins__": {}}, inputs)
            return bool(result)
        except Exception as e:
            LOGGER.error(f"Error evaluating stop_condition '{self._stop_condition}': {e}")
            return False  # Khi error, tiếp tục (để max_iterations làm safety)

    def _post_build(self):
        """Thiết lập inputs/outputs từ inner graph sau khi build."""
        # Lưu user-provided inputs trước khi merge
        user_inputs = self.inputs.copy()

        # Preserve output mappings set via << syntax before _post_build
        # e.g., PARENT["final_total"] << loop["total"] sets loop.outputs["total"].value
        existing_outputs = self.outputs or {}

        # Bắt đầu với inner graph's inputs/outputs
        parsed_inputs = {
            k: Param(type=v.type, required=v.required, default=v.default,
                     description=v.description, value=v.value)
            for k, v in (self._graph.inputs or {}).items()
        }
        parsed_outputs = {
            k: Param(type=v.type, required=v.required, default=v.default,
                     description=v.description, value=v.value)
            for k, v in (self._graph.outputs or {}).items()
        }

        # Thêm variables từ stop_condition vào inputs
        if self._stop_condition:
            for var_name in extract_condition_variables(self._stop_condition):
                if var_name not in parsed_inputs:
                    parsed_inputs[var_name] = Param(type=Any, required=False)

        # Thêm iteration_metrics vào outputs
        if "iteration_metrics" not in parsed_outputs:
            parsed_outputs["iteration_metrics"] = Param(type=Dict, required=False)

        # Merge user-provided inputs với parsed schema
        # User inputs có thể chứa literal values ({"counter": 0})
        self.inputs = self._merge_params(parsed_inputs, user_inputs)
        self.outputs = parsed_outputs

        # Restore .value references from existing outputs (set by << syntax)
        for key, existing_param in existing_outputs.items():
            if key in self.outputs and existing_param.value is not None:
                self.outputs[key].value = existing_param.value

    async def run(
        self,
        state: 'MemoryState',
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Thực thi while loop cho đến khi stop_condition True hoặc max_iterations đạt."""
        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        perf_start = perf_counter()
        _inputs = {}
        _outputs = {}

        try:
            _inputs = self.get_inputs(state, context_id)
            step_inputs = _inputs
            latencies_ms: List[float] = []
            step_count = 0

            # Check stop condition trước iteration đầu tiên
            should_stop = self._evaluate_stop_condition(step_inputs)

            while not should_stop and step_count < self._max_iterations:
                step_name = f"while-{step_count}"
                iter_start = perf_counter()

                self.inject_inputs(state, step_inputs, step_name)
                _outputs = await self._graph.run(state, context_id=step_name)

                latencies_ms.append((perf_counter() - iter_start) * 1000)

                # Merge outputs với previous inputs để giữ các variables không đổi
                step_inputs = {**step_inputs, **_outputs}
                step_count += 1

                # Check stop condition sau iteration
                should_stop = self._evaluate_stop_condition(step_inputs)

            # Cảnh báo nếu max_iterations đạt (có thể infinite loop)
            if step_count >= self._max_iterations and not should_stop:
                LOGGER.warning(
                    f"WhileLoopNode '{self.full_name}': max_iterations ({self._max_iterations}) reached. "
                    f"Condition '{self._stop_condition}' never evaluated to True. "
                    "This may indicate an infinite loop or incorrect stop condition."
                )

            # Tính iteration metrics (tất cả completed iterations đều success, errors propagate)
            iteration_metrics = self._calculate_iteration_metrics(latencies_ms)
            iteration_metrics.update({
                "total_iterations": step_count,
                "success_count": step_count,
                "error_count": 0,
                "max_iterations_reached": step_count >= self._max_iterations,
                "stopped_by_condition": should_stop,
            })

            # Thêm iteration_metrics vào outputs
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
            "max_iterations": self._max_iterations,
            "stop_condition": self._stop_condition
        }
