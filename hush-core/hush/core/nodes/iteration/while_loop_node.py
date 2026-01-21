"""While loop node để iterate khi condition còn true."""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
from time import perf_counter
import traceback
import asyncio

from hush.core.configs.node_config import NodeType
from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.nodes.iteration.base import calculate_iteration_metrics
from hush.core.utils.common import Param, extract_condition_variables
from hush.core.utils.context import _current_graph
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states import MemoryState


class WhileLoopNode(GraphNode):
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

    __slots__ = ['_max_iterations', '_stop_condition', '_compiled_condition', '_raw_inputs', '_token']

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
        # Lưu raw inputs trước khi super().__init__ normalize chúng
        self._raw_inputs = kwargs.get('inputs') or {}

        super().__init__(**kwargs)

        self._token = None  # Cho context manager
        self._max_iterations = max_iterations
        self._stop_condition = stop_condition
        self._compiled_condition = self._compile_condition(stop_condition) if stop_condition else None

    def __enter__(self):
        """Set this node as current graph context."""
        self._token = _current_graph.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Reset graph context."""
        _current_graph.reset(self._token)

    def _compile_condition(self, condition: str):
        """Compile stop condition để tăng performance."""
        try:
            return compile(condition, f'<stop_condition: {condition}>', 'eval')
        except SyntaxError as e:
            LOGGER.error("Invalid stop_condition syntax [str]'%s'[/str]: %s", condition, e)
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
            LOGGER.error("Error evaluating stop_condition [str]'%s'[/str]: %s", self._stop_condition, e)
            return False  # Khi error, tiếp tục (để max_iterations làm safety)

    def build(self):
        """Build WhileLoopNode - build child nodes rồi setup iteration-specific config."""
        # Build child nodes first (như GraphNode)
        for node in self._nodes.values():
            if hasattr(node, 'build'):
                node.build()

        # Setup graph schema, flow type, endpoints (từ GraphNode)
        self._setup_schema()
        self._build_flow_type()
        self._setup_endpoints()

        # Calculate ready_count cho execution scheduling
        self.ready_count = {}
        self.has_soft_preds = set()
        for name in self._nodes:
            hard_pred_count = 0
            has_soft = False
            for pred in self.prevs[name]:
                edge = self._edges_lookup.get((pred, name))
                if edge and edge.soft:
                    has_soft = True
                elif edge and not edge.soft:
                    hard_pred_count += 1
                elif edge is None:
                    hard_pred_count += 1
            if has_soft:
                self.has_soft_preds.add(name)
                hard_pred_count += 1
            self.ready_count[name] = hard_pred_count

        self._is_building = False

        # Iteration-specific post-build
        self._post_build()

    def _post_build(self):
        """Thiết lập inputs/outputs từ graph sau khi build."""
        # Normalize raw inputs (resolve PARENT refs)
        normalized_inputs = self._normalize_params(self._raw_inputs)
        user_inputs = {k: v for k, v in normalized_inputs.items()}

        # Preserve output mappings set via >> syntax before _post_build
        existing_outputs = self.outputs or {}

        # Bắt đầu với graph's inputs/outputs (self giờ là GraphNode)
        parsed_inputs = {
            k: Param(type=v.type, required=v.required, default=v.default,
                     description=v.description, value=v.value)
            for k, v in (self.inputs or {}).items()
        }
        parsed_outputs = {
            k: Param(type=v.type, required=v.required, default=v.default,
                     description=v.description, value=v.value)
            for k, v in (self.outputs or {}).items()
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
        self.inputs = self._merge_params(parsed_inputs, user_inputs)
        self.outputs = parsed_outputs

        # Restore .value references from existing outputs (set by >> syntax)
        for key, existing_param in existing_outputs.items():
            if key in self.outputs and existing_param.value is not None:
                self.outputs[key].value = existing_param.value

    async def _run_graph(
        self,
        state: 'MemoryState',
        context_id: str,
        parent_context: str
    ) -> Dict[str, Any]:
        """Chạy child nodes với parent_context."""
        active_tasks: Dict[str, asyncio.Task] = {}
        ready_count = self.ready_count.copy()
        soft_satisfied: set = set()

        for entry in self.entries:
            task = asyncio.create_task(
                name=entry,
                coro=self._nodes[entry].run(state, context_id, parent_context)
            )
            active_tasks[entry] = task

        while active_tasks:
            done_tasks, _ = await asyncio.wait(
                active_tasks.values(),
                return_when=asyncio.FIRST_COMPLETED
            )

            nodes = self._nodes
            nexts = self.nexts
            edges_lookup = self._edges_lookup

            for task in done_tasks:
                node_name = task.get_name()
                active_tasks.pop(node_name)

                node = nodes[node_name]

                if node.type == "branch":
                    branch_target = node.get_target(state, context_id)
                    from hush.core.nodes.base import END
                    if branch_target != END.name:
                        next_nodes = [branch_target]
                    else:
                        next_nodes = []
                else:
                    next_nodes = nexts[node_name]

                for next_node in next_nodes:
                    edge = edges_lookup.get((node_name, next_node))
                    is_soft = edge and edge.soft

                    if is_soft:
                        if next_node in soft_satisfied:
                            continue
                        soft_satisfied.add(next_node)

                    count = ready_count[next_node] - 1
                    ready_count[next_node] = count

                    if count == 0:
                        task = asyncio.create_task(
                            name=next_node,
                            coro=nodes[next_node].run(state, context_id, parent_context)
                        )
                        active_tasks[next_node] = task

        return self.get_outputs(state, context_id, parent_context)

    async def run(
        self,
        state: 'MemoryState',
        context_id: Optional[str] = None,
        parent_context: Optional[str] = None
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
            _inputs = self.get_inputs(state, context_id, parent_context)
            step_inputs = _inputs
            latencies_ms: List[float] = []
            step_count = 0

            # Check stop condition trước iteration đầu tiên
            should_stop = self._evaluate_stop_condition(step_inputs)

            while not should_stop and step_count < self._max_iterations:
                # Chain context ID để tránh cache conflict
                step_context = f"while[{step_count}]" if not context_id else f"{context_id}.while[{step_count}]"
                iter_start = perf_counter()

                # Inject inputs trực tiếp vào state
                for var_name, value in step_inputs.items():
                    state[self.full_name, var_name, step_context] = value

                # Chạy child nodes với step_context là cả context_id và parent_context
                _outputs = await self._run_graph(state, step_context, step_context)

                latencies_ms.append((perf_counter() - iter_start) * 1000)

                # Merge outputs với previous inputs để giữ các variables không đổi
                step_inputs = {**step_inputs, **_outputs}
                step_count += 1

                # Check stop condition sau iteration
                should_stop = self._evaluate_stop_condition(step_inputs)

            # Cảnh báo nếu max_iterations đạt (có thể infinite loop)
            if step_count >= self._max_iterations and not should_stop:
                LOGGER.warning(
                    "[title]\\[%s][/title] WhileLoopNode [highlight]%s[/highlight]: max_iterations [muted](%s)[/muted] reached. "
                    "Condition [str]'%s'[/str] never evaluated to True. This may indicate an infinite loop or incorrect stop condition.",
                    request_id, self.full_name, self._max_iterations, self._stop_condition
                )

            # Tính iteration metrics
            iteration_metrics = calculate_iteration_metrics(latencies_ms)
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
        """Trả về metadata đặc thù của subclass."""
        return {
            "max_iterations": self._max_iterations,
            "stop_condition": self._stop_condition
        }
