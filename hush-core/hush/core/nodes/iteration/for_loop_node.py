"""ForLoopNode - sequential iteration node for processing items one at a time."""

from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
from time import perf_counter
import traceback
import asyncio

from hush.core.configs.node_config import NodeType
from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.nodes.iteration.base import Each, calculate_iteration_metrics
from hush.core.states.ref import Ref
from hush.core.utils.common import Param
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states import MemoryState


class ForLoopNode(GraphNode):
    """Sequential iteration node - processes items one at a time in order.

    Use ForLoopNode when:
        - Iterations may depend on results from previous iterations
        - You need predictable sequential execution
        - Memory constraints require processing one item at a time
        - Nested loops where inner loop depends on outer loop variables

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

        # Tạo 3 iterations (chạy tuần tự):
        #   iteration 0: {"x": 1, "y": 10, "multiplier": 10}
        #   iteration 1: {"x": 2, "y": 20, "multiplier": 10}
        #   iteration 2: {"x": 3, "y": 30, "multiplier": 10}

        # Output (column-oriented, transposed từ inner graph outputs):
        #   {"result": [r1, r2, r3], "iteration_metrics": {...}}

    Note:
        For parallel iteration where items are independent,
        use MapNode instead for better performance.
    """

    type: NodeType = "for"

    __slots__ = ['_each', '_broadcast_inputs', '_raw_outputs', '_raw_inputs']

    def __init__(
        self,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Khởi tạo ForLoopNode.

        Args:
            inputs: Dict mapping variable names đến values hoặc Each(source).
                    - Each(source): Được iterate (zipped nếu nhiều)
                    - Other values: Broadcast cho tất cả iterations
                    Values có thể là literals hoặc Refs đến upstream nodes.
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

    def build(self):
        """Build ForLoopNode - build child nodes rồi setup iteration-specific config."""
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
        """Thiết lập inputs/outputs sau khi graph được build.

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

        # Build outputs: dẫn xuất từ graph's outputs (column-oriented)
        # self.outputs đã được populate bởi _setup_schema()
        graph_outputs = self.outputs or {}
        parsed_outputs = {
            key: Param(type=List, required=param.required)
            for key, param in graph_outputs.items()
        }
        parsed_outputs["iteration_metrics"] = Param(type=Dict, required=False)

        # Preserve output mappings set via >> syntax before _post_build
        # e.g., loop["result"] >> PARENT["final_result"] sets loop.outputs["result"].value
        existing_outputs = self.outputs or {}

        # Merge với user-provided outputs nếu có
        if self._raw_outputs is not None:
            self.outputs = self._merge_params(parsed_outputs, self._raw_outputs)
        else:
            self.outputs = parsed_outputs

        # Restore .value references from existing outputs (set by >> syntax)
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
                "ForLoopNode [highlight]%s[/highlight]: 'each' variables have different lengths: %s",
                self.full_name, lengths
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

    async def _run_graph(
        self,
        state: 'MemoryState',
        context_id: str,
        parent_context: str
    ) -> Dict[str, Any]:
        """Chạy child nodes với parent_context.

        Reuse logic từ GraphNode.run() nhưng truyền parent_context cho children.
        """
        active_tasks: Dict[str, asyncio.Task] = {}
        ready_count = self.ready_count.copy()
        soft_satisfied: set = set()

        # Start entry nodes
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

            # Cache dict lookups for performance
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
        """Thực thi for loop qua iteration data tuần tự.

        Args:
            state: Workflow state
            context_id: Context của ForLoopNode này
            parent_context: Context của PARENT để resolve PARENT refs trong inputs
        """
        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        perf_start = perf_counter()
        _inputs = {}
        _outputs = {}

        try:
            # Resolve each và broadcast values dùng parent_context
            # (vì có thể là PARENT refs cần resolve từ father's context)
            each_values = self._resolve_values(self._each, state, parent_context)
            broadcast_values = self._resolve_values(self._broadcast_inputs, state, parent_context)

            # Build iteration data
            iteration_data = self._build_iteration_data(each_values, broadcast_values)

            # Lưu inputs để logging
            _inputs = {**each_values, **broadcast_values}

            # Cảnh báo nếu không có iterations
            if not iteration_data:
                LOGGER.warning(
                    "[title]\\[%s][/title] ForLoopNode [highlight]%s[/highlight]: no iteration data. No iterations will be executed.",
                    request_id, self.full_name
                )

            # Execute iterations sequentially
            latencies_ms: List[float] = []
            final_results: List[Dict[str, Any]] = []
            success_count = 0

            for i, loop_data in enumerate(iteration_data):
                # Chain context ID để tránh cache conflict giữa nested loops
                iter_context = f"loop[{i}]" if not context_id else f"{context_id}.loop[{i}]"
                iter_start = perf_counter()

                try:
                    # Inject inputs trực tiếp vào state (không cần duplicate injection)
                    for var_name, value in loop_data.items():
                        state[self.full_name, var_name, iter_context] = value

                    # Chạy child nodes với iter_context là cả context_id và parent_context
                    result = await self._run_graph(state, iter_context, iter_context)
                    final_results.append(result)
                    success_count += 1
                except Exception as e:
                    final_results.append({"error": str(e), "error_type": type(e).__name__})

                latencies_ms.append((perf_counter() - iter_start) * 1000)

            error_count = len(iteration_data) - success_count

            # Tính iteration metrics
            iteration_metrics = calculate_iteration_metrics(latencies_ms)
            iteration_metrics.update({
                "total_iterations": len(iteration_data),
                "success_count": success_count,
                "error_count": error_count,
            })

            # Cảnh báo nếu error rate cao (>10%)
            if iteration_data and error_count / len(iteration_data) > 0.1:
                LOGGER.warning(
                    "[title]\\[%s][/title] ForLoopNode [highlight]%s[/highlight]: high error rate [muted](%s)[/muted]. %s/%s iterations failed.",
                    request_id, self.full_name, f"{error_count / len(iteration_data):.1%}", error_count, len(iteration_data)
                )

            # Transpose results sang column-oriented format
            # Lấy output keys từ self.outputs (không phải self._graph.outputs)
            output_keys = [k for k in self.outputs.keys() if k != "iteration_metrics"]
            _outputs = {
                key: [r.get(key) for r in final_results]
                for key in output_keys
            }
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
            "each": list(self._each.keys()),
            "inputs": list(self._broadcast_inputs.keys())
        }
