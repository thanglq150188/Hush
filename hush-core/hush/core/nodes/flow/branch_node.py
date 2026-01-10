"""Node branch cho định tuyến có điều kiện trong workflow."""

from typing import Dict, Any, Optional, List, Union, Tuple, TYPE_CHECKING

from hush.core.configs.node_config import NodeType
from hush.core.nodes.base import BaseNode
from hush.core.states.ref import Ref
from hush.core.utils.common import Param, extract_condition_variables
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states import MemoryState


class BranchNode(BaseNode):
    """Node đánh giá các điều kiện và định tuyến luồng đến các node đích khác nhau.

    Các điều kiện được precompile để đạt hiệu năng tối đa.
    Hỗ trợ anchor để override định tuyến.
    """

    type: NodeType = "branch"

    __slots__ = [
        'given_candidates',
        'conditions',
        'default',
        'cases',
        'ref_cases',
    ]

    def __init__(
        self,
        cases: Optional[Dict[str, str]] = None,
        candidates: Optional[List[str]] = None,
        default: Optional[str] = None,
        inputs: Dict[str, Any] = None,
        outputs: Dict[str, Any] = None,
        ref_cases: Optional[List[Tuple[Ref, str]]] = None,
        **kwargs
    ):
        # Parse inputs/outputs từ cases
        parsed_inputs, parsed_outputs = self._parse_cases(cases or {}, ref_cases or [])

        # Gọi super().__init__ không truyền inputs/outputs
        super().__init__(**kwargs)

        # Merge parsed với user-provided
        self.inputs = self._merge_params(parsed_inputs, inputs)
        self.outputs = self._merge_params(parsed_outputs, outputs)

        self.cases = cases or {}

        for key, value in self.cases.items():
            if hasattr(value, 'is_base_node'):
                self.cases[key] = value.name

        self.default = default.name if hasattr(default, 'is_base_node') else default
        self.given_candidates = candidates
        self.ref_cases = ref_cases or []

        self.conditions = self._compile_conditions()
        self.core = self._create_core_function()

    def _parse_cases(
        self,
        cases: Dict[str, str],
        ref_cases: List[Tuple[Ref, str]]
    ) -> tuple:
        """Parse inputs/outputs từ cases và ref_cases.

        Args:
            cases: Dict condition_string -> target
            ref_cases: List of (condition_ref, target) tuples

        Returns:
            Tuple[Dict[str, Param], Dict[str, Param]]: (inputs, outputs)
        """
        # Inputs: anchor + các biến trong điều kiện
        inputs = {"anchor": Param(type=str, default=None)}

        # Parse từ string cases
        for condition in cases:
            vars = extract_condition_variables(condition)
            for k, v in vars.items():
                inputs[k] = Param(required=True)

        # Parse từ ref_cases - lấy var name từ mỗi Ref
        for ref, target in ref_cases:
            var_name = ref.var
            if var_name not in inputs:
                inputs[var_name] = Param(required=True)

        # Outputs
        outputs = {
            "target": Param(type=str, required=True),
            "matched": Param(type=str)
        }

        return inputs, outputs

    @property
    def candidates(self) -> List[str]:
        """Danh sách các node đích có thể."""
        if self.given_candidates:
            return self.given_candidates

        # Collect targets from both string cases and ref_cases
        targets = list(self.cases.values())
        targets.extend(target for _, target in self.ref_cases)

        if self.default:
            return targets + [self.default]
        else:
            return targets

    def _compile_conditions(self) -> List[tuple]:
        """Precompile tất cả điều kiện để đạt hiệu năng tối đa."""
        compiled_conditions = []

        for condition, target in self.cases.items():
            try:
                compiled_code = compile(condition, f'<condition: {condition}>', 'eval')
                compiled_conditions.append((compiled_code, condition, target))
            except SyntaxError as e:
                LOGGER.error(f"Cú pháp điều kiện không hợp lệ '{condition}': {e}")
                raise ValueError(f"Cú pháp điều kiện không hợp lệ: {condition}")

        return compiled_conditions

    def _create_core_function(self):
        """Tạo function đánh giá core đã tối ưu."""
        def core(**inputs) -> Dict[str, str]:
            anchor = inputs.get('anchor')
            if anchor:
                return {"target": anchor, "matched": "anchor"}

            target, matched = self._evaluate_conditions(inputs)
            return {"target": target, "matched": matched}

        return core

    def _evaluate_conditions(self, inputs: Dict[str, Any]) -> tuple:
        """Đánh giá tất cả điều kiện và trả về điều kiện khớp đầu tiên."""
        safe_inputs = dict(inputs)

        # First evaluate string-based conditions
        for compiled_cond, condition_str, target in self.conditions:
            try:
                result = eval(compiled_cond, {"__builtins__": {}}, safe_inputs)

                if result:
                    LOGGER.debug(f"Điều kiện '{condition_str}' khớp, định tuyến đến '{target}'")
                    return target, condition_str

            except Exception as e:
                LOGGER.error(f"Lỗi khi đánh giá điều kiện '{condition_str}': {e}")
                continue

        # Then evaluate Ref-based conditions (supports .apply())
        for ref, target in self.ref_cases:
            try:
                # Get the input value for this ref's variable
                value = safe_inputs.get(ref.var)
                # Execute all ops on the ref (including .apply())
                result = ref.execute(value)

                if result:
                    condition_desc = f"ref:{ref.var}"
                    LOGGER.debug(f"Điều kiện '{condition_desc}' khớp, định tuyến đến '{target}'")
                    return target, condition_desc

            except Exception as e:
                LOGGER.error(f"Lỗi khi đánh giá ref condition cho '{ref.var}': {e}")
                continue

        if self.default:
            LOGGER.debug(f"Không có điều kiện khớp, sử dụng target mặc định '{self.default}'")
            return self.default, "default"
        else:
            LOGGER.warning("Không có điều kiện khớp và không có target mặc định")
            return None, None

    def get_target(
        self,
        state: 'MemoryState',
        context_id: Optional[str] = None
    ) -> Optional[str]:
        """Lấy target đã định tuyến từ state."""
        return state[self.full_name, "target", context_id]

    def specific_metadata(self) -> Dict[str, Any]:
        """Trả về metadata riêng của subclass."""
        return {
            "cases": self.cases,
            "default_target": self.default,
            "candidates": self.candidates,
            "num_conditions": len(self.conditions)
        }


class Branch:
    """Fluent builder để tạo BranchNode với cú pháp tự nhiên hơn.

    Thay vì:
        branch = BranchNode(
            name="router",
            cases={
                "score >= 90": "excellent",
                "score >= 70": "good",
            },
            default="fail",
            inputs={"score": PARENT["score"]}
        )

    Có thể viết:
        branch = (Branch("router")
            .if_(PARENT["score"] >= 90, excellent_node)
            .if_(PARENT["score"] >= 70, good_node)
            .otherwise(fail_node))

    Hoặc với string target:
        branch = (Branch("router")
            .if_(PARENT["score"] >= 90, "excellent")
            .if_(PARENT["score"] >= 70, "good")
            .otherwise("fail"))
    """

    __slots__ = ('_name', '_cases', '_default', '_inputs', '_kwargs')

    def __init__(self, name: str, **kwargs):
        """Khởi tạo Branch builder.

        Args:
            name: Tên của BranchNode
            **kwargs: Các tham số khác cho BranchNode
        """
        self._name = name
        self._cases: List[Tuple[Ref, str]] = []  # List of (condition_ref, target)
        self._default: Optional[str] = None
        self._inputs: Dict[str, Any] = {}
        self._kwargs = kwargs

    def if_(self, condition: Ref, target: Union[str, BaseNode]) -> 'Branch':
        """Thêm một case với điều kiện từ Ref.

        Args:
            condition: Ref với comparison operation (e.g., PARENT["score"] >= 90)
            target: Node đích hoặc tên node đích

        Returns:
            self để chain tiếp

        Example:
            .if_(PARENT["score"] >= 90, "excellent")
            .if_(PARENT["score"] >= 70, good_node)
        """
        target_name = target.name if hasattr(target, 'name') else target
        self._cases.append((condition, target_name))
        return self

    def otherwise(self, target: Union[str, BaseNode]) -> 'BranchNode':
        """Đặt default target và build BranchNode.

        Args:
            target: Node đích mặc định khi không có điều kiện nào khớp

        Returns:
            BranchNode đã được tạo
        """
        self._default = target.name if hasattr(target, 'name') else target
        return self._build()

    def build(self) -> 'BranchNode':
        """Build BranchNode (dùng khi không có default).

        Returns:
            BranchNode đã được tạo
        """
        return self._build()

    def _build(self) -> 'BranchNode':
        """Internal build method.

        Uses ref_cases directly to support all Ref operations including .apply().
        """
        all_inputs = {}

        # Build inputs from ref conditions
        for condition_ref, target in self._cases:
            var_name = condition_ref.var
            if var_name not in all_inputs:
                # Create base Ref without ops for input resolution
                base_ref = Ref(condition_ref.raw_node, var_name)
                all_inputs[var_name] = base_ref

        return BranchNode(
            name=self._name,
            cases={},  # No string cases from fluent builder
            ref_cases=self._cases,  # Use ref_cases directly
            default=self._default,
            inputs=all_inputs,
            **self._kwargs
        )
