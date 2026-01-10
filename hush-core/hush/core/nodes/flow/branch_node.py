"""Node branch cho định tuyến có điều kiện trong workflow."""

from typing import Dict, Any, Optional, List, TYPE_CHECKING

from hush.core.configs.node_config import NodeType
from hush.core.nodes.base import BaseNode
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
    ]

    def __init__(
        self,
        cases: Optional[Dict[str, str]] = None,
        candidates: Optional[List[str]] = None,
        default: Optional[str] = None,
        inputs: Dict[str, Any] = None,
        outputs: Dict[str, Any] = None,
        **kwargs
    ):
        # Parse inputs/outputs từ cases
        parsed_inputs, parsed_outputs = self._parse_cases(cases or {})

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

        self.conditions = self._compile_conditions()
        self.core = self._create_core_function()

    def _parse_cases(self, cases: Dict[str, str]) -> tuple:
        """Parse inputs/outputs từ cases.

        Returns:
            Tuple[Dict[str, Param], Dict[str, Param]]: (inputs, outputs)
        """
        # Inputs: anchor + các biến trong điều kiện
        inputs = {"anchor": Param(type=str, default=None)}

        for condition in cases:
            vars = extract_condition_variables(condition)
            for k, v in vars.items():
                inputs[k] = Param(required=True)

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

        if self.default:
            return list(self.cases.values()) + [self.default]
        else:
            return list(self.cases.values())

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

        for compiled_cond, condition_str, target in self.conditions:
            try:
                result = eval(compiled_cond, {"__builtins__": {}}, safe_inputs)

                if result:
                    LOGGER.debug(f"Điều kiện '{condition_str}' khớp, định tuyến đến '{target}'")
                    return target, condition_str

            except Exception as e:
                LOGGER.error(f"Lỗi khi đánh giá điều kiện '{condition_str}': {e}")
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
