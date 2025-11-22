"""Branch node for conditional routing in workflows."""

from typing import Dict, Any, Optional, List

from hush.core.configs.node_config import NodeType
from hush.core.states.workflow_state import WorkflowState
from hush.core.nodes.base import BaseNode
from hush.core.schema import ParamSet
from hush.core.utils.common import extract_condition_variables
from hush.core.loggings import LOGGER



class BranchNode(BaseNode):
    """A node that evaluates conditions and routes flow to different target nodes."""

    type: NodeType = "branch"

    output_schema: ParamSet = (
        ParamSet.new()
            .var("target: str", required=True)
            .var("matched: str")
            .build()
    )

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
        **kwargs
    ):
        """Initialize BranchNode.

        Args:
            cases: Dictionary mapping conditions to target node names
            candidates: Optional explicit list of candidate targets
            default: Default target node if no conditions match
            **kwargs: Additional keyword arguments for BaseNode
        """

        super().__init__(**kwargs)

        self.cases = cases or {}

        for key, value in self.cases.items():
            if hasattr(value, 'is_base_node'):
                self.cases[key] = value.name

        self.default = default.name if hasattr(default, 'is_base_node') else default
        self.given_candidates = candidates

        self._build_schema()

        self.conditions = self._compile_conditions()

        self.core = self._create_core_function()

    def _build_schema(self):
        input_schema = ParamSet.new().var("anchor: str = None")

        for condition in self.cases:
            vars = extract_condition_variables(condition)
            for k, v in vars.items():
                input_schema = input_schema.var(f"{k}: {v}", required=True)

        self.input_schema = input_schema.build()

    @property
    def candidates(self) -> List[str]:
        if self.given_candidates:
            return self.given_candidates

        if self.default:
            return list(self.cases.values()) + [self.default]
        else:
            return list(self.cases.values())

    def _compile_conditions(self) -> List[tuple]:
        """Precompile all conditions for maximum performance."""
        compiled_conditions = []

        for condition, target in self.cases.items():
            try:
                compiled_code = compile(condition, f'<condition: {condition}>', 'eval')
                compiled_conditions.append((compiled_code, condition, target))
            except SyntaxError as e:
                LOGGER.error(f"Invalid condition syntax '{condition}': {e}")
                raise ValueError(f"Invalid condition syntax: {condition}")

        return compiled_conditions

    def _create_core_function(self):
        """Create the optimized core evaluation function."""
        def core(**inputs) -> Dict[str, str]:
            anchor = inputs.get('anchor')
            if anchor:
                return {"target": anchor, "matched": "anchor"}

            target, matched = self._evaluate_conditions(inputs)
            return {"target": target, "matched": matched}

        return core

    def _evaluate_conditions(self, inputs: Dict[str, Any]) -> tuple:
        """Evaluate all conditions and return the first match."""
        safe_inputs = dict(inputs)

        for compiled_cond, condition_str, target in self.conditions:
            try:
                result = eval(compiled_cond, {"__builtins__": {}}, safe_inputs)

                if result:
                    LOGGER.debug(f"Condition '{condition_str}' matched, routing to '{target}'")
                    return target, condition_str

            except Exception as e:
                LOGGER.error(f"Error evaluating condition '{condition_str}': {e}")
                continue

        if self.default:
            LOGGER.debug(f"No conditions matched, using default target '{self.default}'")
            return self.default, "default"
        else:
            LOGGER.warning("No conditions matched and no default target specified")
            return None, None

    def get_target(
        self,
        state: WorkflowState,
        context_id: Optional[str] = None
    ) -> Optional[str]:
        return state[self.full_name, "target", context_id]

    def specific_metadata(self) -> Dict[str, Any]:
        """Return subclass-specific metadata dictionary."""
        return {
            "cases": self.cases,
            "default_target": self.default,
            "candidates": self.candidates,
            "num_conditions": len(self.conditions)
        }
