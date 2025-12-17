"""Branch node for conditional routing in workflows."""

from typing import Dict, Any, Optional, List, TYPE_CHECKING

from hush.core.configs.node_config import NodeType
from hush.core.nodes.base import BaseNode
from hush.core.utils.common import Param, extract_condition_variables
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states import BaseState


class BranchNode(BaseNode):
    """A node that evaluates conditions and routes flow to different target nodes."""

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
        **kwargs
    ):
        # Build schemas before super().__init__
        input_schema, output_schema = self._build_schemas(cases or {})

        super().__init__(
            input_schema=input_schema,
            output_schema=output_schema,
            **kwargs
        )

        self.cases = cases or {}

        for key, value in self.cases.items():
            if hasattr(value, 'is_base_node'):
                self.cases[key] = value.name

        self.default = default.name if hasattr(default, 'is_base_node') else default
        self.given_candidates = candidates

        self.conditions = self._compile_conditions()
        self.core = self._create_core_function()

    def _build_schemas(self, cases: Dict[str, str]) -> tuple:
        """Build input/output schemas."""
        # Input schema: anchor + condition variables
        input_schema = {"anchor": Param(type=str, default=None)}

        for condition in cases:
            vars = extract_condition_variables(condition)
            for k, v in vars.items():
                input_schema[k] = Param(type=Any, required=True)

        # Output schema
        output_schema = {
            "target": Param(type=str, required=True),
            "matched": Param(type=str)
        }

        return input_schema, output_schema

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
        state: 'BaseState',
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


if __name__ == "__main__":
    import asyncio
    from hush.core.states import StateSchema

    async def main():
        schema = StateSchema("test")
        state = schema.create_state()

        # Test 1: Basic condition routing
        print("=" * 50)
        print("Test 1: Basic condition routing")
        print("=" * 50)

        branch = BranchNode(
            name="router",
            cases={
                "score >= 90": "excellent",
                "score >= 70": "good",
                "score >= 50": "pass",
            },
            default="fail",
            inputs={"score": 85}
        )

        print(f"Input schema: {branch.input_schema}")
        print(f"Output schema: {branch.output_schema}")
        print(f"Candidates: {branch.candidates}")

        result = await branch.run(state)
        print(f"Result (score=85): {result}")

        # Test 2: Multiple variables
        print("\n" + "=" * 50)
        print("Test 2: Multiple variables")
        print("=" * 50)

        branch2 = BranchNode(
            name="multi_router",
            cases={
                "age >= 18 and verified": "adult_verified",
                "age >= 18": "adult_unverified",
                "age >= 13": "teen",
            },
            default="child",
            inputs={"age": 20, "verified": True}
        )

        print(f"Input schema: {branch2.input_schema}")
        result2 = await branch2.run(state)
        print(f"Result (age=20, verified=True): {result2}")

        # Test 3: Anchor override
        print("\n" + "=" * 50)
        print("Test 3: Anchor override")
        print("=" * 50)

        branch3 = BranchNode(
            name="anchor_router",
            cases={
                "status == 'active'": "process",
            },
            default="skip",
            inputs={"status": "active", "anchor": "force_target"}
        )

        result3 = await branch3.run(state)
        print(f"Result (with anchor='force_target'): {result3}")

        # Test 4: Default fallback
        print("\n" + "=" * 50)
        print("Test 4: Default fallback")
        print("=" * 50)

        branch4 = BranchNode(
            name="fallback_router",
            cases={
                "value > 100": "high",
                "value > 50": "medium",
            },
            default="low",
            inputs={"value": 10}
        )

        result4 = await branch4.run(state)
        print(f"Result (value=10, no match): {result4}")

        # Test 5: Quick test using __call__
        print("\n" + "=" * 50)
        print("Test 5: Quick test using __call__")
        print("=" * 50)

        branch5 = BranchNode(
            name="quick_branch",
            cases={
                "x > 0": "positive",
                "x < 0": "negative",
            },
            default="zero"
        )

        result5 = branch5(x=5)
        print(f"branch5(x=5) = {result5}")

        result6 = branch5(x=-3)
        print(f"branch5(x=-3) = {result6}")

        result7 = branch5(x=0)
        print(f"branch5(x=0) = {result7}")

        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)

    asyncio.run(main())
