"""Branch node for conditional routing in workflows."""

from typing import Dict, Any, Optional, List, TYPE_CHECKING

from hush.core.configs.node_config import NodeType
from hush.core.nodes.base import BaseNode
from hush.core.utils.common import Param, extract_condition_variables
from hush.core.loggings import LOGGER

if TYPE_CHECKING:
    from hush.core.states import MemoryState


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
        state: 'MemoryState',
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
    from hush.core.states import StateSchema, MemoryState
    from hush.core.nodes import GraphNode, START, END, PARENT
    from hush.core.nodes.transform.code_node import CodeNode

    def test(name: str, condition: bool):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        if not condition:
            raise AssertionError(f"Test failed: {name}")

    async def main():
        # =====================================================================
        # Test 1: BranchNode in a graph with state
        # =====================================================================
        print("\n" + "=" * 50)
        print("Test 1: BranchNode in a graph with state")
        print("=" * 50)

        with GraphNode(name="score_workflow") as graph:
            branch = BranchNode(
                name="router",
                cases={
                    "score >= 90": "excellent",
                    "score >= 70": "good",
                    "score >= 50": "pass",
                },
                default="fail",
                inputs={"score": PARENT["score"]}
            )
            START >> branch >> END

        graph.build()
        schema = StateSchema(graph)

        # Test with score=85
        state = MemoryState(schema, inputs={"score": 85})
        result = await branch.run(state)
        test("score=85 routes to 'good'", result["target"] == "good")
        test("matched is 'score >= 70'", result["matched"] == "score >= 70")

        # Verify state was updated
        test("state has target", state["score_workflow.router", "target", None] == "good")
        test("get_target works", branch.get_target(state) == "good")

        # Test with score=95
        state2 = MemoryState(schema, inputs={"score": 95})
        result = await branch.run(state2)
        test("score=95 routes to 'excellent'", result["target"] == "excellent")

        # Test with score=30 (default)
        state3 = MemoryState(schema, inputs={"score": 30})
        result = await branch.run(state3)
        test("score=30 routes to 'fail' (default)", result["target"] == "fail")

        # =====================================================================
        # Test 2: BranchNode with multiple variables from refs
        # =====================================================================
        print("\n" + "=" * 50)
        print("Test 2: BranchNode with refs to other nodes")
        print("=" * 50)

        with GraphNode(name="user_workflow") as graph2:
            user_data = CodeNode(
                name="user_data",
                code_fn=lambda: {"age": 25, "verified": True},
                inputs={}
            )

            branch2 = BranchNode(
                name="user_router",
                cases={
                    "age >= 18 and verified": "adult_verified",
                    "age >= 18": "adult_unverified",
                    "age >= 13": "teen",
                },
                default="child",
                inputs={
                    "age": user_data["age"],
                    "verified": user_data["verified"]
                }
            )

            START >> user_data >> branch2 >> END

        graph2.build()
        schema2 = StateSchema(graph2)
        state4 = MemoryState(schema2)

        # Inject user_data outputs (simulating execution)
        state4["user_workflow.user_data", "age", None] = 25
        state4["user_workflow.user_data", "verified", None] = True

        result = await branch2.run(state4)
        test("age=25, verified=True -> adult_verified", result["target"] == "adult_verified")

        # Test with different values
        state5 = MemoryState(schema2)
        state5["user_workflow.user_data", "age", None] = 15
        state5["user_workflow.user_data", "verified", None] = False

        result = await branch2.run(state5)
        test("age=15, verified=False -> teen", result["target"] == "teen")

        # =====================================================================
        # Test 3: Anchor override via state
        # =====================================================================
        print("\n" + "=" * 50)
        print("Test 3: Anchor override via state")
        print("=" * 50)

        with GraphNode(name="anchor_workflow") as graph3:
            branch3 = BranchNode(
                name="anchor_router",
                cases={
                    "status == 'active'": "process",
                },
                default="skip",
                inputs={
                    "status": PARENT["status"],
                    "anchor": PARENT["anchor"]
                }
            )
            START >> branch3 >> END

        graph3.build()
        schema3 = StateSchema(graph3)

        # With anchor set
        state6 = MemoryState(schema3, inputs={"status": "active", "anchor": "force_target"})
        result = await branch3.run(state6)
        test("anchor overrides condition", result["target"] == "force_target")
        test("matched is 'anchor'", result["matched"] == "anchor")

        # Without anchor
        state7 = MemoryState(schema3, inputs={"status": "active", "anchor": None})
        result = await branch3.run(state7)
        test("without anchor, condition works", result["target"] == "process")

        # =====================================================================
        # Test 4: Schema extraction
        # =====================================================================
        print("\n" + "=" * 50)
        print("Test 4: Schema extraction")
        print("=" * 50)

        test("router has 'score' in input_schema", "score" in branch.input_schema)
        test("router has 'anchor' in input_schema", "anchor" in branch.input_schema)
        test("router has 'target' in output_schema", "target" in branch.output_schema)
        test("router has 'matched' in output_schema", "matched" in branch.output_schema)

        test("user_router has 'age' in input_schema", "age" in branch2.input_schema)
        test("user_router has 'verified' in input_schema", "verified" in branch2.input_schema)

        # =====================================================================
        # Test 5: Quick __call__ test (for convenience)
        # =====================================================================
        print("\n" + "=" * 50)
        print("Test 5: Quick __call__ test")
        print("=" * 50)

        quick_branch = BranchNode(
            name="quick",
            cases={
                "x > 0": "positive",
                "x < 0": "negative",
            },
            default="zero"
        )

        result = quick_branch(x=5)
        test("x=5 -> positive", result["target"] == "positive")

        result = quick_branch(x=-3)
        test("x=-3 -> negative", result["target"] == "negative")

        result = quick_branch(x=0)
        test("x=0 -> zero (default)", result["target"] == "zero")

        # =====================================================================
        # Summary
        # =====================================================================
        print("\n" + "=" * 50)
        print("All BranchNode tests passed!")
        print("=" * 50)

    asyncio.run(main())
