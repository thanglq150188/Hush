"""Tests for BranchNode - conditional routing node."""

import pytest
from hush.core.nodes.flow.branch_node import BranchNode
from hush.core.nodes.transform.code_node import CodeNode
from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.nodes.base import START, END, PARENT
from hush.core.states import StateSchema, MemoryState


# ============================================================
# Test 1: Basic Score Routing
# ============================================================

class TestBasicScoreRouting:
    """Test basic conditional routing based on score."""

    @pytest.fixture
    def score_graph(self):
        """Create a score routing graph."""
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
        return graph, branch

    @pytest.mark.asyncio
    async def test_score_85_routes_to_good(self, score_graph):
        """Test score=85 routes to 'good'."""
        graph, branch = score_graph
        schema = StateSchema(graph)
        state = MemoryState(schema, inputs={"score": 85})

        result = await branch.run(state)
        assert result["target"] == "good"
        assert result["matched"] == "score >= 70"

    @pytest.mark.asyncio
    async def test_score_95_routes_to_excellent(self, score_graph):
        """Test score=95 routes to 'excellent'."""
        graph, branch = score_graph
        schema = StateSchema(graph)
        state = MemoryState(schema, inputs={"score": 95})

        result = await branch.run(state)
        assert result["target"] == "excellent"

    @pytest.mark.asyncio
    async def test_score_30_routes_to_default(self, score_graph):
        """Test score=30 routes to default 'fail'."""
        graph, branch = score_graph
        schema = StateSchema(graph)
        state = MemoryState(schema, inputs={"score": 30})

        result = await branch.run(state)
        assert result["target"] == "fail"

    @pytest.mark.asyncio
    async def test_state_is_updated(self, score_graph):
        """Test that state is updated with routing result."""
        graph, branch = score_graph
        schema = StateSchema(graph)
        state = MemoryState(schema, inputs={"score": 85})

        await branch.run(state)
        assert state["score_workflow.router", "target", None] == "good"

    @pytest.mark.asyncio
    async def test_get_target_method(self, score_graph):
        """Test get_target method returns correct value."""
        graph, branch = score_graph
        schema = StateSchema(graph)
        state = MemoryState(schema, inputs={"score": 85})

        await branch.run(state)
        assert branch.get_target(state) == "good"


# ============================================================
# Test 2: Multiple Variables with Refs
# ============================================================

class TestMultipleVariablesRouting:
    """Test routing with multiple variables from refs."""

    @pytest.fixture
    def user_routing_graph(self):
        """Create a user routing graph with multiple conditions."""
        with GraphNode(name="user_workflow") as graph:
            user_data = CodeNode(
                name="user_data",
                code_fn=lambda: {"age": 25, "verified": True},
                inputs={}
            )

            branch = BranchNode(
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

            START >> user_data >> branch >> END

        graph.build()
        return graph, branch

    @pytest.mark.asyncio
    async def test_adult_verified_routing(self, user_routing_graph):
        """Test age=25, verified=True routes to adult_verified."""
        graph, branch = user_routing_graph
        schema = StateSchema(graph)
        state = MemoryState(schema)

        state["user_workflow.user_data", "age", None] = 25
        state["user_workflow.user_data", "verified", None] = True

        result = await branch.run(state)
        assert result["target"] == "adult_verified"

    @pytest.mark.asyncio
    async def test_teen_routing(self, user_routing_graph):
        """Test age=15, verified=False routes to teen."""
        graph, branch = user_routing_graph
        schema = StateSchema(graph)
        state = MemoryState(schema)

        state["user_workflow.user_data", "age", None] = 15
        state["user_workflow.user_data", "verified", None] = False

        result = await branch.run(state)
        assert result["target"] == "teen"


# ============================================================
# Test 3: Anchor Override
# ============================================================

class TestAnchorOverride:
    """Test anchor parameter overriding conditions."""

    @pytest.fixture
    def anchor_graph(self):
        """Create a graph with anchor support."""
        with GraphNode(name="anchor_workflow") as graph:
            branch = BranchNode(
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
            START >> branch >> END

        graph.build()
        return graph, branch

    @pytest.mark.asyncio
    async def test_anchor_overrides_condition(self, anchor_graph):
        """Test that anchor overrides normal condition evaluation."""
        graph, branch = anchor_graph
        schema = StateSchema(graph)
        state = MemoryState(schema, inputs={"status": "active", "anchor": "force_target"})

        result = await branch.run(state)
        assert result["target"] == "force_target"
        assert result["matched"] == "anchor"

    @pytest.mark.asyncio
    async def test_without_anchor_condition_works(self, anchor_graph):
        """Test normal condition when anchor is None."""
        graph, branch = anchor_graph
        schema = StateSchema(graph)
        state = MemoryState(schema, inputs={"status": "active", "anchor": None})

        result = await branch.run(state)
        assert result["target"] == "process"


# ============================================================
# Test 4: Schema Extraction
# ============================================================

class TestBranchSchemaExtraction:
    """Test automatic schema extraction from cases."""

    def test_inputs_from_conditions(self):
        """Test that inputs are extracted from condition variables."""
        branch = BranchNode(
            name="test",
            cases={
                "score >= 90": "excellent",
            },
            default="fail"
        )
        assert "score" in branch.inputs
        assert "anchor" in branch.inputs  # anchor is always present

    def test_outputs_have_target_and_matched(self):
        """Test that outputs include target and matched."""
        branch = BranchNode(
            name="test",
            cases={"x > 0": "positive"},
            default="zero"
        )
        assert "target" in branch.outputs
        assert "matched" in branch.outputs

    def test_multiple_condition_variables(self):
        """Test extraction of multiple variables from conditions."""
        branch = BranchNode(
            name="test",
            cases={
                "age >= 18 and verified": "adult",
            },
            default="child"
        )
        assert "age" in branch.inputs
        assert "verified" in branch.inputs


# ============================================================
# Test 5: Quick __call__ Test
# ============================================================

class TestBranchQuickCall:
    """Test direct __call__ invocation."""

    def test_positive_routing(self):
        """Test routing to positive."""
        branch = BranchNode(
            name="quick",
            cases={
                "x > 0": "positive",
                "x < 0": "negative",
            },
            default="zero"
        )
        result = branch(x=5)
        assert result["target"] == "positive"

    def test_negative_routing(self):
        """Test routing to negative."""
        branch = BranchNode(
            name="quick",
            cases={
                "x > 0": "positive",
                "x < 0": "negative",
            },
            default="zero"
        )
        result = branch(x=-3)
        assert result["target"] == "negative"

    def test_default_routing(self):
        """Test routing to default."""
        branch = BranchNode(
            name="quick",
            cases={
                "x > 0": "positive",
                "x < 0": "negative",
            },
            default="zero"
        )
        result = branch(x=0)
        assert result["target"] == "zero"


# ============================================================
# Test 6: Candidates Property
# ============================================================

class TestBranchCandidates:
    """Test candidates property."""

    def test_candidates_from_cases(self):
        """Test that candidates are derived from cases and default."""
        branch = BranchNode(
            name="test",
            cases={
                "x > 0": "positive",
                "x < 0": "negative",
            },
            default="zero"
        )
        candidates = branch.candidates
        assert "positive" in candidates
        assert "negative" in candidates
        assert "zero" in candidates

    def test_explicit_candidates(self):
        """Test explicit candidates override."""
        branch = BranchNode(
            name="test",
            cases={"x > 0": "a"},
            default="b",
            candidates=["c", "d", "e"]
        )
        assert branch.candidates == ["c", "d", "e"]


# ============================================================
# Test 7: Fluent Branch Builder Syntax
# ============================================================

from hush.core.nodes.flow.branch_node import Branch


class TestBranchFluentBuilder:
    """Test fluent Branch builder với if_/otherwise syntax."""

    def test_basic_fluent_syntax(self):
        """Test basic fluent syntax: Branch().if_().otherwise()"""
        branch = (Branch("router")
            .if_(PARENT["score"] >= 90, "excellent")
            .if_(PARENT["score"] >= 70, "good")
            .if_(PARENT["score"] >= 50, "pass")
            .otherwise("fail"))

        # Verify it's a BranchNode
        assert isinstance(branch, BranchNode)
        assert branch.name == "router"

        # Verify ref_cases were created (fluent builder uses ref_cases)
        assert len(branch.ref_cases) == 3
        assert branch.default == "fail"

        # Test routing
        result = branch(score=95)
        assert result["target"] == "excellent"

        result = branch(score=75)
        assert result["target"] == "good"

        result = branch(score=55)
        assert result["target"] == "pass"

        result = branch(score=30)
        assert result["target"] == "fail"

    def test_comparison_operators(self):
        """Test various comparison operators in conditions."""
        branch = (Branch("comparisons")
            .if_(PARENT["x"] == 0, "zero")
            .if_(PARENT["x"] < 0, "negative")
            .if_(PARENT["x"] > 100, "large")
            .if_(PARENT["x"] <= 10, "small")
            .if_(PARENT["x"] >= 50, "medium_large")
            .otherwise("medium"))

        assert branch(x=0)["target"] == "zero"
        assert branch(x=-5)["target"] == "negative"
        assert branch(x=150)["target"] == "large"
        assert branch(x=5)["target"] == "small"
        assert branch(x=75)["target"] == "medium_large"
        assert branch(x=30)["target"] == "medium"

    def test_fluent_with_node_targets(self):
        """Test fluent syntax with node objects as targets."""
        with GraphNode(name="fluent_graph") as graph:
            excellent = CodeNode(
                name="excellent",
                code_fn=lambda: {"grade": "A"},
                inputs={}
            )
            good = CodeNode(
                name="good",
                code_fn=lambda: {"grade": "B"},
                inputs={}
            )
            fail = CodeNode(
                name="fail",
                code_fn=lambda: {"grade": "F"},
                inputs={}
            )

            branch = (Branch("grader")
                .if_(PARENT["score"] >= 90, excellent)
                .if_(PARENT["score"] >= 70, good)
                .otherwise(fail))

            START >> branch
            branch >> [excellent, good, fail]
            [excellent, good, fail] >> END

        graph.build()

        # Verify node names were extracted
        assert "excellent" in branch.candidates
        assert "good" in branch.candidates
        assert "fail" in branch.candidates

    def test_fluent_without_otherwise(self):
        """Test fluent syntax without default (using build())."""
        branch = (Branch("no_default")
            .if_(PARENT["status"] == "active", "process")
            .if_(PARENT["status"] == "pending", "queue")
            .build())

        assert isinstance(branch, BranchNode)
        assert branch.default is None

        result = branch(status="active")
        assert result["target"] == "process"

        result = branch(status="pending")
        assert result["target"] == "queue"

        # Unknown status returns None
        result = branch(status="unknown")
        assert result["target"] is None

    @pytest.mark.asyncio
    async def test_fluent_in_graph_execution(self):
        """Test fluent Branch in full graph execution."""
        with GraphNode(name="fluent_workflow") as graph:
            branch = (Branch("scorer")
                .if_(PARENT["score"] >= 90, "excellent")
                .if_(PARENT["score"] >= 70, "good")
                .otherwise("fail"))

            excellent = CodeNode(
                name="excellent",
                code_fn=lambda: {"result": "A grade!"},
                inputs={},
                outputs={"*": PARENT}
            )
            good = CodeNode(
                name="good",
                code_fn=lambda: {"result": "B grade!"},
                inputs={},
                outputs={"*": PARENT}
            )
            fail = CodeNode(
                name="fail",
                code_fn=lambda: {"result": "Try again!"},
                inputs={},
                outputs={"*": PARENT}
            )

            # Each branch target goes to END independently
            START >> branch
            branch >> [excellent, good, fail]
            [excellent, good, fail] >> END

        graph.build()
        schema = StateSchema(graph)

        # Test score 95 → excellent
        state = schema.create_state(inputs={"score": 95})
        result = await graph.run(state)
        assert result["result"] == "A grade!"

        # Test score 75 → good
        state = schema.create_state(inputs={"score": 75})
        result = await graph.run(state)
        assert result["result"] == "B grade!"

        # Test score 40 → fail
        state = schema.create_state(inputs={"score": 40})
        result = await graph.run(state)
        assert result["result"] == "Try again!"

    def test_fluent_inputs_are_refs(self):
        """Test that fluent builder correctly creates Ref inputs."""
        branch = (Branch("ref_test")
            .if_(PARENT["value"] >= 100, "high")
            .otherwise("low"))

        # Verify input is a Ref
        from hush.core.states.ref import Ref
        assert "value" in branch.inputs
        assert isinstance(branch.inputs["value"].value, Ref)

    def test_fluent_ne_operator(self):
        """Test != operator in fluent syntax."""
        branch = (Branch("ne_test")
            .if_(PARENT["status"] != "disabled", "active")
            .otherwise("inactive"))

        assert branch(status="enabled")["target"] == "active"
        assert branch(status="disabled")["target"] == "inactive"

    def test_fluent_with_apply(self):
        """Test fluent syntax with .apply() for custom functions."""
        def is_high_score(score):
            return score >= 90

        def is_passing(score):
            return score >= 50

        branch = (Branch("apply_router")
            .if_(PARENT["score"].apply(is_high_score), "excellent")
            .if_(PARENT["score"].apply(is_passing), "pass")
            .otherwise("fail"))

        # Test routing with custom functions
        assert branch(score=95)["target"] == "excellent"
        assert branch(score=75)["target"] == "pass"
        assert branch(score=30)["target"] == "fail"

    def test_fluent_apply_with_args(self):
        """Test .apply() with additional arguments."""
        def is_above_threshold(value, threshold):
            return value > threshold

        branch = (Branch("threshold_router")
            .if_(PARENT["value"].apply(is_above_threshold, 100), "high")
            .if_(PARENT["value"].apply(is_above_threshold, 50), "medium")
            .otherwise("low"))

        assert branch(value=150)["target"] == "high"
        assert branch(value=75)["target"] == "medium"
        assert branch(value=25)["target"] == "low"

    def test_fluent_apply_with_builtin_functions(self):
        """Test .apply() with builtin functions like len."""
        branch = (Branch("len_router")
            .if_(PARENT["x"].apply(len) > 5, "long")
            .if_(PARENT["x"].apply(len) > 2, "medium")
            .otherwise("short"))

        assert branch(x="hello!")["target"] == "long"  # len=6
        assert branch(x="hello")["target"] == "medium"  # len=5
        assert branch(x="hi")["target"] == "short"  # len=2
        assert branch(x=[1, 2, 3, 4, 5, 6])["target"] == "long"  # len=6
