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
