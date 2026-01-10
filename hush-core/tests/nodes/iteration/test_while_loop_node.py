"""Tests for WhileLoopNode - conditional iteration node."""

import pytest
from hush.core.nodes.iteration.while_loop_node import WhileLoopNode
from hush.core.nodes.transform.code_node import code_node
from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.nodes.base import START, END, PARENT
from hush.core.states import StateSchema, MemoryState


# ============================================================
# Test 1: Simple Counter Loop
# ============================================================

class TestSimpleCounterLoop:
    """Test basic counter loop that stops at a threshold."""

    @pytest.mark.asyncio
    async def test_count_to_five(self):
        """Test counting loop that stops when counter >= 5."""
        @code_node
        def increment(counter: int):
            new_counter = counter + 1
            return {"new_counter": new_counter}

        with WhileLoopNode(
            name="counter_loop",
            inputs={"counter": 0},
            stop_condition="counter >= 5",
            max_iterations=10
        ) as loop:
            node = increment(
                inputs={"counter": PARENT["counter"]},
                outputs={"new_counter": PARENT["counter"]}
            )
            START >> node >> END

        loop.build()
        schema = StateSchema(loop)
        state = MemoryState(schema)

        result = await loop.run(state)

        assert result.get('iteration_metrics', {}).get('total_iterations') == 5
        assert result.get('iteration_metrics', {}).get('stopped_by_condition') is True


# ============================================================
# Test 2: Accumulator Loop
# ============================================================

class TestAccumulatorLoop:
    """Test accumulator loop that sums until threshold."""

    @pytest.mark.asyncio
    async def test_sum_until_100(self):
        """Test accumulator loop that stops when total >= 100."""
        @code_node
        def accumulate(total: int, step: int):
            new_total = total + step
            return {"new_total": new_total}

        with WhileLoopNode(
            name="accumulator_loop",
            inputs={"total": 0, "step": 15},
            max_iterations=10,
            stop_condition="total >= 100"
        ) as loop:
            node = accumulate(
                inputs={"total": PARENT["total"], "step": PARENT["step"]},
                outputs={"new_total": PARENT["total"]}
            )
            START >> node >> END

        loop.build()
        schema = StateSchema(loop)
        state = MemoryState(schema)

        result = await loop.run(state)

        # 0+15*7=105 >= 100, so 7 iterations
        assert result.get('iteration_metrics', {}).get('total_iterations') == 7


# ============================================================
# Test 3: Max Iterations Safety
# ============================================================

class TestMaxIterationsSafety:
    """Test that max_iterations prevents infinite loops."""

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self):
        """Test that loop stops at max_iterations when no stop_condition."""
        @code_node
        def infinite_loop(value: int):
            new_value = value + 1
            return {"new_value": new_value}

        with WhileLoopNode(
            name="safe_loop",
            max_iterations=5,
            inputs={"value": 0}
        ) as loop:
            node = infinite_loop(
                inputs={"value": PARENT["value"]},
                outputs={"new_value": PARENT["value"]}
            )
            START >> node >> END

        loop.build()
        schema = StateSchema(loop)
        state = MemoryState(schema)

        result = await loop.run(state)

        assert result.get('iteration_metrics', {}).get('max_iterations_reached') is True
        assert result.get('iteration_metrics', {}).get('total_iterations') == 5


# ============================================================
# Test 4: Complex Condition
# ============================================================

class TestComplexCondition:
    """Test complex stop conditions with multiple variables."""

    @pytest.mark.asyncio
    async def test_or_condition(self):
        """Test stop condition with OR: x >= 10 or done == True."""
        @code_node
        def complex_step(x: int, done: bool):
            new_x = x + 3
            new_done = new_x > 8
            return {"new_x": new_x, "new_done": new_done}

        with WhileLoopNode(
            name="complex_loop",
            inputs={"x": 0, "done": False},
            stop_condition="x >= 10 or done"
        ) as loop:
            node = complex_step(
                inputs={"x": PARENT["x"], "done": PARENT["done"]},
                outputs={"new_x": PARENT["x"], "new_done": PARENT["done"]}
            )
            START >> node >> END

        loop.build()
        schema = StateSchema(loop)
        state = MemoryState(schema)

        result = await loop.run(state)

        # 0->3->6->9 (done=True at 9>8), so 3 iterations
        assert result.get('iteration_metrics', {}).get('total_iterations') == 3


# ============================================================
# Test 5: WhileLoop in GraphNode with Ref
# ============================================================

class TestWhileLoopWithRef:
    """Test WhileLoopNode inside GraphNode with Ref inputs."""

    @pytest.mark.asyncio
    async def test_ref_inputs(self):
        """Test WhileLoop receiving inputs via Ref from upstream node."""
        @code_node
        def get_config():
            return {
                "config": {
                    "settings": {
                        "start_value": 0,
                        "increment": 3,
                        "limit": 10
                    }
                }
            }

        @code_node
        def increment_by(counter: int, step: int):
            new_counter = counter + step
            return {"new_counter": new_counter}

        with GraphNode(name="ref_while_graph") as graph:
            config_node = get_config()

            with WhileLoopNode(
                name="ref_loop",
                inputs={
                    "counter": config_node["config"]["settings"]["start_value"],
                    "step": config_node["config"]["settings"]["increment"]
                },
                stop_condition="counter >= 10"
            ) as loop:
                node = increment_by(
                    inputs={"counter": PARENT["counter"], "step": PARENT["step"]},
                    outputs={"new_counter": PARENT["counter"]}
                )
                START >> node >> END

            START >> config_node >> loop >> END

        graph.build()
        schema = StateSchema(graph)
        state = MemoryState(schema)

        await graph.run(state)

        loop_result = state[loop.full_name, "iteration_metrics", None]
        iterations = loop_result.get('total_iterations', 0) if loop_result else 0
        # 0->3->6->9->12, 4 iterations to reach counter >= 10
        assert iterations == 4


# ============================================================
# Test 6: Schema Extraction
# ============================================================

class TestSchemaExtraction:
    """Test that WhileLoopNode extracts inputs/outputs schema correctly."""

    def test_inputs_include_condition_vars(self):
        """Test that inputs include variables from stop_condition."""
        @code_node
        def increment(counter: int):
            return {"new_counter": counter + 1}

        with WhileLoopNode(
            name="schema_test_loop",
            inputs={"counter": 0},
            stop_condition="counter >= 5",
            max_iterations=10
        ) as loop:
            node = increment(
                inputs={"counter": PARENT["counter"]},
                outputs={"new_counter": PARENT["counter"]}
            )
            START >> node >> END

        loop.build()

        assert "counter" in loop.inputs
        assert "iteration_metrics" in loop.outputs
