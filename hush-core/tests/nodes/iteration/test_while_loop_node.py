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


# ============================================================
# Test 7: New Output Mapping Syntax with <<
# ============================================================

class TestWhileLoopNewOutputSyntax:
    """Test WhileLoopNode with new PARENT[...] << node[...] syntax."""

    @pytest.mark.asyncio
    async def test_ellipsis_forward_all_outputs(self):
        """Test PARENT[...] << node[...] forwards all outputs in WhileLoop.

        When using PARENT[...] << node[...], all node outputs are mapped to
        parent outputs with the same key names.
        """
        @code_node
        def compute_step(counter: int, total: int):
            new_counter = counter + 1
            new_total = total + counter
            return {"new_counter": new_counter, "new_total": new_total}

        with WhileLoopNode(
            name="ellipsis_while_loop",
            inputs={"counter": 0, "total": 0},
            stop_condition="counter >= 5",
            max_iterations=10
        ) as loop:
            node = compute_step(
                inputs={"counter": PARENT["counter"], "total": PARENT["total"]}
            )
            # Forward all outputs to parent with same key names
            PARENT[...] << node[...]
            # Map outputs to loop iteration variables (counter and total)
            PARENT["counter"] << node["new_counter"]
            PARENT["total"] << node["new_total"]
            START >> node >> END

        loop.build()
        schema = StateSchema(loop)
        state = MemoryState(schema)

        result = await loop.run(state)

        # Counter: 0->1->2->3->4->5, stops at 5 iterations
        assert result.get('iteration_metrics', {}).get('total_iterations') == 5
        # Total: 0+0=0, 0+1=1, 1+2=3, 3+3=6, 6+4=10
        # The outputs are mapped via PARENT["counter"]<<node["new_counter"],
        # so the loop output key is "counter" not "new_counter"
        assert result.get('total') == 10
        assert result.get('counter') == 5

    @pytest.mark.asyncio
    async def test_specific_key_mapping(self):
        """Test PARENT["key"] << node["key"] for specific mapping in WhileLoop.

        Maps node output "new_a" to parent output "a" (for loop variable update),
        and also creates explicit parent outputs "result_a" and "result_b".
        """
        @code_node
        def fibonacci_step(a: int, b: int):
            new_a = b
            new_b = a + b
            return {"new_a": new_a, "new_b": new_b}

        with WhileLoopNode(
            name="fibonacci_while_loop",
            inputs={"a": 0, "b": 1},
            stop_condition="b >= 21",
            max_iterations=20
        ) as loop:
            node = fibonacci_step(
                inputs={"a": PARENT["a"], "b": PARENT["b"]}
            )
            # Map to loop iteration variables
            PARENT["a"] << node["new_a"]
            PARENT["b"] << node["new_b"]
            START >> node >> END

        loop.build()
        schema = StateSchema(loop)
        state = MemoryState(schema)

        result = await loop.run(state)

        # Fibonacci: 0,1 -> 1,1 -> 1,2 -> 2,3 -> 3,5 -> 5,8 -> 8,13 -> 13,21
        # Stop when b >= 21, so 7 iterations
        assert result.get('iteration_metrics', {}).get('total_iterations') == 7
        # The mapped output keys are "a" and "b" (the PARENT keys)
        assert result.get('a') == 13
        assert result.get('b') == 21

    @pytest.mark.asyncio
    async def test_new_syntax_replaces_old_outputs(self):
        """Test new << syntax completely replaces old outputs= syntax."""
        @code_node
        def double_step(value: int):
            doubled = value * 2
            return {"doubled": doubled}

        with WhileLoopNode(
            name="new_syntax_loop",
            inputs={"value": 1},
            stop_condition="value >= 16",
            max_iterations=10
        ) as loop:
            node = double_step(
                inputs={"value": PARENT["value"]}
            )
            # New syntax only: map doubled output to loop's value variable
            PARENT["value"] << node["doubled"]
            START >> node >> END

        loop.build()
        schema = StateSchema(loop)
        state = MemoryState(schema)

        result = await loop.run(state)

        # Value: 1->2->4->8->16, stops at 4 iterations
        assert result.get('iteration_metrics', {}).get('total_iterations') == 4
        assert result.get('value') == 16
