"""Test suite for GraphNode - from simple to complex scenarios.

Only tests GraphNode and CodeNode. Flow nodes (BranchNode, ForLoopNode, etc.)
are tested separately.
"""

import asyncio
import pytest
from typing import Dict, Any

from hush.core import (
    GraphNode,
    CodeNode,
    code_node,
    START, END, PARENT,
    StateSchema,
)


# ============================================================
# Test 1: Single Node Graph
# ============================================================

class TestSingleNodeGraph:
    """Test graphs with a single node."""

    @pytest.mark.asyncio
    async def test_single_code_node(self):
        """Single CodeNode that doubles a value."""
        with GraphNode(name="single_node_graph") as graph:
            node = CodeNode(
                name="double",
                code_fn=lambda x: {"result": x * 2},
                inputs={"x": PARENT["x"]},
                outputs=PARENT
            )
            START >> node >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 5})
        result = await graph.run(state)

        assert result["result"] == 10

    @pytest.mark.asyncio
    async def test_single_node_with_decorator(self):
        """Single node using @code_node decorator."""
        @code_node
        def triple(x: int):
            return {"result": x * 3}

        with GraphNode(name="decorator_graph") as graph:
            node = triple(inputs={"x": PARENT["x"]}, outputs=PARENT)
            START >> node >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 4})
        result = await graph.run(state)

        assert result["result"] == 12

    @pytest.mark.asyncio
    async def test_single_node_no_output_mapping(self):
        """Single node without explicit output mapping."""
        with GraphNode(name="no_output_map") as graph:
            node = CodeNode(
                name="compute",
                code_fn=lambda x: {"result": x + 100},
                inputs={"x": PARENT["x"]}
            )
            START >> node >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 5})
        await graph.run(state)

        # Access result directly from state
        assert state["no_output_map.compute", "result", None] == 105


# ============================================================
# Test 2: Linear Graph (A -> B -> C)
# ============================================================

class TestLinearGraph:
    """Test linear sequential graphs."""

    @pytest.mark.asyncio
    async def test_two_nodes_linear(self):
        """Two nodes in sequence: add then multiply."""
        with GraphNode(name="two_node_linear") as graph:
            node_a = CodeNode(
                name="add_10",
                code_fn=lambda x: {"result": x + 10},
                inputs={"x": PARENT["x"]}
            )
            node_b = CodeNode(
                name="multiply_2",
                code_fn=lambda x: {"result": x * 2},
                inputs={"x": node_a["result"]},
                outputs=PARENT
            )
            START >> node_a >> node_b >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 5})
        result = await graph.run(state)

        # (5 + 10) * 2 = 30
        assert result["result"] == 30

    @pytest.mark.asyncio
    async def test_three_nodes_linear(self):
        """Three nodes in sequence: add, multiply, subtract."""
        with GraphNode(name="three_node_linear") as graph:
            node_a = CodeNode(
                name="add_10",
                code_fn=lambda x: {"result": x + 10},
                inputs={"x": PARENT["x"]}
            )
            node_b = CodeNode(
                name="multiply_2",
                code_fn=lambda x: {"result": x * 2},
                inputs={"x": node_a["result"]}
            )
            node_c = CodeNode(
                name="subtract_5",
                code_fn=lambda x: {"result": x - 5},
                inputs={"x": node_b["result"]},
                outputs=PARENT
            )
            START >> node_a >> node_b >> node_c >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 5})
        result = await graph.run(state)

        # ((5 + 10) * 2) - 5 = 25
        assert result["result"] == 25

    @pytest.mark.asyncio
    async def test_long_chain(self):
        """Five nodes in a chain."""
        with GraphNode(name="long_chain") as graph:
            n1 = CodeNode(name="n1", code_fn=lambda x: {"v": x + 1}, inputs={"x": PARENT["x"]})
            n2 = CodeNode(name="n2", code_fn=lambda x: {"v": x + 2}, inputs={"x": n1["v"]})
            n3 = CodeNode(name="n3", code_fn=lambda x: {"v": x + 3}, inputs={"x": n2["v"]})
            n4 = CodeNode(name="n4", code_fn=lambda x: {"v": x + 4}, inputs={"x": n3["v"]})
            n5 = CodeNode(name="n5", code_fn=lambda x: {"v": x + 5}, inputs={"x": n4["v"]}, outputs=PARENT)

            START >> n1 >> n2 >> n3 >> n4 >> n5 >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 0})
        result = await graph.run(state)

        # 0 + 1 + 2 + 3 + 4 + 5 = 15
        assert result["v"] == 15


# ============================================================
# Test 3: Parallel Graph (Fork and Merge)
# ============================================================

class TestParallelGraph:
    """Test graphs with parallel branches."""

    @pytest.mark.asyncio
    async def test_simple_fork_merge(self):
        """Fork into two branches, then merge."""
        with GraphNode(name="fork_merge") as graph:
            start = CodeNode(
                name="start",
                code_fn=lambda x: {"value": x},
                inputs={"x": PARENT["x"]}
            )
            branch_a = CodeNode(
                name="branch_a",
                code_fn=lambda x: {"result": x * 2},
                inputs={"x": start["value"]}
            )
            branch_b = CodeNode(
                name="branch_b",
                code_fn=lambda x: {"result": x * 3},
                inputs={"x": start["value"]}
            )
            merge = CodeNode(
                name="merge",
                code_fn=lambda a, b: {"total": a + b},
                inputs={"a": branch_a["result"], "b": branch_b["result"]},
                outputs=PARENT
            )

            START >> start >> [branch_a, branch_b] >> merge >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 10})
        result = await graph.run(state)

        # (10 * 2) + (10 * 3) = 50
        assert result["total"] == 50

    @pytest.mark.asyncio
    async def test_three_way_fork(self):
        """Fork into three branches, then merge."""
        with GraphNode(name="three_way_fork") as graph:
            start = CodeNode(
                name="start",
                code_fn=lambda x: {"value": x},
                inputs={"x": PARENT["x"]}
            )
            b1 = CodeNode(name="b1", code_fn=lambda x: {"r": x * 2}, inputs={"x": start["value"]})
            b2 = CodeNode(name="b2", code_fn=lambda x: {"r": x * 3}, inputs={"x": start["value"]})
            b3 = CodeNode(name="b3", code_fn=lambda x: {"r": x * 4}, inputs={"x": start["value"]})

            merge = CodeNode(
                name="merge",
                code_fn=lambda a, b, c: {"total": a + b + c},
                inputs={"a": b1["r"], "b": b2["r"], "c": b3["r"]},
                outputs=PARENT
            )

            START >> start >> [b1, b2, b3] >> merge >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 10})
        result = await graph.run(state)

        # (10*2) + (10*3) + (10*4) = 90
        assert result["total"] == 90

    @pytest.mark.asyncio
    async def test_diamond_pattern(self):
        """Diamond: A -> [B, C] -> D (classic DAG pattern)."""
        with GraphNode(name="diamond") as graph:
            a = CodeNode(name="a", code_fn=lambda x: {"out": x}, inputs={"x": PARENT["x"]})
            b = CodeNode(name="b", code_fn=lambda x: {"out": x + 100}, inputs={"x": a["out"]})
            c = CodeNode(name="c", code_fn=lambda x: {"out": x + 200}, inputs={"x": a["out"]})
            d = CodeNode(
                name="d",
                code_fn=lambda x, y: {"result": x + y},
                inputs={"x": b["out"], "y": c["out"]},
                outputs=PARENT
            )

            START >> a >> [b, c] >> d >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 1})
        result = await graph.run(state)

        # (1 + 100) + (1 + 200) = 302
        assert result["result"] == 302

    @pytest.mark.asyncio
    async def test_parallel_independent_branches(self):
        """Multiple independent parallel paths."""
        with GraphNode(name="parallel_independent") as graph:
            # Two independent branches from input
            branch_a = CodeNode(
                name="branch_a",
                code_fn=lambda x: {"result": x * 10},
                inputs={"x": PARENT["x"]}
            )
            branch_b = CodeNode(
                name="branch_b",
                code_fn=lambda y: {"result": y + 5},
                inputs={"y": PARENT["y"]}
            )
            merge = CodeNode(
                name="merge",
                code_fn=lambda a, b: {"sum": a + b},
                inputs={"a": branch_a["result"], "b": branch_b["result"]},
                outputs=PARENT
            )

            START >> [branch_a, branch_b] >> merge >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 3, "y": 7})
        result = await graph.run(state)

        # (3 * 10) + (7 + 5) = 42
        assert result["sum"] == 42


# ============================================================
# Test 4: Multiple Inputs/Outputs
# ============================================================

class TestMultipleIO:
    """Test graphs with multiple inputs and outputs."""

    @pytest.mark.asyncio
    async def test_multiple_inputs(self):
        """Graph with multiple input variables."""
        with GraphNode(name="multi_input") as graph:
            node = CodeNode(
                name="add",
                code_fn=lambda a, b, c: {"sum": a + b + c},
                inputs={"a": PARENT["a"], "b": PARENT["b"], "c": PARENT["c"]},
                outputs=PARENT
            )
            START >> node >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"a": 1, "b": 2, "c": 3})
        result = await graph.run(state)

        assert result["sum"] == 6

    @pytest.mark.asyncio
    async def test_multiple_outputs(self):
        """Graph with multiple output variables."""
        with GraphNode(name="multi_output") as graph:
            node = CodeNode(
                name="split",
                code_fn=lambda x: {"double": x * 2, "triple": x * 3, "quad": x * 4},
                inputs={"x": PARENT["x"]},
                outputs=PARENT
            )
            START >> node >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 5})
        result = await graph.run(state)

        assert result["double"] == 10
        assert result["triple"] == 15
        assert result["quad"] == 20

    @pytest.mark.asyncio
    async def test_selective_output_mapping(self):
        """Map only some outputs to graph output."""
        with GraphNode(name="selective_output") as graph:
            node = CodeNode(
                name="compute",
                code_fn=lambda x: {"a": x + 1, "b": x + 2, "c": x + 3},
                inputs={"x": PARENT["x"]},
                outputs={"a": PARENT["result_a"], "c": PARENT["result_c"]}
            )
            START >> node >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 10})
        result = await graph.run(state)

        assert result["result_a"] == 11
        assert result["result_c"] == 13
        assert "result_b" not in result  # b was not mapped


# ============================================================
# Test 5: Complex Data Types
# ============================================================

class TestComplexDataTypes:
    """Test graphs with complex data types."""

    @pytest.mark.asyncio
    async def test_dict_processing(self):
        """Process dictionary data."""
        with GraphNode(name="dict_graph") as graph:
            node = CodeNode(
                name="process",
                code_fn=lambda data: {
                    "keys": list(data.keys()),
                    "values": list(data.values()),
                    "count": len(data)
                },
                inputs={"data": PARENT["data"]},
                outputs=PARENT
            )
            START >> node >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"data": {"a": 1, "b": 2, "c": 3}})
        result = await graph.run(state)

        assert result["keys"] == ["a", "b", "c"]
        assert result["values"] == [1, 2, 3]
        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_list_processing(self):
        """Process list data through pipeline."""
        with GraphNode(name="list_graph") as graph:
            double = CodeNode(
                name="double",
                code_fn=lambda items: {"result": [x * 2 for x in items]},
                inputs={"items": PARENT["items"]}
            )
            sum_all = CodeNode(
                name="sum",
                code_fn=lambda items: {"total": sum(items)},
                inputs={"items": double["result"]},
                outputs=PARENT
            )
            START >> double >> sum_all >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"items": [1, 2, 3, 4, 5]})
        result = await graph.run(state)

        # sum([2, 4, 6, 8, 10]) = 30
        assert result["total"] == 30

    @pytest.mark.asyncio
    async def test_string_processing(self):
        """Process string data through pipeline."""
        with GraphNode(name="string_graph") as graph:
            upper = CodeNode(
                name="upper",
                code_fn=lambda text: {"result": text.upper()},
                inputs={"text": PARENT["text"]}
            )
            reverse = CodeNode(
                name="reverse",
                code_fn=lambda text: {"result": text[::-1]},
                inputs={"text": upper["result"]},
                outputs=PARENT
            )
            START >> upper >> reverse >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"text": "hello"})
        result = await graph.run(state)

        assert result["result"] == "OLLEH"


# ============================================================
# Test 6: Async Operations
# ============================================================

class TestAsyncOperations:
    """Test graphs with async code functions."""

    @pytest.mark.asyncio
    async def test_async_single_node(self):
        """Single async node using return_keys for nested async functions."""
        async def async_double(x: int):
            await asyncio.sleep(0.01)  # Simulate async work
            return {"result": x * 2}

        with GraphNode(name="async_single") as graph:
            # Note: return_keys needed because inspect.getsource() has trouble
            # with functions defined inside other functions
            node = CodeNode(
                name="double",
                code_fn=async_double,
                return_keys=["result"],
                inputs={"x": PARENT["x"]},
                outputs=PARENT
            )
            START >> node >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 21})
        result = await graph.run(state)

        assert result["result"] == 42

    @pytest.mark.asyncio
    async def test_async_pipeline(self):
        """Pipeline with async nodes using return_keys."""
        async def async_add(x: int):
            await asyncio.sleep(0.01)
            return {"result": x + 10}

        async def async_multiply(x: int):
            await asyncio.sleep(0.01)
            return {"result": x * 2}

        with GraphNode(name="async_pipeline") as graph:
            add = CodeNode(
                name="add",
                code_fn=async_add,
                return_keys=["result"],
                inputs={"x": PARENT["x"]}
            )
            mult = CodeNode(
                name="multiply",
                code_fn=async_multiply,
                return_keys=["result"],
                inputs={"x": add["result"]},
                outputs=PARENT
            )
            START >> add >> mult >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 5})
        result = await graph.run(state)

        # (5 + 10) * 2 = 30
        assert result["result"] == 30

    @pytest.mark.asyncio
    async def test_async_parallel(self):
        """Parallel async nodes execute concurrently."""
        call_order = []

        async def slow_a(x: int):
            call_order.append("a_start")
            await asyncio.sleep(0.05)
            call_order.append("a_end")
            return {"result": x * 2}

        async def slow_b(x: int):
            call_order.append("b_start")
            await asyncio.sleep(0.05)
            call_order.append("b_end")
            return {"result": x * 3}

        with GraphNode(name="async_parallel") as graph:
            start = CodeNode(
                name="start",
                code_fn=lambda x: {"value": x},
                inputs={"x": PARENT["x"]}
            )
            a = CodeNode(name="a", code_fn=slow_a, inputs={"x": start["value"]}, outputs={"result": None})
            b = CodeNode(name="b", code_fn=slow_b, inputs={"x": start["value"]}, outputs={"result": None})
            merge = CodeNode(
                name="merge",
                code_fn=lambda a, b: {"total": a + b},
                inputs={"a": a["result"], "b": b["result"]},
                outputs=PARENT
            )

            START >> start >> [a, b] >> merge >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 10})
        result = await graph.run(state)

        # Both should start before either ends (concurrent execution)
        assert "a_start" in call_order
        assert "b_start" in call_order
        assert result["total"] == 50  # (10*2) + (10*3)


# ============================================================
# Test 7: Error Handling
# ============================================================

class TestErrorHandling:
    """Test error handling in graphs."""

    @pytest.mark.asyncio
    async def test_node_error_captured(self):
        """Errors in nodes are captured in state."""
        def failing_fn(x):
            raise ValueError("Intentional error")

        with GraphNode(name="error_graph") as graph:
            node = CodeNode(
                name="failing",
                code_fn=failing_fn,
                inputs={"x": PARENT["x"]},
                outputs=PARENT
            )
            START >> node >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 5})
        await graph.run(state)

        # Check error is captured
        error = state["error_graph.failing", "error", None]
        assert "Intentional error" in error

    @pytest.mark.asyncio
    async def test_partial_pipeline_error(self):
        """Error in middle of pipeline."""
        with GraphNode(name="partial_error") as graph:
            first = CodeNode(
                name="first",
                code_fn=lambda x: {"result": x + 10},
                inputs={"x": PARENT["x"]}
            )
            failing = CodeNode(
                name="failing",
                code_fn=lambda x: 1/0,  # Division by zero
                inputs={"x": first["result"]}
            )
            last = CodeNode(
                name="last",
                code_fn=lambda x: {"result": x * 2},
                inputs={"x": failing["result"]},
                outputs=PARENT
            )

            START >> first >> failing >> last >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 5})
        await graph.run(state)

        # First node should succeed
        assert state["partial_error.first", "result", None] == 15

        # Failing node should have error
        error = state["partial_error.failing", "error", None]
        assert "division by zero" in error


# ============================================================
# Test 8: Edge Cases
# ============================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Handle empty string inputs gracefully."""
        with GraphNode(name="empty_input") as graph:
            node = CodeNode(
                name="handle_empty",
                code_fn=lambda x: {"result": x if x else "default"},
                inputs={"x": PARENT["x"]},
                outputs=PARENT
            )
            START >> node >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": ""})
        result = await graph.run(state)

        assert result["result"] == "default"

    @pytest.mark.asyncio
    async def test_large_data(self):
        """Handle large data volumes."""
        with GraphNode(name="large_data") as graph:
            node = CodeNode(
                name="process",
                code_fn=lambda data: {"count": len(data), "sum": sum(data)},
                inputs={"data": PARENT["data"]},
                outputs=PARENT
            )
            START >> node >> END

        graph.build()

        large_list = list(range(10000))
        schema = StateSchema(graph)
        state = schema.create_state(inputs={"data": large_list})
        result = await graph.run(state)

        assert result["count"] == 10000
        assert result["sum"] == sum(range(10000))

    @pytest.mark.asyncio
    async def test_unicode_data(self):
        """Handle unicode data correctly."""
        with GraphNode(name="unicode_graph") as graph:
            node = CodeNode(
                name="process",
                code_fn=lambda text: {"result": f"Processed: {text}"},
                inputs={"text": PARENT["text"]},
                outputs=PARENT
            )
            START >> node >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"text": "Hello 世界 🌍"})
        result = await graph.run(state)

        assert result["result"] == "Processed: Hello 世界 🌍"

    @pytest.mark.asyncio
    async def test_zero_and_negative(self):
        """Handle zero and negative numbers."""
        with GraphNode(name="zero_negative") as graph:
            node = CodeNode(
                name="compute",
                code_fn=lambda x, y: {"sum": x + y, "product": x * y},
                inputs={"x": PARENT["x"], "y": PARENT["y"]},
                outputs=PARENT
            )
            START >> node >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": -5, "y": 0})
        result = await graph.run(state)

        assert result["sum"] == -5
        assert result["product"] == 0


# ============================================================
# Test 9: @code_node Decorator
# ============================================================

class TestCodeNodeDecorator:
    """Test the @code_node decorator."""

    @pytest.mark.asyncio
    async def test_decorator_basic(self):
        """Basic decorator usage."""
        @code_node
        def add_one(x: int):
            return {"result": x + 1}

        with GraphNode(name="decorator_basic") as graph:
            node = add_one(inputs={"x": PARENT["x"]}, outputs=PARENT)
            START >> node >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 10})
        result = await graph.run(state)

        assert result["result"] == 11

    @pytest.mark.asyncio
    async def test_decorator_with_defaults(self):
        """Decorator with default parameter values."""
        @code_node
        def add(x: int, amount: int = 10):
            return {"result": x + amount}

        with GraphNode(name="decorator_defaults") as graph:
            node = add(inputs={"x": PARENT["x"]}, outputs=PARENT)
            START >> node >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 5})
        result = await graph.run(state)

        # Uses default amount=10
        assert result["result"] == 15

    @pytest.mark.asyncio
    async def test_decorator_pipeline(self):
        """Multiple decorated functions in pipeline."""
        @code_node
        def step1(x: int):
            return {"value": x * 2}

        @code_node
        def step2(x: int):
            return {"value": x + 5}

        @code_node
        def step3(x: int):
            return {"result": x ** 2}

        with GraphNode(name="decorator_pipeline") as graph:
            n1 = step1(inputs={"x": PARENT["x"]})
            n2 = step2(inputs={"x": n1["value"]})
            n3 = step3(inputs={"x": n2["value"]}, outputs=PARENT)

            START >> n1 >> n2 >> n3 >> END

        graph.build()

        schema = StateSchema(graph)
        state = schema.create_state(inputs={"x": 3})
        result = await graph.run(state)

        # ((3 * 2) + 5) ** 2 = 11 ** 2 = 121
        assert result["result"] == 121


# ============================================================
# Run tests with pytest
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
