"""Workflow integration tests for hush-observability.

This module tests the complete observability flow using Hush engine
with multiple node types: CodeNode, BranchNode, ParserNode.

The tests verify that:
1. Trace metadata is correctly collected during workflow execution
2. Execution order is properly recorded
3. Tracers can successfully flush to observability backends (Langfuse)
"""

import uuid
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from hush.core import Hush, GraphNode, START, END, PARENT
from hush.core.nodes import CodeNode, BranchNode, ParserNode, code_node


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_langfuse_client():
    """Create a mock LangfuseClient for testing flush operations."""
    mock_client = MagicMock()
    mock_trace = MagicMock()
    mock_span = MagicMock()
    mock_generation = MagicMock()

    mock_trace.span.return_value = mock_span
    mock_trace.generation.return_value = mock_generation
    mock_span.span.return_value = mock_span
    mock_span.generation.return_value = mock_generation
    mock_client.trace.return_value = mock_trace
    mock_trace.get_trace_url.return_value = "https://cloud.langfuse.com/trace/test-123"

    return mock_client


# ============================================================================
# CodeNode Tests
# ============================================================================

class TestCodeNodeWorkflow:
    """Test CodeNode execution with trace metadata collection."""

    @pytest.mark.asyncio
    async def test_simple_code_node_workflow(self):
        """Test a simple workflow with a single CodeNode."""
        with GraphNode(name="simple-code-workflow") as graph:
            code = CodeNode(
                name="processor",
                code_fn=lambda text: {"processed": text.upper(), "length": len(text)},
                inputs={"text": PARENT["text"]},
                outputs=PARENT
            )
            START >> code >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"text": "hello world"})

        # Verify outputs
        assert result["processed"] == "HELLO WORLD"
        assert result["length"] == 11
        # Verify $state is included
        assert "$state" in result

    @pytest.mark.asyncio
    async def test_code_node_trace_metadata(self):
        """Test that CodeNode correctly records trace metadata."""
        with GraphNode(name="code-trace-test") as graph:
            code = CodeNode(
                name="adder",
                code_fn=lambda a, b: {"sum": a + b},
                inputs={"a": PARENT["a"], "b": PARENT["b"]},
                outputs=PARENT
            )
            START >> code >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"a": 5, "b": 3},
            request_id="test-request-123"
        )

        # Access state via $state key
        state = result["$state"]

        # Check execution order
        assert len(state.execution_order) >= 1

        # Check trace metadata
        trace_metadata = state.trace_metadata
        assert len(trace_metadata) >= 1

        # Find the adder node metadata
        adder_key = None
        for key in trace_metadata:
            if "adder" in key:
                adder_key = key
                break

        assert adder_key is not None
        metadata = trace_metadata[adder_key]
        assert metadata["name"] == "adder"
        assert "a" in metadata["input_vars"]
        assert "b" in metadata["input_vars"]
        assert "sum" in metadata["output_vars"]

    @pytest.mark.asyncio
    async def test_chained_code_nodes(self):
        """Test workflow with multiple chained CodeNodes."""
        with GraphNode(name="chained-workflow") as graph:
            node1 = CodeNode(name="double", code_fn=lambda x: {"y": x * 2}, inputs={"x": PARENT["x"]})
            node2 = CodeNode(name="add_ten", code_fn=lambda y: {"z": y + 10}, inputs={"y": node1["y"]})
            node3 = CodeNode(name="square", code_fn=lambda z: {"result": z ** 2}, inputs={"z": node2["z"]}, outputs=PARENT)

            START >> node1 >> node2 >> node3 >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"x": 5})

        # x=5 -> y=10 -> z=20 -> result=400
        assert result["result"] == 400


# ============================================================================
# BranchNode Tests
# ============================================================================

class TestBranchNodeWorkflow:
    """Test BranchNode execution with conditional routing."""

    @pytest.mark.asyncio
    async def test_simple_branch_workflow(self):
        """Test a workflow with BranchNode for conditional routing."""
        with GraphNode(name="branch-workflow") as graph:
            branch = BranchNode(
                name="router",
                cases={
                    "score >= 70": "high_path",
                },
                default="low_path",
                inputs={"score": PARENT["score"]}
            )

            high = CodeNode(
                name="high_path",
                code_fn=lambda score: {"grade": "A", "message": "Excellent!"},
                inputs={"score": PARENT["score"]},
                outputs=PARENT
            )

            low = CodeNode(
                name="low_path",
                code_fn=lambda score: {"grade": "F", "message": "Try harder!"},
                inputs={"score": PARENT["score"]},
                outputs=PARENT
            )

            START >> branch
            branch > [high, low] > END

        engine = Hush(graph)

        # Test high score path
        result_high = await engine.run(inputs={"score": 85})
        assert result_high["grade"] == "A"

        # Test low score path
        result_low = await engine.run(inputs={"score": 50})
        assert result_low["grade"] == "F"

    @pytest.mark.asyncio
    async def test_branch_trace_metadata(self):
        """Test that BranchNode correctly records trace metadata."""
        with GraphNode(name="branch-trace-test") as graph:
            branch = BranchNode(
                name="checker",
                cases={
                    "value > 0": "positive",
                },
                default="negative",
                inputs={"value": PARENT["value"]}
            )

            pos = CodeNode(
                name="positive",
                code_fn=lambda value: {"output": value},
                inputs={"value": PARENT["value"]},
                outputs=PARENT
            )
            neg = CodeNode(
                name="negative",
                code_fn=lambda value: {"output": value},
                inputs={"value": PARENT["value"]},
                outputs=PARENT
            )

            START >> branch
            branch > [pos, neg] > END

        engine = Hush(graph)
        result = await engine.run(inputs={"value": 10}, request_id="branch-test-123")

        # Check that branch node recorded metadata
        state = result["$state"]
        trace_metadata = state.trace_metadata
        branch_found = False
        for key, metadata in trace_metadata.items():
            if "checker" in key:
                branch_found = True
                assert "value" in metadata["input_vars"] or "anchor" in metadata["input_vars"]
                break

        assert branch_found, "BranchNode should record trace metadata"


# ============================================================================
# ParserNode Tests
# ============================================================================

class TestParserNodeWorkflow:
    """Test ParserNode execution with structured output extraction."""

    @pytest.mark.asyncio
    async def test_xml_parser_workflow(self):
        """Test workflow with ParserNode for XML parsing."""
        with GraphNode(name="parser-workflow") as graph:
            generator = CodeNode(
                name="generator",
                code_fn=lambda query: {"response": "<category>tech</category><confidence>0.95</confidence>"},
                inputs={"query": PARENT["query"]}
            )

            parser = ParserNode(
                name="parser",
                format="xml",
                extract_schema=["category: str", "confidence: str"],
                inputs={"text": generator["response"]},
                outputs=PARENT
            )

            START >> generator >> parser >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"query": "What is AI?"})

        assert result["category"] == "tech"
        assert result["confidence"] == "0.95"

    @pytest.mark.asyncio
    async def test_json_parser_workflow(self):
        """Test workflow with ParserNode for JSON parsing."""
        with GraphNode(name="json-parser-workflow") as graph:
            generator = CodeNode(
                name="json_gen",
                code_fn=lambda data: {"json_str": '{"name": "John", "age": "30"}'},
                inputs={"data": PARENT["data"]}
            )

            parser = ParserNode(
                name="json_parser",
                format="json",
                extract_schema=["name: str", "age: str"],
                inputs={"text": generator["json_str"]},
                outputs=PARENT
            )

            START >> generator >> parser >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"data": "test"})

        assert result["name"] == "John"
        assert result["age"] == "30"


# ============================================================================
# Tracer Integration Tests
# ============================================================================

class TestLangfuseTracerIntegration:
    """Test LangfuseTracer integration with workflows."""

    @pytest.mark.asyncio
    async def test_prepare_flush_data_structure(self):
        """Test that prepare_flush_data returns correct structure."""
        from hush.observability import LangfuseTracer

        tracer = LangfuseTracer(resource_key="langfuse:vpbank")

        # Create a simple workflow
        with GraphNode(name="flush-data-test") as graph:
            node = CodeNode(
                name="doubler",
                code_fn=lambda x: {"y": x * 2},
                inputs={"x": PARENT["x"]},
                outputs=PARENT
            )
            START >> node >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"x": 5}, request_id="test-req-001")

        # Get state from result
        state = result["$state"]

        # Prepare flush data
        flush_data = tracer.prepare_flush_data("flush-data-test", state)

        # Verify structure
        assert flush_data["tracer_type"] == "LangfuseTracer"
        assert flush_data["tracer_config"]["resource_key"] == "langfuse:vpbank"
        assert flush_data["workflow_name"] == "flush-data-test"
        assert flush_data["request_id"] == "test-req-001"
        assert "execution_order" in flush_data
        assert "nodes_trace_data" in flush_data
        assert len(flush_data["execution_order"]) > 0

    def test_tracer_flush_with_mock_hub(self, mock_langfuse_client):
        """Test LangfuseTracer.flush() with mocked ResourceHub."""
        from hush.observability import LangfuseTracer

        # Prepare mock flush data
        request_id = str(uuid.uuid4())
        flush_data = {
            "tracer_type": "LangfuseTracer",
            "tracer_config": {"resource_key": "langfuse:vpbank"},
            "workflow_name": "test-workflow",
            "request_id": request_id,
            "user_id": "test-user",
            "session_id": "test-session",
            "execution_order": [
                {
                    "node": "root",
                    "parent": None,
                    "context_id": None,
                    "contain_generation": False,
                },
                {
                    "node": "processor",
                    "parent": "root",
                    "context_id": None,
                    "contain_generation": False,
                },
            ],
            "nodes_trace_data": {
                "root": {
                    "name": "root",
                    "input": {"query": "test"},
                    "output": {"result": "done"},
                    "metadata": {},
                },
                "processor": {
                    "name": "processor",
                    "input": {"x": 5},
                    "output": {"y": 10},
                    "metadata": {},
                },
            },
        }

        # Mock get_hub
        mock_hub = MagicMock()
        mock_hub.langfuse.return_value = mock_langfuse_client

        with patch("hush.core.registry.get_hub", return_value=mock_hub):
            LangfuseTracer.flush(flush_data)

        # Verify trace was created
        mock_langfuse_client.trace.assert_called_once()
        mock_langfuse_client.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_with_tracer(self, mock_langfuse_client):
        """Test complete workflow execution with tracer."""
        from hush.observability import LangfuseTracer

        with GraphNode(name="tracer-workflow") as graph:
            node = CodeNode(
                name="doubler",
                code_fn=lambda value: {"doubled": value * 2},
                inputs={"value": PARENT["value"]},
                outputs=PARENT
            )
            START >> node >> END

        engine = Hush(graph)
        tracer = LangfuseTracer(resource_key="langfuse:vpbank")

        # Mock the hub for flush
        mock_hub = MagicMock()
        mock_hub.langfuse.return_value = mock_langfuse_client

        with patch("hush.core.registry.get_hub", return_value=mock_hub):
            result = await engine.run(
                inputs={"value": 21},
                tracer=tracer
            )

        assert result["doubled"] == 42


# ============================================================================
# Complex Workflow Tests
# ============================================================================

class TestComplexWorkflow:
    """Test complex workflows combining multiple node types."""

    @pytest.mark.asyncio
    async def test_pipeline_with_branch_and_parser(self):
        """Test a pipeline combining CodeNode, BranchNode, and ParserNode."""
        with GraphNode(name="complex-pipeline") as graph:
            classifier = CodeNode(
                name="classifier",
                code_fn=lambda text: {
                    "is_question": text.strip().endswith("?"),
                    "text": text
                },
                inputs={"text": PARENT["text"]}
            )

            router = BranchNode(
                name="router",
                cases={"is_question == True": "answerer"},
                default="acknowledger",
                inputs={"is_question": classifier["is_question"]}
            )

            answerer = CodeNode(
                name="answerer",
                code_fn=lambda text: {"response": f"<answer>This is the answer to: {text}</answer>"},
                inputs={"text": classifier["text"]}
            )

            acknowledger = CodeNode(
                name="acknowledger",
                code_fn=lambda text: {"response": f"<statement>Acknowledged: {text}</statement>"},
                inputs={"text": classifier["text"]}
            )

            # Both paths parse the response
            parser_q = ParserNode(
                name="parser_q",
                format="xml",
                extract_schema=["answer: str"],
                inputs={"text": answerer["response"]},
                outputs={"answer": PARENT["result"]}
            )

            parser_s = ParserNode(
                name="parser_s",
                format="xml",
                extract_schema=["statement: str"],
                inputs={"text": acknowledger["response"]},
                outputs={"statement": PARENT["result"]}
            )

            START >> classifier >> router
            router > [answerer, acknowledger]
            answerer >> parser_q > END
            acknowledger >> parser_s > END

        engine = Hush(graph)

        # Test question path
        result_q = await engine.run(inputs={"text": "What is AI?"})
        assert "answer to" in result_q.get("result", "").lower() or "answer" in str(result_q)

        # Test statement path
        result_s = await engine.run(inputs={"text": "AI is powerful"})
        assert "acknowledged" in result_s.get("result", "").lower() or "statement" in str(result_s)

    @pytest.mark.asyncio
    async def test_full_trace_collection(self):
        """Test that all nodes in a complex workflow collect trace metadata."""
        with GraphNode(name="full-trace-workflow") as graph:
            a = CodeNode(name="step_a", code_fn=lambda x: {"a_out": x + 1}, inputs={"x": PARENT["x"]})
            b = CodeNode(name="step_b", code_fn=lambda y: {"b_out": y * 2}, inputs={"y": a["a_out"]})
            c = CodeNode(name="step_c", code_fn=lambda z: {"c_out": z ** 2}, inputs={"z": b["b_out"]}, outputs=PARENT)

            START >> a >> b >> c >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"x": 5}, request_id="full-trace-req")

        # Access state via $state
        state = result["$state"]

        # Check all nodes have trace metadata
        trace_metadata = state.trace_metadata

        node_names = ["step_a", "step_b", "step_c"]
        for name in node_names:
            found = any(name in key for key in trace_metadata.keys())
            assert found, f"Node {name} should have trace metadata"

        # Verify execution order
        execution_order = state.execution_order
        executed_nodes = [e["node"] for e in execution_order]

        # All nodes should be executed
        assert any("step_a" in n for n in executed_nodes)
        assert any("step_b" in n for n in executed_nodes)
        assert any("step_c" in n for n in executed_nodes)

    @pytest.mark.asyncio
    async def test_trace_flush_data_contains_io_values(self):
        """Test that flush_data includes actual input/output values from state."""
        from hush.observability import LangfuseTracer

        with GraphNode(name="io-values-workflow") as graph:
            node = CodeNode(
                name="multiplier",
                code_fn=lambda a, b: {"product": a * b},
                inputs={"a": PARENT["a"], "b": PARENT["b"]},
                outputs=PARENT
            )
            START >> node >> END

        engine = Hush(graph)
        tracer = LangfuseTracer(resource_key="langfuse:vpbank")

        result = await engine.run(inputs={"a": 7, "b": 6}, request_id="io-test-req")
        state = result["$state"]

        flush_data = tracer.prepare_flush_data("io-values-workflow", state)

        # Find the multiplier node data
        multiplier_data = None
        for key, data in flush_data["nodes_trace_data"].items():
            if "multiplier" in key:
                multiplier_data = data
                break

        assert multiplier_data is not None, "Should have multiplier node data"

        # Input values should be captured
        assert multiplier_data["input"].get("a") == 7
        assert multiplier_data["input"].get("b") == 6

        # Output values should be captured
        assert multiplier_data["output"].get("product") == 42


# ============================================================================
# Code Node Decorator Tests
# ============================================================================

class TestCodeNodeDecorator:
    """Test @code_node decorator with workflow."""

    @pytest.mark.asyncio
    async def test_decorated_code_node(self):
        """Test using @code_node decorator in workflow."""
        @code_node
        def square_fn(n: int):
            """Square a number."""
            return {"squared": n ** 2}

        with GraphNode(name="decorator-workflow") as graph:
            node = square_fn(inputs={"n": PARENT["n"]}, outputs=PARENT)
            START >> node >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"n": 9})

        assert result["squared"] == 81


# ============================================================================
# Callable Syntax Tests
# ============================================================================

class TestCallableSyntax:
    """Test Hush callable syntax."""

    @pytest.mark.asyncio
    async def test_callable_syntax(self):
        """Test using engine(inputs) instead of engine.run(inputs)."""
        with GraphNode(name="callable-test") as graph:
            node = CodeNode(
                name="adder",
                code_fn=lambda x, y: {"sum": x + y},
                inputs={"x": PARENT["x"], "y": PARENT["y"]},
                outputs=PARENT
            )
            START >> node >> END

        engine = Hush(graph)

        # Use callable syntax
        result = await engine({"x": 10, "y": 20})

        assert result["sum"] == 30
        assert "$state" in result


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == "__main__":
    import asyncio

    print("Running workflow integration tests...")
    print("=" * 60)

    # CodeNode tests
    print("\n1. Testing simple CodeNode workflow...")
    asyncio.run(TestCodeNodeWorkflow().test_simple_code_node_workflow())
    print("   PASSED")

    print("\n2. Testing CodeNode trace metadata...")
    asyncio.run(TestCodeNodeWorkflow().test_code_node_trace_metadata())
    print("   PASSED")

    print("\n3. Testing chained CodeNodes...")
    asyncio.run(TestCodeNodeWorkflow().test_chained_code_nodes())
    print("   PASSED")

    # BranchNode tests
    print("\n4. Testing BranchNode workflow...")
    asyncio.run(TestBranchNodeWorkflow().test_simple_branch_workflow())
    print("   PASSED")

    # ParserNode tests
    print("\n5. Testing XML ParserNode workflow...")
    asyncio.run(TestParserNodeWorkflow().test_xml_parser_workflow())
    print("   PASSED")

    print("\n6. Testing JSON ParserNode workflow...")
    asyncio.run(TestParserNodeWorkflow().test_json_parser_workflow())
    print("   PASSED")

    # Tracer tests
    print("\n7. Testing prepare_flush_data structure...")
    asyncio.run(TestLangfuseTracerIntegration().test_prepare_flush_data_structure())
    print("   PASSED")

    # Complex workflow tests
    print("\n8. Testing full trace collection...")
    asyncio.run(TestComplexWorkflow().test_full_trace_collection())
    print("   PASSED")

    print("\n9. Testing flush_data contains IO values...")
    asyncio.run(TestComplexWorkflow().test_trace_flush_data_contains_io_values())
    print("   PASSED")

    # Decorator test
    print("\n10. Testing @code_node decorator...")
    asyncio.run(TestCodeNodeDecorator().test_decorated_code_node())
    print("   PASSED")

    # Callable syntax test
    print("\n11. Testing callable syntax...")
    asyncio.run(TestCallableSyntax().test_callable_syntax())
    print("   PASSED")

    print("\n" + "=" * 60)
    print("All workflow integration tests completed!")
