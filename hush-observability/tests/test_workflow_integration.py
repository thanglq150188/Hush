"""Workflow integration tests for hush-observability.

This module tests the complete observability flow using Hush engine
with multiple node types: CodeNode, BranchNode, ParserNode.

All tests flush traces to Langfuse (langfuse:vpbank) for real observability.

The tests verify that:
1. Trace metadata is correctly collected during workflow execution
2. Execution order is properly recorded
3. Tracers successfully flush to Langfuse backend
"""

import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv(Path(__file__).parent.parent / ".env")

import pytest

from hush.core import Hush, GraphNode, START, END, PARENT
from hush.core.nodes import CodeNode, BranchNode, ParserNode, code_node
from hush.core.tracers import BaseTracer
from hush.observability import LangfuseTracer
from hush.providers import LLMNode, LLMChainNode


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def tracer():
    """Create LangfuseTracer for langfuse:vpbank."""
    return LangfuseTracer(resource_key="langfuse:vpbank")


@pytest.fixture(scope="session", autouse=True)
def wait_for_flush():
    """Wait for background flush to complete after all tests."""
    yield
    # Shutdown the worker process gracefully
    BaseTracer.shutdown_executor()


# ============================================================================
# CodeNode Tests
# ============================================================================

class TestCodeNodeWorkflow:
    """Test CodeNode execution with trace metadata collection."""

    @pytest.mark.asyncio
    async def test_simple_code_node_workflow(self, tracer):
        """Test a simple workflow with a single CodeNode."""
        with GraphNode(name="simple-code-workflow") as graph:
            code = CodeNode(
                name="processor",
                code_fn=lambda text: {"processed": text.upper(), "length": len(text)},
                inputs={"text": PARENT["text"]},
                outputs={"*": PARENT}
            )
            START >> code >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"text": "hello world"}, tracer=tracer)

        # Verify outputs
        assert result["processed"] == "HELLO WORLD"
        assert result["length"] == 11
        # Verify $state is included
        assert "$state" in result

    @pytest.mark.asyncio
    async def test_code_node_trace_metadata(self, tracer):
        """Test that CodeNode correctly records trace metadata."""
        with GraphNode(name="code-trace-test") as graph:
            code = CodeNode(
                name="adder",
                code_fn=lambda a, b: {"sum": a + b},
                inputs={"a": PARENT["a"], "b": PARENT["b"]},
                outputs={"*": PARENT}
            )
            START >> code >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"a": 5, "b": 3},
            request_id="test-request-123",
            tracer=tracer
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
    async def test_chained_code_nodes(self, tracer):
        """Test workflow with multiple chained CodeNodes."""
        with GraphNode(name="chained-workflow") as graph:
            node1 = CodeNode(name="double", code_fn=lambda x: {"y": x * 2}, inputs={"x": PARENT["x"]})
            node2 = CodeNode(name="add_ten", code_fn=lambda y: {"z": y + 10}, inputs={"y": node1["y"]})
            node3 = CodeNode(name="square", code_fn=lambda z: {"result": z ** 2}, inputs={"z": node2["z"]}, outputs={"*": PARENT})

            START >> node1 >> node2 >> node3 >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"x": 5}, tracer=tracer)

        # x=5 -> y=10 -> z=20 -> result=400
        assert result["result"] == 400


# ============================================================================
# BranchNode Tests
# ============================================================================

class TestBranchNodeWorkflow:
    """Test BranchNode execution with conditional routing."""

    @pytest.mark.asyncio
    async def test_simple_branch_workflow(self, tracer):
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
                outputs={"*": PARENT}
            )

            low = CodeNode(
                name="low_path",
                code_fn=lambda score: {"grade": "F", "message": "Try harder!"},
                inputs={"score": PARENT["score"]},
                outputs={"*": PARENT}
            )

            START >> branch
            branch > [high, low] > END

        engine = Hush(graph)

        # Test high score path
        result_high = await engine.run(inputs={"score": 85}, tracer=tracer)
        assert result_high["grade"] == "A"

        # Test low score path
        result_low = await engine.run(inputs={"score": 50}, tracer=tracer)
        assert result_low["grade"] == "F"

    @pytest.mark.asyncio
    async def test_branch_trace_metadata(self, tracer):
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
                outputs={"*": PARENT}
            )
            neg = CodeNode(
                name="negative",
                code_fn=lambda value: {"output": value},
                inputs={"value": PARENT["value"]},
                outputs={"*": PARENT}
            )

            START >> branch
            branch > [pos, neg] > END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"value": 10},
            request_id="branch-test-123",
            tracer=tracer
        )

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
    async def test_xml_parser_workflow(self, tracer):
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
                outputs={"*": PARENT}
            )

            START >> generator >> parser >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"query": "What is AI?"}, tracer=tracer)

        assert result["category"] == "tech"
        assert result["confidence"] == "0.95"

    @pytest.mark.asyncio
    async def test_json_parser_workflow(self, tracer):
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
                outputs={"*": PARENT}
            )

            START >> generator >> parser >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"data": "test"}, tracer=tracer)

        assert result["name"] == "John"
        assert result["age"] == "30"


# ============================================================================
# Tracer Integration Tests
# ============================================================================

class TestLangfuseTracerIntegration:
    """Test LangfuseTracer integration with workflows."""

    @pytest.mark.asyncio
    async def test_prepare_flush_data_structure(self, tracer):
        """Test that prepare_flush_data returns correct structure."""
        # Create a simple workflow
        with GraphNode(name="flush-data-test") as graph:
            node = CodeNode(
                name="doubler",
                code_fn=lambda x: {"y": x * 2},
                inputs={"x": PARENT["x"]},
                outputs={"*": PARENT}
            )
            START >> node >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"x": 5}, request_id="test-req-001", tracer=tracer)

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

    @pytest.mark.asyncio
    async def test_workflow_with_tracer(self, tracer):
        """Test complete workflow execution with tracer."""
        with GraphNode(name="tracer-workflow") as graph:
            node = CodeNode(
                name="doubler",
                code_fn=lambda value: {"doubled": value * 2},
                inputs={"value": PARENT["value"]},
                outputs={"*": PARENT}
            )
            START >> node >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"value": 21}, tracer=tracer)

        assert result["doubled"] == 42


# ============================================================================
# Complex Workflow Tests
# ============================================================================

class TestComplexWorkflow:
    """Test complex workflows combining multiple node types."""

    @pytest.mark.asyncio
    async def test_pipeline_with_branch_and_parser(self, tracer):
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
        result_q = await engine.run(inputs={"text": "What is AI?"}, tracer=tracer)
        assert "answer to" in result_q.get("result", "").lower() or "answer" in str(result_q)

        # Test statement path
        result_s = await engine.run(inputs={"text": "AI is powerful"}, tracer=tracer)
        assert "acknowledged" in result_s.get("result", "").lower() or "statement" in str(result_s)

    @pytest.mark.asyncio
    async def test_full_trace_collection(self, tracer):
        """Test that all nodes in a complex workflow collect trace metadata."""
        with GraphNode(name="full-trace-workflow") as graph:
            a = CodeNode(name="step_a", code_fn=lambda x: {"a_out": x + 1}, inputs={"x": PARENT["x"]})
            b = CodeNode(name="step_b", code_fn=lambda y: {"b_out": y * 2}, inputs={"y": a["a_out"]})
            c = CodeNode(name="step_c", code_fn=lambda z: {"c_out": z ** 2}, inputs={"z": b["b_out"]}, outputs={"*": PARENT})

            START >> a >> b >> c >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"x": 5}, request_id="full-trace-req", tracer=tracer)

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
    async def test_trace_flush_data_contains_io_values(self, tracer):
        """Test that flush_data includes actual input/output values from state."""
        with GraphNode(name="io-values-workflow") as graph:
            node = CodeNode(
                name="multiplier",
                code_fn=lambda a, b: {"product": a * b},
                inputs={"a": PARENT["a"], "b": PARENT["b"]},
                outputs={"*": PARENT}
            )
            START >> node >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"a": 7, "b": 6}, request_id="io-test-req", tracer=tracer)
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
# LLMNode Tests
# ============================================================================

class TestLLMNodeWorkflow:
    """Test LLMNode execution with real LLM API calls."""

    @pytest.mark.asyncio
    async def test_simple_llm_node(self, tracer):
        """Test a simple workflow with LLMNode using gpt-4o."""
        with GraphNode(name="simple-llm-workflow") as graph:
            llm = LLMNode(
                name="chat",
                resource_key="qwen3-30b-a3b",
                inputs={"messages": PARENT["messages"]},
                outputs={"*": PARENT}
            )
            START >> llm >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"messages": [{"role": "user", "content": "Say 'Hello' and nothing else."}]},
            tracer=tracer
        )

        # Verify outputs
        assert "content" in result
        assert result["role"] == "assistant"
        assert "model_used" in result
        assert "tokens_used" in result

    @pytest.mark.asyncio
    async def test_llm_node_trace_metadata(self, tracer):
        """Test that LLMNode correctly records trace metadata with model/usage."""
        with GraphNode(name="llm-trace-test") as graph:
            llm = LLMNode(
                name="generator",
                resource_key="qwen3-30b-a3b",
                inputs={"messages": PARENT["messages"]},
                outputs={"*": PARENT}
            )
            START >> llm >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"messages": [{"role": "user", "content": "Say 'test' only."}]},
            request_id="llm-trace-123",
            tracer=tracer
        )

        # Access state
        state = result["$state"]
        trace_metadata = state.trace_metadata

        # Find the generator node metadata
        generator_key = None
        for key in trace_metadata:
            if "generator" in key:
                generator_key = key
                break

        assert generator_key is not None
        metadata = trace_metadata[generator_key]
        assert metadata["name"] == "generator"
        assert metadata["contain_generation"] is True
        # Check model and usage are recorded
        assert "model" in metadata
        assert "usage" in metadata

    @pytest.mark.asyncio
    async def test_llm_node_with_temperature(self, tracer):
        """Test LLMNode with custom temperature setting."""
        with GraphNode(name="llm-temp-workflow") as graph:
            llm = LLMNode(
                name="creative",
                resource_key="qwen3-30b-a3b",
                inputs={
                    "messages": PARENT["messages"],
                    "temperature": PARENT["temperature"]
                },
                outputs={"*": PARENT}
            )
            START >> llm >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={
                "messages": [{"role": "user", "content": "Say 'creative' only."}],
                "temperature": 0.7
            },
            tracer=tracer
        )

        assert "content" in result
        assert result["role"] == "assistant"


# ============================================================================
# LLMChainNode Tests
# ============================================================================

class TestLLMChainNodeWorkflow:
    """Test LLMChainNode execution - combines prompt formatting, LLM, and parsing."""

    @pytest.mark.asyncio
    async def test_simple_llm_chain(self, tracer):
        """Test LLMChainNode with simple prompt template."""
        with GraphNode(name="llm-chain-simple") as graph:
            chain = LLMChainNode(
                name="summarizer",
                resource_key="qwen3-30b-a3b",
                system_prompt="You are a helpful assistant.",
                user_prompt="Repeat exactly: {text}",
                inputs={"text": PARENT["text"]},
                outputs={"*": PARENT}
            )
            START >> chain >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"text": "Hello World"},
            tracer=tracer
        )

        # Check outputs
        assert "content" in result
        assert result["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_llm_chain_with_parser(self, tracer):
        """Test LLMChainNode with structured output parsing."""
        with GraphNode(name="llm-chain-parser") as graph:
            chain = LLMChainNode(
                name="classifier",
                resource_key="qwen3-30b-a3b",
                system_prompt="You are a sentiment classifier. Always output in XML format.",
                user_prompt="Classify the sentiment of: {text}\n\nOutput exactly: <sentiment>positive or negative</sentiment><confidence>a number between 0 and 1</confidence>",
                extract_schema=["sentiment: str", "confidence: str"],
                parser="xml",
                inputs={"text": PARENT["text"]},
                outputs={"*": PARENT}
            )
            START >> chain >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"text": "I love this product!"},
            tracer=tracer
        )

        # Check parsed outputs
        assert "sentiment" in result
        assert "confidence" in result

    @pytest.mark.asyncio
    async def test_llm_chain_trace_hierarchy(self, tracer):
        """Test that LLMChainNode creates proper trace hierarchy (prompt -> llm -> parser)."""
        with GraphNode(name="llm-chain-hierarchy") as graph:
            chain = LLMChainNode(
                name="analyzer",
                resource_key="qwen3-30b-a3b",
                user_prompt="Analyze: {query}\n\nOutput: <result>your analysis</result>",
                extract_schema=["result: str"],
                parser="xml",
                inputs={"query": PARENT["query"]},
                outputs={"*": PARENT}
            )
            START >> chain >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"query": "What is 2+2?"},
            request_id="chain-hierarchy-123",
            tracer=tracer
        )

        # Access state and check execution order includes child nodes
        state = result["$state"]
        execution_order = state.execution_order
        executed_nodes = [e["node"] for e in execution_order]

        # Should have the chain and its child nodes (prompt, llm, parser)
        assert any("analyzer" in n for n in executed_nodes)

    @pytest.mark.asyncio
    async def test_llm_chain_flush_data(self, tracer):
        """Test that LLMChainNode flush_data contains model and usage info."""
        with GraphNode(name="llm-chain-flush") as graph:
            chain = LLMChainNode(
                name="responder",
                resource_key="qwen3-30b-a3b",
                user_prompt="Say '{word}' exactly.",
                inputs={"word": PARENT["word"]},
                outputs={"*": PARENT}
            )
            START >> chain >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"word": "test"},
            request_id="chain-flush-123",
            tracer=tracer
        )

        state = result["$state"]
        flush_data = tracer.prepare_flush_data("llm-chain-flush", state)

        # Check that LLM node has model and usage
        llm_node_found = False
        for key, data in flush_data["nodes_trace_data"].items():
            if "llm" in key.lower():
                llm_node_found = True
                # Model and usage should be in trace data
                assert "model" in data or data.get("metadata", {}).get("model")
                break

        assert llm_node_found, "Should have LLM node in trace data"


# ============================================================================
# Code Node Decorator Tests
# ============================================================================

class TestCodeNodeDecorator:
    """Test @code_node decorator with workflow."""

    @pytest.mark.asyncio
    async def test_decorated_code_node(self, tracer):
        """Test using @code_node decorator in workflow."""
        @code_node
        def square_fn(n: int):
            """Square a number."""
            return {"squared": n ** 2}

        with GraphNode(name="decorator-workflow") as graph:
            node = square_fn(inputs={"n": PARENT["n"]}, outputs={"*": PARENT})
            START >> node >> END

        engine = Hush(graph)
        result = await engine.run(inputs={"n": 9}, tracer=tracer)

        assert result["squared"] == 81


# ============================================================================
# Callable Syntax Tests
# ============================================================================

class TestCallableSyntax:
    """Test Hush callable syntax."""

    @pytest.mark.asyncio
    async def test_callable_syntax(self, tracer):
        """Test using engine(inputs) instead of engine.run(inputs)."""
        with GraphNode(name="callable-test") as graph:
            node = CodeNode(
                name="adder",
                code_fn=lambda x, y: {"sum": x + y},
                inputs={"x": PARENT["x"], "y": PARENT["y"]},
                outputs={"*": PARENT}
            )
            START >> node >> END

        engine = Hush(graph)

        # Use callable syntax
        result = await engine({"x": 10, "y": 20}, tracer=tracer)

        assert result["sum"] == 30
        assert "$state" in result


# ============================================================================
# Complex Loop Workflow Tests (ForLoop, MapNode, WhileLoop)
# ============================================================================

class TestComplexLoopWorkflows:
    """Test complex workflows with ForLoop, MapNode, WhileLoop - with and without tags."""

    @pytest.mark.asyncio
    async def test_map_node_workflow(self, tracer):
        """Test MapNode workflow - process each item in a list."""
        from hush.core.nodes.iteration.map_node import MapNode
        from hush.core.nodes.iteration.base import Each

        @code_node
        def process_item(item: int, multiplier: int):
            """Process a single item."""
            return {"result": item * multiplier}

        @code_node
        def aggregate(results: list):
            """Aggregate all results."""
            total = sum(results) if results else 0
            return {"total": total, "count": len(results)}

        with GraphNode(name="map-node-workflow") as graph:
            with MapNode(
                name="process_items",
                inputs={
                    "item": Each(PARENT["items"]),
                    "multiplier": PARENT["multiplier"],
                }
            ) as map_node:
                proc = process_item(
                    name="processor",
                    inputs={"item": PARENT["item"], "multiplier": PARENT["multiplier"]},
                    outputs={"*": PARENT},
                )
                START >> proc >> END

            agg = aggregate(
                name="aggregator",
                inputs={"results": map_node["result"]},
                outputs={"*": PARENT},
            )

            START >> map_node >> agg >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"items": [1, 2, 3, 4, 5], "multiplier": 2},
            request_id="map-node-001",
            user_id="test-user",
            session_id="map-session",
            tracer=tracer,
        )

        # items [1,2,3,4,5] * 2 = [2,4,6,8,10] -> total = 30
        assert result["total"] == 30
        assert result["count"] == 5

    @pytest.mark.asyncio
    async def test_map_node_with_static_tags(self):
        """Test MapNode workflow with static tracer tags."""
        from hush.core.nodes.iteration.map_node import MapNode
        from hush.core.nodes.iteration.base import Each

        # Create tracer with static tags
        tracer = LangfuseTracer(resource_key="langfuse:vpbank", tags=["production", "batch-processing"])

        @code_node
        def square(n: int):
            return {"squared": n ** 2}

        with GraphNode(name="map-with-static-tags") as graph:
            with MapNode(
                name="square_items",
                inputs={"n": Each(PARENT["numbers"])}
            ) as map_node:
                sq = square(name="square", inputs={"n": PARENT["n"]}, outputs={"*": PARENT})
                START >> sq >> END

            START >> map_node >> END
            map_node["squared"] >> PARENT["results"]

        engine = Hush(graph)
        result = await engine.run(
            inputs={"numbers": [2, 3, 4]},
            request_id="map-static-tags-001",
            user_id="alice",
            session_id="static-tags-session",
            tracer=tracer,
        )

        assert result["results"] == [4, 9, 16]

    @pytest.mark.asyncio
    async def test_map_node_with_dynamic_tags(self):
        """Test MapNode workflow with dynamic tags from node output."""
        from hush.core.nodes.iteration.map_node import MapNode
        from hush.core.nodes.iteration.base import Each

        tracer = LangfuseTracer(resource_key="langfuse:vpbank", tags=["experiment"])

        @code_node
        def analyze_item(value: int):
            """Analyze item and add dynamic tags based on value."""
            result = value * 10
            tags = ["processed"]
            if result > 50:
                tags.append("high-value")
            return {"score": result, "$tags": tags}

        @code_node
        def summarize(scores: list):
            """Summarize scores with dynamic tag."""
            total = sum(scores)
            tags = ["summarized"]
            if total > 100:
                tags.append("large-total")
            return {"total": total, "$tags": tags}

        with GraphNode(name="map-with-dynamic-tags") as graph:
            with MapNode(
                name="analyze_items",
                inputs={"value": Each(PARENT["values"])}
            ) as map_node:
                analyze = analyze_item(
                    name="analyzer",
                    inputs={"value": PARENT["value"]},
                    outputs={"*": PARENT},
                )
                START >> analyze >> END

            summary = summarize(
                name="summary",
                inputs={"scores": map_node["score"]},
                outputs={"*": PARENT},
            )

            START >> map_node >> summary >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"values": [5, 8, 12]},  # scores: 50, 80, 120 -> total: 250
            request_id="map-dynamic-tags-001",
            user_id="bob",
            session_id="dynamic-tags-session",
            tracer=tracer,
        )

        assert result["total"] == 250

    @pytest.mark.asyncio
    async def test_for_loop_workflow(self, tracer):
        """Test ForLoopNode workflow - iterate with Each."""
        from hush.core.nodes.iteration.for_loop_node import ForLoopNode
        from hush.core.nodes.iteration.base import Each

        @code_node
        def process_x(x: int):
            return {"doubled": x * 2}

        with GraphNode(name="for-loop-workflow") as graph:
            with ForLoopNode(
                name="iterate_values",
                inputs={"x": Each([10, 20, 30])}
            ) as for_loop:
                proc = process_x(
                    name="doubler",
                    inputs={"x": PARENT["x"]},
                    outputs={"*": PARENT},
                )
                START >> proc >> END

            for_loop["doubled"] >> PARENT["results"]
            START >> for_loop >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={},
            request_id="for-loop-001",
            user_id="test-user",
            session_id="for-loop-session",
            tracer=tracer,
        )

        assert result["results"] == [20, 40, 60]

    @pytest.mark.asyncio
    async def test_nested_for_loops(self, tracer):
        """Test nested ForLoopNode workflow - outer and inner loops."""
        from hush.core.nodes.iteration.for_loop_node import ForLoopNode
        from hush.core.nodes.iteration.base import Each

        @code_node
        def validate(x: int):
            return {"validated_x": x, "$tags": ["validated"]}

        @code_node
        def multiply(x: int, y: int):
            product = x * y
            tags = ["multiplied"]
            if product > 50:
                tags.append("large-product")
            return {"product": product, "$tags": tags}

        @code_node
        def summarize(products: list):
            return {"total": sum(products) if products else 0}

        with GraphNode(name="nested-for-loops") as graph:
            with ForLoopNode(
                name="outer_loop",
                inputs={"x": Each([2, 3])}
            ) as outer:
                validate_node = validate(
                    name="validate",
                    inputs={"x": PARENT["x"]},
                )

                with ForLoopNode(
                    name="inner_loop",
                    inputs={
                        "y": Each([10, 20]),
                        "x": validate_node["validated_x"],
                    }
                ) as inner:
                    mult_node = multiply(
                        name="multiply",
                        inputs={"x": PARENT["x"], "y": PARENT["y"]},
                        outputs={"*": PARENT},
                    )
                    START >> mult_node >> END

                summarize_node = summarize(
                    name="summarize",
                    inputs={"products": inner["product"]},
                    outputs={"*": PARENT},
                )

                START >> validate_node >> inner >> summarize_node >> END

            outer["total"] >> PARENT["results"]
            START >> outer >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={},
            request_id="nested-for-loops-001",
            user_id="charlie",
            session_id="nested-loops-session",
            tracer=tracer,
        )

        # x=2: products=[20,40] total=60
        # x=3: products=[30,60] total=90
        assert result["results"] == [60, 90]

    @pytest.mark.asyncio
    async def test_nested_for_loops_with_tags(self):
        """Test nested ForLoopNode with both static and dynamic tags."""
        from hush.core.nodes.iteration.for_loop_node import ForLoopNode
        from hush.core.nodes.iteration.base import Each

        # Create tracer with static tags
        tracer = LangfuseTracer(
            resource_key="langfuse:vpbank",
            tags=["experiment", "batch-job", "ml-team"]
        )

        @code_node
        def validate(x: int):
            return {"validated_x": x, "$tags": ["validated"]}

        @code_node
        def multiply(x: int, y: int):
            product = x * y
            tags = ["multiplied"]
            if product > 50:
                tags.append("large-product")
            return {"product": product, "$tags": tags}

        @code_node
        def summarize(products: list):
            total = sum(products) if products else 0
            tags = ["summarized"]
            if total > 100:
                tags.append("high-total")
            return {"total": total, "$tags": tags}

        with GraphNode(name="nested-loops-with-tags") as graph:
            with ForLoopNode(
                name="outer_loop",
                inputs={"x": Each([3, 4, 5])}
            ) as outer:
                validate_node = validate(
                    name="validate",
                    inputs={"x": PARENT["x"]},
                )

                with ForLoopNode(
                    name="inner_loop",
                    inputs={
                        "y": Each([10, 20, 30]),
                        "x": validate_node["validated_x"],
                    }
                ) as inner:
                    mult_node = multiply(
                        name="multiply",
                        inputs={"x": PARENT["x"], "y": PARENT["y"]},
                        outputs={"*": PARENT},
                    )
                    START >> mult_node >> END

                summarize_node = summarize(
                    name="summarize",
                    inputs={"products": inner["product"]},
                    outputs={"*": PARENT},
                )

                START >> validate_node >> inner >> summarize_node >> END

            outer["total"] >> PARENT["results"]
            START >> outer >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={},
            request_id="nested-loops-tags-001",
            user_id="alice",
            session_id="nested-tags-session",
            tracer=tracer,
        )

        # x=3: [30,60,90] -> 180
        # x=4: [40,80,120] -> 240
        # x=5: [50,100,150] -> 300
        assert result["results"] == [180, 240, 300]

    @pytest.mark.asyncio
    async def test_while_loop_workflow(self, tracer):
        """Test WhileLoopNode workflow - iterate until condition met."""
        from hush.core.nodes.iteration.while_loop_node import WhileLoopNode

        @code_node
        def halve(value: int):
            new_val = value // 2
            tags = []
            if new_val < 10:
                tags.append("small-value")
            return {"new_value": new_val, "$tags": tags} if tags else {"new_value": new_val}

        with GraphNode(name="while-loop-workflow") as graph:
            with WhileLoopNode(
                name="halve_loop",
                inputs={"value": PARENT["start_value"]},
                stop_condition="value < 5",
                max_iterations=10,
            ) as while_loop:
                halve_node = halve(
                    name="halve",
                    inputs={"value": PARENT["value"]},
                )
                halve_node["new_value"] >> PARENT["value"]
                START >> halve_node >> END

            while_loop["value"] >> PARENT["final_value"]
            START >> while_loop >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"start_value": 100},
            request_id="while-loop-001",
            user_id="test-user",
            session_id="while-session",
            tracer=tracer,
        )

        # 100 -> 50 -> 25 -> 12 -> 6 -> 3 (stops at < 5)
        assert result["final_value"] == 3

    @pytest.mark.asyncio
    async def test_while_loop_with_tags(self):
        """Test WhileLoopNode with static and dynamic tags."""
        from hush.core.nodes.iteration.while_loop_node import WhileLoopNode

        tracer = LangfuseTracer(
            resource_key="langfuse:vpbank",
            tags=["iteration-test", "monitoring"]
        )

        @code_node
        def halve_with_tags(value: int):
            new_val = value // 2
            tags = ["halved"]
            if new_val < 10:
                tags.append("small-value")
            if new_val < 5:
                tags.append("very-small")
            return {"new_value": new_val, "$tags": tags}

        with GraphNode(name="while-loop-with-tags") as graph:
            with WhileLoopNode(
                name="halve_loop",
                inputs={"value": PARENT["start_value"]},
                stop_condition="value < 3",
                max_iterations=15,
            ) as while_loop:
                halve_node = halve_with_tags(
                    name="halve",
                    inputs={"value": PARENT["value"]},
                )
                halve_node["new_value"] >> PARENT["value"]
                START >> halve_node >> END

            while_loop["value"] >> PARENT["final_value"]
            START >> while_loop >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"start_value": 200},
            request_id="while-loop-tags-001",
            user_id="bob",
            session_id="while-tags-session",
            tracer=tracer,
        )

        # 200 -> 100 -> 50 -> 25 -> 12 -> 6 -> 3 -> 1 (stops at < 3)
        assert result["final_value"] == 1

    @pytest.mark.asyncio
    async def test_complex_pipeline_with_all_loop_types(self):
        """Test complex pipeline combining MapNode, ForLoopNode, and various node types."""
        from hush.core.nodes.iteration.for_loop_node import ForLoopNode
        from hush.core.nodes.iteration.map_node import MapNode
        from hush.core.nodes.iteration.base import Each

        tracer = LangfuseTracer(
            resource_key="langfuse:vpbank",
            tags=["complex-pipeline", "production", "ml-team"]
        )

        @code_node
        def preprocess(text: str):
            """Preprocess input text."""
            cleaned = text.strip().lower()
            tags = ["preprocessed"]
            if len(cleaned) > 20:
                tags.append("long-text")
            return {"cleaned": cleaned, "$tags": tags}

        @code_node
        def tokenize(text: str):
            """Split text into tokens."""
            tokens = text.split()
            tags = ["tokenized"]
            if len(tokens) > 5:
                tags.append("many-tokens")
            return {"tokens": tokens, "count": len(tokens), "$tags": tags}

        @code_node
        def analyze_token(token: str, weight: int):
            """Analyze a single token."""
            score = len(token) * weight
            return {"score": score}

        @code_node
        def aggregate_scores(scores: list):
            """Aggregate all token scores."""
            total = sum(scores) if scores else 0
            avg = total / len(scores) if scores else 0
            tags = ["aggregated"]
            if avg > 15:
                tags.append("high-avg-score")
            return {"total": total, "average": avg, "$tags": tags}

        @code_node
        def classify(average: float):
            """Classify based on average score."""
            if average > 20:
                category, confidence = "high", 0.9
            elif average > 10:
                category, confidence = "medium", 0.7
            else:
                category, confidence = "low", 0.5
            return {
                "category": category,
                "confidence": confidence,
                "$tags": [f"category:{category}"]
            }

        with GraphNode(name="complex-text-analysis-pipeline") as graph:
            preprocess_node = preprocess(
                name="preprocess",
                inputs={"text": PARENT["input_text"]},
            )

            tokenize_node = tokenize(
                name="tokenize",
                inputs={"text": preprocess_node["cleaned"]},
            )

            with MapNode(
                name="analyze_tokens",
                inputs={
                    "token": Each(tokenize_node["tokens"]),
                    "weight": PARENT["weight"],
                }
            ) as map_node:
                analyze = analyze_token(
                    name="analyze",
                    inputs={"token": PARENT["token"], "weight": PARENT["weight"]},
                    outputs={"*": PARENT},
                )
                START >> analyze >> END

            aggregate_node = aggregate_scores(
                name="aggregate",
                inputs={"scores": map_node["score"]},
            )

            classify_node = classify(
                name="classify",
                inputs={"average": aggregate_node["average"]},
                outputs={"*": PARENT},
            )

            START >> preprocess_node >> tokenize_node >> map_node >> aggregate_node >> classify_node >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={
                "input_text": "Machine learning transforms modern industries significantly",
                "weight": 3
            },
            request_id="complex-pipeline-001",
            user_id="data-scientist",
            session_id="analysis-session",
            tracer=tracer,
        )

        assert "category" in result
        assert "confidence" in result
        assert result["category"] in ["high", "medium", "low"]


class TestLangfuseTracerWithTags:
    """Test LangfuseTracer specifically with static and dynamic tags."""

    @pytest.mark.asyncio
    async def test_tracer_with_static_tags_only(self):
        """Test LangfuseTracer with only static tags."""
        tracer = LangfuseTracer(
            resource_key="langfuse:vpbank",
            tags=["prod", "ml-team", "v2.0"]
        )

        assert tracer.tags == ["prod", "ml-team", "v2.0"]

        with GraphNode(name="static-tags-only") as graph:
            node = CodeNode(
                name="processor",
                code_fn=lambda x: {"result": x * 2},
                inputs={"x": PARENT["x"]},
                outputs={"*": PARENT}
            )
            START >> node >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"x": 21},
            request_id="static-only-001",
            user_id="user-static",
            session_id="static-session",
            tracer=tracer,
        )

        assert result["result"] == 42

    @pytest.mark.asyncio
    async def test_tracer_with_dynamic_tags_only(self):
        """Test LangfuseTracer with only dynamic tags from node output."""
        tracer = LangfuseTracer(resource_key="langfuse:vpbank")  # No static tags

        @code_node
        def process_with_tags(x: int):
            result = x * 3
            tags = ["computed", "runtime"]
            if result > 50:
                tags.append("large-result")
            return {"result": result, "$tags": tags}

        with GraphNode(name="dynamic-tags-only") as graph:
            node = process_with_tags(
                name="processor",
                inputs={"x": PARENT["x"]},
                outputs={"*": PARENT}
            )
            START >> node >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"x": 20},  # result = 60, triggers "large-result" tag
            request_id="dynamic-only-001",
            user_id="user-dynamic",
            session_id="dynamic-session",
            tracer=tracer,
        )

        assert result["result"] == 60

    @pytest.mark.asyncio
    async def test_tracer_with_merged_tags(self):
        """Test LangfuseTracer with both static and dynamic tags merged."""
        tracer = LangfuseTracer(
            resource_key="langfuse:vpbank",
            tags=["static-tag", "env:production"]
        )

        @code_node
        def process_with_dynamic(x: int):
            return {"result": x + 100, "$tags": ["dynamic-tag", "processed"]}

        with GraphNode(name="merged-tags-workflow") as graph:
            node = process_with_dynamic(
                name="processor",
                inputs={"x": PARENT["x"]},
                outputs={"*": PARENT}
            )
            START >> node >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"x": 50},
            request_id="merged-tags-001",
            user_id="user-merged",
            session_id="merged-session",
            tracer=tracer,
        )

        assert result["result"] == 150

    @pytest.mark.asyncio
    async def test_tracer_tags_in_config(self):
        """Test that tags are correctly stored in tracer config."""
        tracer = LangfuseTracer(
            resource_key="langfuse:vpbank",
            tags=["test-tag", "flush-check"]
        )

        # Check tags are stored in tracer
        assert tracer.tags == ["test-tag", "flush-check"]

        # Check tracer config includes resource_key
        config = tracer._get_tracer_config()
        assert config["resource_key"] == "langfuse:vpbank"

        with GraphNode(name="tags-config-test") as graph:
            node = CodeNode(
                name="processor",
                code_fn=lambda x: {"y": x},
                inputs={"x": PARENT["x"]},
                outputs={"*": PARENT}
            )
            START >> node >> END

        engine = Hush(graph)
        result = await engine.run(
            inputs={"x": 1},
            request_id="tags-config-001",
            user_id="test-user",
            session_id="test-session",
            tracer=tracer,
        )

        assert result["y"] == 1


# ============================================================================
# Run tests directly
# ============================================================================

if __name__ == "__main__":
    import asyncio

    tracer = LangfuseTracer(resource_key="langfuse:vpbank")

    print("Running workflow integration tests with Langfuse tracing...")
    print("=" * 60)

    # CodeNode tests
    print("\n1. Testing simple CodeNode workflow...")
    asyncio.run(TestCodeNodeWorkflow().test_simple_code_node_workflow(tracer))
    print("   PASSED")

    print("\n2. Testing CodeNode trace metadata...")
    asyncio.run(TestCodeNodeWorkflow().test_code_node_trace_metadata(tracer))
    print("   PASSED")

    print("\n3. Testing chained CodeNodes...")
    asyncio.run(TestCodeNodeWorkflow().test_chained_code_nodes(tracer))
    print("   PASSED")

    # BranchNode tests
    print("\n4. Testing BranchNode workflow...")
    asyncio.run(TestBranchNodeWorkflow().test_simple_branch_workflow(tracer))
    print("   PASSED")

    print("\n5. Testing BranchNode trace metadata...")
    asyncio.run(TestBranchNodeWorkflow().test_branch_trace_metadata(tracer))
    print("   PASSED")

    # ParserNode tests
    print("\n6. Testing XML ParserNode workflow...")
    asyncio.run(TestParserNodeWorkflow().test_xml_parser_workflow(tracer))
    print("   PASSED")

    print("\n7. Testing JSON ParserNode workflow...")
    asyncio.run(TestParserNodeWorkflow().test_json_parser_workflow(tracer))
    print("   PASSED")

    # Tracer tests
    print("\n8. Testing prepare_flush_data structure...")
    asyncio.run(TestLangfuseTracerIntegration().test_prepare_flush_data_structure(tracer))
    print("   PASSED")

    print("\n9. Testing workflow with tracer...")
    asyncio.run(TestLangfuseTracerIntegration().test_workflow_with_tracer(tracer))
    print("   PASSED")

    # Complex workflow tests
    print("\n10. Testing pipeline with branch and parser...")
    asyncio.run(TestComplexWorkflow().test_pipeline_with_branch_and_parser(tracer))
    print("   PASSED")

    print("\n11. Testing full trace collection...")
    asyncio.run(TestComplexWorkflow().test_full_trace_collection(tracer))
    print("   PASSED")

    print("\n12. Testing flush_data contains IO values...")
    asyncio.run(TestComplexWorkflow().test_trace_flush_data_contains_io_values(tracer))
    print("   PASSED")

    # Decorator test
    print("\n13. Testing @code_node decorator...")
    asyncio.run(TestCodeNodeDecorator().test_decorated_code_node(tracer))
    print("   PASSED")

    # Callable syntax test
    print("\n14. Testing callable syntax...")
    asyncio.run(TestCallableSyntax().test_callable_syntax(tracer))
    print("   PASSED")

    # LLMNode tests
    print("\n15. Testing simple LLMNode workflow...")
    asyncio.run(TestLLMNodeWorkflow().test_simple_llm_node(tracer))
    print("   PASSED")

    print("\n16. Testing LLMNode trace metadata...")
    asyncio.run(TestLLMNodeWorkflow().test_llm_node_trace_metadata(tracer))
    print("   PASSED")

    print("\n17. Testing LLMNode with temperature...")
    asyncio.run(TestLLMNodeWorkflow().test_llm_node_with_temperature(tracer))
    print("   PASSED")

    # LLMChainNode tests
    print("\n18. Testing simple LLMChainNode...")
    asyncio.run(TestLLMChainNodeWorkflow().test_simple_llm_chain(tracer))
    print("   PASSED")

    print("\n19. Testing LLMChainNode with parser...")
    asyncio.run(TestLLMChainNodeWorkflow().test_llm_chain_with_parser(tracer))
    print("   PASSED")

    print("\n20. Testing LLMChainNode trace hierarchy...")
    asyncio.run(TestLLMChainNodeWorkflow().test_llm_chain_trace_hierarchy(tracer))
    print("   PASSED")

    print("\n21. Testing LLMChainNode flush data...")
    asyncio.run(TestLLMChainNodeWorkflow().test_llm_chain_flush_data(tracer))
    print("   PASSED")

    # Complex Loop Workflow tests
    print("\n22. Testing MapNode workflow...")
    asyncio.run(TestComplexLoopWorkflows().test_map_node_workflow(tracer))
    print("   PASSED")

    print("\n23. Testing MapNode with static tags...")
    asyncio.run(TestComplexLoopWorkflows().test_map_node_with_static_tags())
    print("   PASSED")

    print("\n24. Testing MapNode with dynamic tags...")
    asyncio.run(TestComplexLoopWorkflows().test_map_node_with_dynamic_tags())
    print("   PASSED")

    print("\n25. Testing ForLoopNode workflow...")
    asyncio.run(TestComplexLoopWorkflows().test_for_loop_workflow(tracer))
    print("   PASSED")

    print("\n26. Testing nested ForLoopNodes...")
    asyncio.run(TestComplexLoopWorkflows().test_nested_for_loops(tracer))
    print("   PASSED")

    print("\n27. Testing nested ForLoopNodes with tags...")
    asyncio.run(TestComplexLoopWorkflows().test_nested_for_loops_with_tags())
    print("   PASSED")

    print("\n28. Testing WhileLoopNode workflow...")
    asyncio.run(TestComplexLoopWorkflows().test_while_loop_workflow(tracer))
    print("   PASSED")

    print("\n29. Testing WhileLoopNode with tags...")
    asyncio.run(TestComplexLoopWorkflows().test_while_loop_with_tags())
    print("   PASSED")

    print("\n30. Testing complex pipeline with all loop types...")
    asyncio.run(TestComplexLoopWorkflows().test_complex_pipeline_with_all_loop_types())
    print("   PASSED")

    # Tracer Tag tests
    print("\n31. Testing tracer with static tags only...")
    asyncio.run(TestLangfuseTracerWithTags().test_tracer_with_static_tags_only())
    print("   PASSED")

    print("\n32. Testing tracer with dynamic tags only...")
    asyncio.run(TestLangfuseTracerWithTags().test_tracer_with_dynamic_tags_only())
    print("   PASSED")

    print("\n33. Testing tracer with merged tags...")
    asyncio.run(TestLangfuseTracerWithTags().test_tracer_with_merged_tags())
    print("   PASSED")

    print("\n34. Testing tracer tags in config...")
    asyncio.run(TestLangfuseTracerWithTags().test_tracer_tags_in_config())
    print("   PASSED")

    print("\n" + "=" * 60)
    print("All workflow integration tests completed!")
    BaseTracer.shutdown_executor()
    print("Traces have been flushed to Langfuse (langfuse:vpbank)")
