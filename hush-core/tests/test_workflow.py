"""Tests for the Hush workflow orchestrator."""

import pytest
from hush.core import Hush, START, END, CodeNode, PARENT


class TestHushBasic:
    """Basic Hush workflow tests."""

    def test_hush_creation(self):
        """Test Hush can be created."""
        flow = Hush("test-workflow")
        assert flow.name == "test-workflow"
        assert flow.compiled is False
        assert flow.schema is None

    def test_hush_with_description(self):
        """Test Hush with description."""
        flow = Hush("test", description="A test workflow")
        assert flow.description == "A test workflow"

    def test_hush_repr(self):
        """Test Hush string representation."""
        flow = Hush("test")
        assert "test" in repr(flow)
        assert "not compiled" in repr(flow)

        with flow:
            node = CodeNode(name="dummy", code_fn=lambda: None)
            START >> node >> END

        flow.compile()
        assert "compiled" in repr(flow)


class TestHushContextManager:
    """Test Hush context manager behavior."""

    def test_context_manager_enters_and_exits(self):
        """Test context manager properly enters and exits."""
        flow = Hush("test")

        with flow:
            # Inside context, graph should be in building mode
            assert flow.graph._is_building is True

        # After exiting, we can still access the graph
        assert flow.graph is not None

    def test_nodes_registered_in_context(self):
        """Test nodes are registered when created in context."""
        with Hush("test") as flow:
            node1 = CodeNode(name="node1", code_fn=lambda: {"x": 1})
            node2 = CodeNode(name="node2", code_fn=lambda x: {"y": x + 1})
            START >> node1 >> node2 >> END

        assert "node1" in flow.graph._nodes
        assert "node2" in flow.graph._nodes


class TestHushCompile:
    """Test Hush compilation."""

    def test_compile_creates_schema(self):
        """Test compile creates StateSchema."""
        with Hush("test") as flow:
            node = CodeNode(name="node", code_fn=lambda: {"out": 1})
            START >> node >> END

        flow.compile()

        assert flow.compiled is True
        assert flow.schema is not None
        assert "test:workflow" in flow.schema.name

    def test_compile_returns_self(self):
        """Test compile returns self for chaining."""
        with Hush("test") as flow:
            node = CodeNode(name="node", code_fn=lambda: {"out": 1})
            START >> node >> END

        result = flow.compile()
        assert result is flow

    def test_compile_warns_orphan_node(self):
        """Test compile warns about orphan nodes."""
        with Hush("test") as flow:
            # No START >> node connection, but have another valid path
            orphan = CodeNode(name="orphan", code_fn=lambda: {})
            valid = CodeNode(name="valid", code_fn=lambda: {})
            START >> valid >> END

        flow.compile()
        # Should compile but warn about orphan
        assert flow.compiled is True


class TestHushRun:
    """Test Hush workflow execution."""

    @pytest.mark.asyncio
    async def test_run_simple_workflow(self):
        """Test running a simple workflow that returns constant."""
        with Hush("test") as flow:
            node = CodeNode(
                name="constant",
                code_fn=lambda: {"result": 42},
                outputs={"result": PARENT}
            )
            START >> node >> END

        flow.compile()
        result = await flow.run(inputs={})

        assert result == {"result": 42}

    @pytest.mark.asyncio
    async def test_run_without_compile_raises(self):
        """Test run raises if not compiled."""
        with Hush("test") as flow:
            node = CodeNode(name="node", code_fn=lambda: {})
            START >> node >> END

        with pytest.raises(RuntimeError, match="not been compiled"):
            await flow.run(inputs={})

    @pytest.mark.asyncio
    async def test_run_generates_ids(self):
        """Test run generates IDs if not provided."""
        with Hush("test") as flow:
            passthrough = CodeNode(
                name="passthrough",
                code_fn=lambda: {}
            )
            START >> passthrough >> END

        flow.compile()

        # Should not raise
        result = await flow.run(inputs={})
        assert result == {}

    @pytest.mark.asyncio
    async def test_run_with_custom_ids(self):
        """Test run with custom IDs."""
        with Hush("test") as flow:
            passthrough = CodeNode(
                name="passthrough",
                code_fn=lambda: {}
            )
            START >> passthrough >> END

        flow.compile()

        result = await flow.run(
            inputs={},
            user_id="user-123",
            session_id="session-456",
            request_id="request-789"
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_run_multi_node_pipeline(self):
        """Test running a multi-node pipeline with constant outputs."""
        with Hush("pipeline") as flow:
            step1 = CodeNode(
                name="step1",
                code_fn=lambda: {"value": 10}
            )
            step2 = CodeNode(
                name="step2",
                code_fn=lambda: {"final": 20},
                outputs={"final": PARENT}
            )
            START >> step1 >> step2 >> END

        flow.compile()
        result = await flow.run(inputs={})

        assert result == {"final": 20}


class TestHushWithTracer:
    """Test Hush with tracer integration."""

    @pytest.mark.asyncio
    async def test_run_with_none_tracer(self):
        """Test run with tracer=None works."""
        with Hush("test") as flow:
            node = CodeNode(
                name="node",
                code_fn=lambda: {"ok": True}
            )
            START >> node >> END

        flow.compile()
        result = await flow.run(inputs={}, tracer=None)

        assert result == {}  # No outputs mapped to PARENT

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_with_langfuse_tracer(self):
        """Test running workflow with Langfuse tracer.

        This test pushes traces to Langfuse cloud.
        Run with: pytest -m integration
        """
        import time

        try:
            from hush.observability.langfuse import LangfuseTracer, LangfuseConfig
        except ImportError:
            pytest.skip("hush-observability not installed")

        # Langfuse config from resources.yaml
        tracer = LangfuseTracer(
            config=LangfuseConfig(
                public_key="pk-lf-ecd32a21-4c71-4276-b753-dfb8481aa062",
                secret_key="sk-lf-a53719f5-856d-418c-86bf-98c7a09f7105",
                host="https://cloud.langfuse.com"
            )
        )

        # Create a multi-step workflow
        with Hush("hush-integration-test") as flow:
            step1 = CodeNode(
                name="process_input",
                code_fn=lambda: {"processed": "Hello from Hush!"},
                outputs={"processed": PARENT}
            )
            step2 = CodeNode(
                name="transform",
                code_fn=lambda: {"transformed": "Data transformed"},
            )
            step3 = CodeNode(
                name="finalize",
                code_fn=lambda: {"result": "Workflow complete!"},
                outputs={"result": PARENT}
            )
            START >> step1 >> step2 >> step3 >> END

        flow.compile()

        result = await flow.run(
            inputs={},
            tracer=tracer,
            user_id="test-user",
            session_id="test-session"
        )

        # Wait for background flush to complete
        time.sleep(2)

        assert result == {"processed": "Hello from Hush!", "result": "Workflow complete!"}
        print(f"\nTrace pushed to Langfuse: https://cloud.langfuse.com")


class TestHushShow:
    """Test Hush show/debug methods."""

    def test_show_not_compiled(self, capsys):
        """Test show before compile."""
        flow = Hush("test")
        flow.show()

        captured = capsys.readouterr()
        assert "not compiled" in captured.out

    def test_show_compiled(self, capsys):
        """Test show after compile."""
        with Hush("test") as flow:
            node = CodeNode(name="node", code_fn=lambda: {})
            START >> node >> END

        flow.compile()
        flow.show()

        captured = capsys.readouterr()
        assert "Workflow: test" in captured.out