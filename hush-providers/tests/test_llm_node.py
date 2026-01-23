"""Tests for LLMNode functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestLLMNode:
    """Tests for LLMNode."""

    def test_import(self):
        """Test LLMNode can be imported."""
        from hush.providers.nodes import LLMNode
        assert LLMNode is not None

    def test_node_type_instant_response(self):
        """Test LLMNode has correct type with instant_response mode."""
        from hush.providers.nodes import LLMNode

        node = LLMNode(
            name="test_llm",
            resource_key="gpt-4",
            instant_response=True
        )

        assert node.type == "llm"
        assert node.resource_key == "gpt-4"
        assert node.instant_response is True

    def test_input_schema(self):
        """Test LLMNode has required input schema."""
        from hush.providers.nodes import LLMNode

        node = LLMNode(
            name="schema_test",
            resource_key="gpt-4",
            instant_response=True
        )

        assert "messages" in node.inputs

    def test_output_schema(self):
        """Test LLMNode has expected output schema."""
        from hush.providers.nodes import LLMNode

        node = LLMNode(
            name="output_test",
            resource_key="gpt-4",
            instant_response=True
        )

        assert "content" in node.outputs
        assert "role" in node.outputs
        assert "model_used" in node.outputs

    def test_streaming_mode(self):
        """Test LLMNode can be created in streaming mode."""
        from hush.providers.nodes import LLMNode

        node = LLMNode(
            name="stream_test",
            resource_key="gpt-4",
            instant_response=True,
            stream=True
        )

        assert node.stream is True

    def test_metadata(self):
        """Test specific_metadata returns model info."""
        from hush.providers.nodes import LLMNode

        node = LLMNode(
            name="metadata_test",
            resource_key="gpt-4",
            instant_response=True
        )

        metadata = node.specific_metadata()
        assert metadata["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_instant_response_execution(self):
        """Test LLMNode instant response mode execution."""
        from hush.providers.nodes import LLMNode
        from hush.core.states import StateSchema, MemoryState

        node = LLMNode(
            name="instant_test",
            resource_key="test-model",
            instant_response=True
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema, inputs={
            "messages": [{"role": "user", "content": "Hello"}]
        })

        result = await node.run(state)

        assert "content" in result
        assert result["role"] == "assistant"
        assert result["model_used"] == "test-model"


class TestLLMNodeIntegration:
    """Integration tests for LLMNode with real ResourceHub."""

    @pytest.mark.asyncio
    async def test_llm_node_with_hub(self, hub):
        """Test LLMNode works with ResourceHub."""
        from hush.providers.nodes import LLMNode
        from hush.core.states import StateSchema, MemoryState

        # Check if or-claude-4-sonnet is available
        if not hub.has("llm:or-claude-4-sonnet"):
            pytest.skip("llm:or-claude-4-sonnet not configured in resources.yaml")

        node = LLMNode(
            name="chat",
            resource_key="or-claude-4-sonnet"
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema, inputs={
            "messages": [{"role": "user", "content": "Say 'Hello' in exactly one word."}]
        })

        result = await node.run(state)

        assert "content" in result
        assert len(result["content"]) > 0
        print(f"LLM Response: {result['content']}")

    @pytest.mark.asyncio
    async def test_llm_node_streaming_with_tokens(self, hub):
        """Test LLMNode streaming mode with token verification via STREAM_SERVICE."""
        import asyncio
        from hush.providers.nodes import LLMNode
        from hush.core.states import StateSchema, MemoryState
        from hush.core import STREAM_SERVICE

        # Check if or-claude-4-sonnet is available
        if not hub.has("llm:or-claude-4-sonnet"):
            pytest.skip("llm:or-claude-4-sonnet not configured in resources.yaml")

        node = LLMNode(
            name="stream_chat",
            resource_key="or-claude-4-sonnet",
            stream=True
        )

        schema = StateSchema(node=node)
        request_id = "test-stream-request-001"
        state = MemoryState(
            schema,
            inputs={"messages": [{"role": "user", "content": "Count from 1 to 5, one number per line."}]},
            request_id=request_id
        )

        # Collect streamed chunks
        chunks_received = []
        content_parts = []

        async def collect_chunks():
            """Collect chunks from STREAM_SERVICE."""
            channel_name = node.identity(None)
            async for chunk in STREAM_SERVICE.get(request_id, channel_name):
                chunks_received.append(chunk)
                if chunk.choices and chunk.choices[0].delta.content:
                    content_parts.append(chunk.choices[0].delta.content)

        # Run streaming and chunk collection concurrently
        collector_task = asyncio.create_task(collect_chunks())

        # Execute the node (this will push chunks to STREAM_SERVICE)
        result = await node.run(state)

        # Give time for background tasks to complete pushing to STREAM_SERVICE
        await asyncio.sleep(0.1)

        # Wait for collector to finish (should end when STREAM_SERVICE.end() is called)
        await asyncio.wait_for(collector_task, timeout=5.0)

        # Verify chunks were received
        assert len(chunks_received) > 0, "Should receive streaming chunks"
        print(f"Received {len(chunks_received)} chunks")

        # Verify content parts were collected
        streamed_content = "".join(content_parts)
        assert len(streamed_content) > 0, "Should have streamed content"
        print(f"Streamed content: {streamed_content}")

        # Verify final result matches accumulated stream
        assert "content" in result
        assert result["content"] == streamed_content, "Final result should match streamed content"
        print(f"Final result content: {result['content']}")

        # Verify tokens_used is populated
        assert "tokens_used" in result
        if result["tokens_used"]:
            print(f"Tokens used: {result['tokens_used']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
