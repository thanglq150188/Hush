"""Tests for LLMChainNode functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestLLMChainNode:
    """Tests for LLMChainNode."""

    def test_import(self):
        """Test LLMChainNode can be imported."""
        from hush.providers.nodes import LLMChainNode
        assert LLMChainNode is not None

    def test_simple_chain_creation(self):
        """Test creating a simple LLMChainNode (text generation mode)."""
        from hush.providers.nodes import LLMChainNode

        with patch('hush.providers.nodes.llm.ResourceHub') as mock_hub:
            mock_instance = Mock()
            mock_instance.llm.return_value = Mock(
                generate=AsyncMock(),
                stream=AsyncMock()
            )
            mock_hub.instance.return_value = mock_instance

            node = LLMChainNode(
                name="simple_chain",
                resource_key="gpt-4",
                system_prompt="You are helpful.",
                user_prompt="Help with: {task}"
            )

            assert node.name == "simple_chain"
            assert node.type == "graph"  # LLMChainNode is a GraphNode
            assert node.resource_key == "gpt-4"

    def test_structured_chain_creation(self):
        """Test creating LLMChainNode with structured output (parser mode)."""
        from hush.providers.nodes import LLMChainNode

        with patch('hush.providers.nodes.llm.ResourceHub') as mock_hub:
            mock_instance = Mock()
            mock_instance.llm.return_value = Mock(
                generate=AsyncMock(),
                stream=AsyncMock()
            )
            mock_hub.instance.return_value = mock_instance

            node = LLMChainNode(
                name="structured_chain",
                resource_key="gpt-4",
                user_prompt="Classify: {text}\n<category>...</category>",
                extract_schema=["category: str", "confidence: float"],
                parser="xml"
            )

            assert node.extract_schema == ["category: str", "confidence: float"]
            assert node.parser == "xml"

    def test_chain_with_messages_template(self):
        """Test creating LLMChainNode with complex messages_template."""
        from hush.providers.nodes import LLMChainNode

        with patch('hush.providers.nodes.llm.ResourceHub') as mock_hub:
            mock_instance = Mock()
            mock_instance.llm.return_value = Mock(
                generate=AsyncMock(),
                stream=AsyncMock()
            )
            mock_hub.instance.return_value = mock_instance

            node = LLMChainNode(
                name="vision_chain",
                resource_key="gpt-4o",
                messages_template=[
                    {"role": "system", "content": "You are a vision expert."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Analyze: {query}"},
                        {"type": "image_url", "image_url": {"url": "{image_url}"}}
                    ]}
                ]
            )

            assert node.messages_template is not None

    def test_chain_has_internal_nodes(self):
        """Test that LLMChainNode creates internal nodes."""
        from hush.providers.nodes import LLMChainNode

        with patch('hush.providers.nodes.llm.ResourceHub') as mock_hub:
            mock_instance = Mock()
            mock_instance.llm.return_value = Mock(
                generate=AsyncMock(),
                stream=AsyncMock()
            )
            mock_hub.instance.return_value = mock_instance

            node = LLMChainNode(
                name="internal_test",
                resource_key="gpt-4",
                user_prompt="Test {var}"
            )

            # Should have internal nodes (prompt, llm)
            assert "prompt" in node._nodes
            assert "llm" in node._nodes

    def test_chain_with_parser_has_parser_node(self):
        """Test that LLMChainNode with extract_schema has parser node."""
        from hush.providers.nodes import LLMChainNode

        with patch('hush.providers.nodes.llm.ResourceHub') as mock_hub:
            mock_instance = Mock()
            mock_instance.llm.return_value = Mock(
                generate=AsyncMock(),
                stream=AsyncMock()
            )
            mock_hub.instance.return_value = mock_instance

            node = LLMChainNode(
                name="parser_test",
                resource_key="gpt-4",
                user_prompt="Extract: {text}",
                extract_schema=["result: str"]
            )

            # Should have parser node when extract_schema is provided
            assert "parser" in node._nodes

    def test_metadata(self):
        """Test specific_metadata returns chain info."""
        from hush.providers.nodes import LLMChainNode

        with patch('hush.providers.nodes.llm.ResourceHub') as mock_hub:
            mock_instance = Mock()
            mock_instance.llm.return_value = Mock(
                generate=AsyncMock(),
                stream=AsyncMock()
            )
            mock_hub.instance.return_value = mock_instance

            node = LLMChainNode(
                name="metadata_test",
                resource_key="gpt-4",
                system_prompt="System",
                user_prompt="User {var}",
                extract_schema=["field: str"],
                parser="json"
            )

            metadata = node.specific_metadata()
            assert metadata["resource_key"] == "gpt-4"
            assert metadata["system_prompt"] == "System"
            assert metadata["user_prompt"] == "User {var}"
            assert metadata["extract_schema"] == ["field: str"]
            assert metadata["parser"] == "json"


class TestLLMChainNodeIntegration:
    """Integration tests for LLMChainNode with real ResourceHub."""

    @pytest.mark.asyncio
    async def test_llm_chain_simple_generation(self, hub):
        """Test LLMChainNode simple text generation with real LLM."""
        from hush.providers.nodes import LLMChainNode
        from hush.core.states import StateSchema, MemoryState

        # Check if claude-4-sonnet is available
        if not hub.has("llm:claude-4-sonnet"):
            pytest.skip("llm:claude-4-sonnet not configured in resources.yaml")

        node = LLMChainNode(
            name="simple_chain",
            resource_key="claude-4-sonnet",
            system_prompt="You are a helpful assistant.",
            user_prompt="Say hello to {name} in one sentence."
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema, inputs={"name": "Alice"})

        result = await node.run(state)

        assert "content" in result
        print(f"LLMChainNode response: {result['content']}")

    @pytest.mark.asyncio
    async def test_llm_chain_structured_output(self, hub):
        """Test LLMChainNode with structured output parsing."""
        from hush.providers.nodes import LLMChainNode
        from hush.core.states import StateSchema, MemoryState

        # Check if claude-4-sonnet is available
        if not hub.has("llm:claude-4-sonnet"):
            pytest.skip("llm:claude-4-sonnet not configured in resources.yaml")

        node = LLMChainNode(
            name="structured_chain",
            resource_key="claude-4-sonnet",
            user_prompt="""Classify the sentiment of this text: "{text}"

Output your response in XML format:
<sentiment>positive/negative/neutral</sentiment>
<confidence>0.0-1.0</confidence>""",
            extract_schema=["sentiment: str", "confidence: float"],
            parser="xml"
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema, inputs={
            "text": "I love this product! It's amazing!"
        })

        result = await node.run(state)

        assert "sentiment" in result
        assert "confidence" in result
        print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
