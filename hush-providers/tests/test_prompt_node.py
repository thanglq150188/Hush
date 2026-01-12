"""Tests for PromptNode functionality."""

import pytest
from hush.core.states import StateSchema, MemoryState


class TestPromptNode:
    """Tests for PromptNode."""

    def test_import(self):
        """Test PromptNode can be imported."""
        from hush.providers.nodes import PromptNode
        assert PromptNode is not None

    def test_simple_prompt_creation(self):
        """Test creating a simple PromptNode with system and user prompts."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="test_prompt",
            system_prompt="You are {assistant_name}.",
            user_prompt="Help me with {task}."
        )

        assert node.name == "test_prompt"
        assert node.type == "prompt"
        assert node.system_prompt == "You are {assistant_name}."
        assert node.user_prompt == "Help me with {task}."
        assert "assistant_name" in node.inputs
        assert "task" in node.inputs

    def test_messages_template_creation(self):
        """Test creating PromptNode with complex messages_template."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="vision_prompt",
            messages_template=[
                {"role": "system", "content": "You are a vision expert."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze: {query}"},
                    {"type": "image_url", "image_url": {"url": "{image_url}"}}
                ]}
            ]
        )

        assert node.name == "vision_prompt"
        assert "query" in node.inputs
        assert "image_url" in node.inputs

    def test_variable_extraction(self):
        """Test that variables are correctly extracted from templates."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="multi_var",
            system_prompt="Context: {context}, Time: {time}",
            user_prompt="Query: {query}, Format: {format}"
        )

        assert "context" in node.inputs
        assert "time" in node.inputs
        assert "query" in node.inputs
        assert "format" in node.inputs

    @pytest.mark.asyncio
    async def test_format_simple_prompts(self):
        """Test formatting simple prompts."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="format_test",
            system_prompt="You are {name}.",
            user_prompt="Help with {task}."
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema, inputs={
            "name": "Claude",
            "task": "coding"
        })

        result = await node.run(state)

        assert "messages" in result
        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are Claude."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Help with coding."

    @pytest.mark.asyncio
    async def test_format_with_conversation_history(self):
        """Test formatting with conversation history injection."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="history_test",
            system_prompt="You are an assistant.",
            user_prompt="Continue: {message}"
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema, inputs={
            "message": "What was my question?",
            "conversation_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        })

        result = await node.run(state)
        messages = result["messages"]

        # Should have: system, history (2 messages), user
        assert len(messages) == 4

    def test_output_schema(self):
        """Test that output schema has 'messages' key."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="output_test",
            user_prompt="Test {var}"
        )

        assert "messages" in node.outputs

    def test_metadata(self):
        """Test specific_metadata returns correct info."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="metadata_test",
            system_prompt="System prompt",
            user_prompt="User prompt"
        )

        metadata = node.specific_metadata()
        assert metadata["system_prompt"] == "System prompt"
        assert metadata["user_prompt"] == "User prompt"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
