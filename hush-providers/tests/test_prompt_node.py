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
            inputs={
                "system_prompt": "You are {assistant_name}.",
                "user_prompt": "Help me with {task}.",
                "assistant_name": "Claude",
                "task": "coding"
            }
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
            inputs={
                "messages_template": [
                    {"role": "system", "content": "You are a vision expert."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Analyze: {query}"},
                        {"type": "image_url", "image_url": {"url": "{image_url}"}}
                    ]}
                ],
                "query": "What is this?",
                "image_url": "https://example.com/image.png"
            }
        )

        assert node.name == "vision_prompt"
        assert "query" in node.inputs
        assert "image_url" in node.inputs

    def test_fixed_schema(self):
        """Test that PromptNode has fixed input schema."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="schema_test",
            inputs={
                "user_prompt": "Test"
            }
        )

        # Should have all fixed schema keys
        assert "system_prompt" in node.inputs
        assert "user_prompt" in node.inputs
        assert "messages_template" in node.inputs
        assert "conversation_history" in node.inputs
        assert "tool_results" in node.inputs

    @pytest.mark.asyncio
    async def test_format_simple_prompts(self):
        """Test formatting simple prompts."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="format_test",
            inputs={
                "system_prompt": "You are {name}.",
                "user_prompt": "Help with {task}.",
                "name": "Claude",
                "task": "coding"
            }
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema)

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
            inputs={
                "system_prompt": "You are an assistant.",
                "user_prompt": "Continue: {message}",
                "message": "What was my question?",
                "conversation_history": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            }
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema)

        result = await node.run(state)
        messages = result["messages"]

        # Should have: system, history (2 messages), user
        assert len(messages) == 4

    def test_output_schema(self):
        """Test that output schema has 'messages' key."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="output_test",
            inputs={
                "user_prompt": "Test"
            }
        )

        assert "messages" in node.outputs

    def test_metadata(self):
        """Test specific_metadata returns correct info."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="metadata_test",
            inputs={
                "system_prompt": "System prompt",
                "user_prompt": "User prompt"
            }
        )

        metadata = node.specific_metadata()
        assert metadata["system_prompt"] == "System prompt"
        assert metadata["user_prompt"] == "User prompt"


class TestPromptNodeWithVars:
    """Tests for PromptNode with template variables."""

    @pytest.mark.asyncio
    async def test_vars_formatting(self):
        """Test that template vars are used for formatting."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="vars_test",
            inputs={
                "user_prompt": "Hello {name}, teach me about {topic}.",
                "name": "Alice",
                "topic": "Python"
            }
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema)

        result = await node.run(state)

        messages = result["messages"]
        assert len(messages) == 1
        assert messages[0]["content"] == "Hello Alice, teach me about Python."

    @pytest.mark.asyncio
    async def test_vars_from_state(self):
        """Test vars passed via state override defaults."""
        from hush.providers.nodes import PromptNode

        # Declare variables in inputs (with defaults) to add them to schema
        node = PromptNode(
            name="vars_state_test",
            inputs={
                "system_prompt": "You are {role}.",
                "user_prompt": "Task: {task}",
                "role": "default role",  # placeholder, will be overridden
                "task": "default task"   # placeholder, will be overridden
            }
        )

        schema = StateSchema(node=node)
        # Override via state inputs
        state = MemoryState(schema, inputs={
            "role": "a helpful assistant",
            "task": "explain code"
        })

        result = await node.run(state)

        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["content"] == "Task: explain code"

    @pytest.mark.asyncio
    async def test_empty_vars(self):
        """Test with no template variables (empty vars)."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="no_vars_test",
            inputs={
                "system_prompt": "You are helpful.",
                "user_prompt": "Hello!"
            }
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema)

        result = await node.run(state)

        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0]["content"] == "You are helpful."
        assert messages[1]["content"] == "Hello!"


class TestPromptNodeMessagesTemplate:
    """Tests for PromptNode with messages_template."""

    @pytest.mark.asyncio
    async def test_messages_template_formatting(self):
        """Test messages_template with vars formatting."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="template_test",
            inputs={
                "messages_template": [
                    {"role": "system", "content": "You are {role}."},
                    {"role": "user", "content": "Help with {task}."}
                ],
                "role": "Claude",
                "task": "coding"
            }
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema)

        result = await node.run(state)

        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0]["content"] == "You are Claude."
        assert messages[1]["content"] == "Help with coding."

    @pytest.mark.asyncio
    async def test_messages_template_precedence(self):
        """Test that messages_template takes precedence over prompts."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="precedence_test",
            inputs={
                "system_prompt": "This should be ignored.",
                "user_prompt": "This too.",
                "messages_template": [
                    {"role": "user", "content": "Only this should appear."}
                ]
            }
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema)

        result = await node.run(state)

        messages = result["messages"]
        assert len(messages) == 1
        assert messages[0]["content"] == "Only this should appear."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
