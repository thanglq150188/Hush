"""Tests for PromptNode functionality."""

import pytest
from hush.core.states import StateSchema, MemoryState


class TestPromptNodeUnified:
    """Tests for PromptNode with unified prompt parameter."""

    def test_import(self):
        """Test PromptNode can be imported."""
        from hush.providers.nodes import PromptNode
        assert PromptNode is not None

    def test_string_prompt_creation(self):
        """Test creating PromptNode with string prompt (user message only)."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="string_prompt",
            inputs={
                "prompt": "Hello {name}, help me with {task}.",
                "name": "Alice",
                "task": "coding"
            }
        )

        assert node.name == "string_prompt"
        assert node.type == "prompt"
        assert "name" in node.inputs
        assert "task" in node.inputs

    def test_dict_prompt_creation(self):
        """Test creating PromptNode with dict prompt (system/user)."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="dict_prompt",
            inputs={
                "prompt": {
                    "system": "You are {role}.",
                    "user": "Help me with {task}."
                },
                "role": "Claude",
                "task": "coding"
            }
        )

        assert node.name == "dict_prompt"
        assert "role" in node.inputs
        assert "task" in node.inputs

    def test_list_prompt_creation(self):
        """Test creating PromptNode with list prompt (full messages)."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="list_prompt",
            inputs={
                "prompt": [
                    {"role": "system", "content": "You are {role}."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Analyze: {query}"},
                        {"type": "image_url", "image_url": {"url": "{image_url}"}}
                    ]}
                ],
                "role": "a vision expert",
                "query": "What is this?",
                "image_url": "https://example.com/image.png"
            }
        )

        assert node.name == "list_prompt"
        assert "query" in node.inputs
        assert "image_url" in node.inputs

    @pytest.mark.asyncio
    async def test_format_string_prompt(self):
        """Test formatting string prompt (user message only)."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="format_string",
            inputs={
                "prompt": "Hello {name}, help me with {task}.",
                "name": "Alice",
                "task": "coding"
            }
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema)

        result = await node.run(state)

        assert "messages" in result
        messages = result["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello Alice, help me with coding."

    @pytest.mark.asyncio
    async def test_format_dict_prompt(self):
        """Test formatting dict prompt with system/user keys."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="format_dict",
            inputs={
                "prompt": {
                    "system": "You are {role}.",
                    "user": "Help with {task}."
                },
                "role": "Claude",
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
    async def test_format_dict_prompt_user_only(self):
        """Test formatting dict prompt with only user key."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="format_dict_user",
            inputs={
                "prompt": {"user": "Hello {name}!"},
                "name": "Bob"
            }
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema)

        result = await node.run(state)

        messages = result["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello Bob!"

    @pytest.mark.asyncio
    async def test_format_list_prompt(self):
        """Test formatting list prompt (full messages array)."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="format_list",
            inputs={
                "prompt": [
                    {"role": "system", "content": "You are {role}."},
                    {"role": "user", "content": "Help with {task}."}
                ],
                "role": "an assistant",
                "task": "math"
            }
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema)

        result = await node.run(state)

        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0]["content"] == "You are an assistant."
        assert messages[1]["content"] == "Help with math."

    @pytest.mark.asyncio
    async def test_format_list_prompt_multimodal(self):
        """Test formatting list prompt with multimodal content."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="format_multimodal",
            inputs={
                "prompt": [
                    {"role": "system", "content": "You are a vision expert."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe: {query}"},
                        {"type": "image_url", "image_url": {"url": "{image_url}"}}
                    ]}
                ],
                "query": "this image",
                "image_url": "https://example.com/cat.jpg"
            }
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema)

        result = await node.run(state)

        messages = result["messages"]
        assert len(messages) == 2
        assert messages[1]["content"][0]["text"] == "Describe: this image"
        assert messages[1]["content"][1]["image_url"]["url"] == "https://example.com/cat.jpg"

    @pytest.mark.asyncio
    async def test_prompt_with_conversation_history(self):
        """Test prompt with conversation history injection."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="history_test",
            inputs={
                "prompt": {"system": "You are helpful.", "user": "Continue: {message}"},
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
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == "Continue: What was my question?"

    @pytest.mark.asyncio
    async def test_prompt_with_tool_results(self):
        """Test prompt with tool results appended."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="tool_test",
            inputs={
                "prompt": {"user": "What's the weather?"},
                "tool_results": [
                    {"role": "tool", "content": "Weather: Sunny, 25C", "tool_call_id": "call_123"}
                ]
            }
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema)

        result = await node.run(state)
        messages = result["messages"]

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "tool"
        assert messages[1]["tool_call_id"] == "call_123"

    def test_metadata_with_prompt(self):
        """Test specific_metadata returns prompt info."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="metadata_test",
            inputs={
                "prompt": {"system": "System prompt", "user": "User prompt"}
            }
        )

        metadata = node.specific_metadata()
        assert "prompt" in metadata
        assert metadata["prompt"]["system"] == "System prompt"
        assert metadata["prompt"]["user"] == "User prompt"


class TestPromptNodeLegacy:
    """Tests for PromptNode backward compatibility with legacy API."""

    def test_legacy_prompt_creation(self):
        """Test creating PromptNode with legacy system_prompt/user_prompt."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="legacy_prompt",
            inputs={
                "system_prompt": "You are {assistant_name}.",
                "user_prompt": "Help me with {task}.",
                "assistant_name": "Claude",
                "task": "coding"
            }
        )

        assert node.name == "legacy_prompt"
        assert node.type == "prompt"
        assert "assistant_name" in node.inputs
        assert "task" in node.inputs

    def test_legacy_messages_template_creation(self):
        """Test creating PromptNode with legacy messages_template."""
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
        assert "prompt" in node.inputs
        assert "system_prompt" in node.inputs
        assert "user_prompt" in node.inputs
        assert "messages_template" in node.inputs
        assert "conversation_history" in node.inputs
        assert "tool_results" in node.inputs

    @pytest.mark.asyncio
    async def test_format_legacy_prompts(self):
        """Test formatting legacy prompts."""
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
    async def test_format_with_conversation_history_legacy(self):
        """Test formatting with conversation history injection (legacy)."""
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

    def test_metadata_legacy(self):
        """Test specific_metadata returns correct info for legacy format."""
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
                "prompt": "Hello {name}, teach me about {topic}.",
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
                "prompt": {"system": "You are {role}.", "user": "Task: {task}"},
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
                "prompt": {"system": "You are helpful.", "user": "Hello!"}
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
    """Tests for PromptNode with messages_template (legacy)."""

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
        """Test that messages_template takes precedence over prompts (legacy)."""
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


class TestPromptNodePromptPrecedence:
    """Tests for prompt parameter precedence."""

    @pytest.mark.asyncio
    async def test_new_prompt_over_legacy(self):
        """Test that new `prompt` takes precedence over legacy keys."""
        from hush.providers.nodes import PromptNode

        node = PromptNode(
            name="precedence_test",
            inputs={
                "prompt": {"user": "New prompt wins!"},
                "system_prompt": "Legacy system (ignored)",
                "user_prompt": "Legacy user (ignored)"
            }
        )

        schema = StateSchema(node=node)
        state = MemoryState(schema)

        result = await node.run(state)

        messages = result["messages"]
        assert len(messages) == 1
        assert messages[0]["content"] == "New prompt wins!"

    @pytest.mark.asyncio
    async def test_dynamic_prompt_from_state(self):
        """Test receiving prompt dynamically from state (Pattern 2)."""
        from hush.providers.nodes import PromptNode

        # Node with prompt declared but no default value
        node = PromptNode(
            name="dynamic_prompt",
            inputs={
                "prompt": None,  # Will be provided at runtime
                "name": "default"
            }
        )

        schema = StateSchema(node=node)
        # Provide prompt at runtime via state
        state = MemoryState(schema, inputs={
            "prompt": {"system": "You are {name}.", "user": "Hello!"},
            "name": "a dynamic assistant"
        })

        result = await node.run(state)

        messages = result["messages"]
        assert len(messages) == 2
        assert messages[0]["content"] == "You are a dynamic assistant."
        assert messages[1]["content"] == "Hello!"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
