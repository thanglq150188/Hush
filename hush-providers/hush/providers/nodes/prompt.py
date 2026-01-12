"""Prompt Node for hush-providers.

This module provides PromptNode for building chat messages from templates.
"""

from typing import Dict, Any, Optional, List, Set
from functools import lru_cache
import re
import asyncio

from hush.core.nodes import BaseNode
from hush.core.configs import NodeType
from hush.core.utils.common import Param


class VariableExtractor:
    """Efficiently extracts and caches template variables."""

    @staticmethod
    @lru_cache(maxsize=128)
    def from_string(template: str) -> Set[str]:
        """Extract variables from a template string."""
        return set(re.findall(r'{([^}]*)}', template))

    @classmethod
    def from_content(cls, content: Any) -> Set[str]:
        """Extract variables from any content type recursively."""
        variables = set()

        if isinstance(content, str):
            variables.update(cls.from_string(content))
        elif isinstance(content, list):
            for item in content:
                variables.update(cls.from_content(item))
        elif isinstance(content, dict):
            for value in content.values():
                variables.update(cls.from_content(value))

        return variables


class OptimizedMessageFormatter:
    """Optimized message formatting with minimal overhead."""

    @staticmethod
    def format_value(value: Any, context: Dict[str, Any]) -> Any:
        """Optimized recursive formatter with type-specific handling."""
        value_type = type(value)

        if value_type is str:
            return value.format_map(context)
        elif value_type is dict:
            return {k: OptimizedMessageFormatter.format_value(v, context) for k, v in value.items()}
        elif value_type is list:
            return [OptimizedMessageFormatter.format_value(item, context) for item in value]

        return value

    @staticmethod
    def format_message(template: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Format a complete message template with minimal copying."""
        return OptimizedMessageFormatter.format_value(template, context)


class PrecompiledMessageBuilder:
    """Pre-compiled message builder that avoids repeated work."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        messages_template: Optional[List[Dict[str, Any]]] = None
    ):
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.messages_template = messages_template

        self._precompiled_templates = None
        self._simple_system_template = None
        self._simple_user_template = None

        self._precompile()

    def _precompile(self):
        """Pre-compile templates to avoid repeated parsing."""
        if self.messages_template:
            self._precompiled_templates = []
            for template in self.messages_template:
                compiled = self._precompile_single_template(template)
                self._precompiled_templates.append(compiled)

        if self.system_prompt:
            self._simple_system_template = self._precompile_string(self.system_prompt)
        if self.user_prompt:
            self._simple_user_template = self._precompile_string(self.user_prompt)

    def _precompile_string(self, template: str):
        """Pre-compile a string template for faster formatting."""
        if '{' not in template:
            return {'static': template, 'needs_formatting': False}
        return {'template': template, 'needs_formatting': True}

    def _precompile_single_template(self, template: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-compile a single message template."""
        compiled = {}
        for key, value in template.items():
            if isinstance(value, str) and '{' in value:
                compiled[key] = {'template': value, 'needs_formatting': True}
            elif isinstance(value, (dict, list)):
                compiled[key] = {'value': value, 'needs_formatting': True}
            else:
                compiled[key] = {'value': value, 'needs_formatting': False}
        return compiled

    def build_messages(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build messages using pre-compiled templates."""
        messages = []

        if self._precompiled_templates:
            for compiled_template in self._precompiled_templates:
                message = {}
                for key, compiled_value in compiled_template.items():
                    if compiled_value['needs_formatting']:
                        if 'template' in compiled_value:
                            message[key] = compiled_value['template'].format_map(context)
                        else:
                            message[key] = OptimizedMessageFormatter.format_value(
                                compiled_value['value'], context
                            )
                    else:
                        message[key] = compiled_value['value']
                messages.append(message)
        else:
            if self._simple_system_template:
                if self._simple_system_template['needs_formatting']:
                    content = self._simple_system_template['template'].format_map(context)
                else:
                    content = self._simple_system_template['static']
                messages.append({"role": "system", "content": content})

            if self._simple_user_template:
                if self._simple_user_template['needs_formatting']:
                    content = self._simple_user_template['template'].format_map(context)
                else:
                    content = self._simple_user_template['static']
                messages.append({"role": "user", "content": content})

        return messages

    def insert_conversation_history(
        self,
        messages: List[Dict[str, Any]],
        history: List[Dict[str, Any]]
    ) -> None:
        """Insert conversation history in-place for memory efficiency."""
        if not history:
            return

        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get('role') == 'user':
                last_user_idx = i
                break

        if last_user_idx is None:
            messages.extend(history)
        else:
            messages[last_user_idx:last_user_idx] = history


class PromptNode(BaseNode):
    """Node for building chat messages from prompt templates.

    Supports two modes:
    1. Simple mode: system_prompt + user_prompt with {variable} placeholders
    2. Complex mode: messages_template for multi-turn conversations with images

    Features:
    - Auto-variable extraction from templates
    - Pre-compiled templates for fast execution
    - Conversation history injection
    - Tool results support

    Example:
        ```python
        # Simple mode
        prompt = PromptNode(
            name="chat_prompt",
            system_prompt="You are {assistant_name}, a helpful assistant.",
            user_prompt="Help me with: {task}",
            inputs={"assistant_name": "Claude", "task": INPUT}
        )

        # Complex mode with images
        prompt = PromptNode(
            name="vision_prompt",
            messages_template=[
                {"role": "system", "content": "You are a vision expert."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze: {query}"},
                    {"type": "image_url", "image_url": {"url": "{image_url}"}}
                ]}
            ],
            inputs={"query": INPUT, "image_url": INPUT}
        )
        ```
    """

    __slots__ = ['system_prompt', 'user_prompt', 'messages_template', '_message_builder']

    type: NodeType = "prompt"

    def __init__(
        self,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        messages_template: Optional[List[Dict[str, Any]]] = None,
        inputs: Dict[str, Any] = None,
        outputs: Dict[str, Any] = None,
        **kwargs
    ):
        """Initialize PromptNode.

        Args:
            user_prompt: User prompt template with {variable} placeholders
            system_prompt: System prompt template with {variable} placeholders
            messages_template: Complex message template for multi-turn conversations
            inputs: Input variable mappings
            outputs: Output variable mappings
            **kwargs: Additional keyword arguments for BaseNode
        """
        # Initialize base without inputs/outputs first
        super().__init__(**kwargs)

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.messages_template = messages_template

        # Pre-compile message builder
        self._message_builder = PrecompiledMessageBuilder(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            messages_template=messages_template
        )

        # Build input schema from extracted variables
        parsed_inputs = self._build_input_schema()
        parsed_outputs = {"messages": Param(type=list, required=True)}

        # Merge with user-provided
        self.inputs = self._merge_params(parsed_inputs, inputs)
        self.outputs = self._merge_params(parsed_outputs, outputs)

        # Set core function
        self.core = self._format

    def _extract_variables(self) -> Set[str]:
        """Extract template variables from all templates."""
        variables = set()

        if self.messages_template:
            for msg in self.messages_template:
                variables.update(VariableExtractor.from_content(msg))

        if self.user_prompt:
            variables.update(VariableExtractor.from_string(self.user_prompt))

        if self.system_prompt:
            variables.update(VariableExtractor.from_string(self.system_prompt))

        return variables

    def _build_input_schema(self) -> Dict[str, Param]:
        """Build input schema based on extracted variables."""
        schema = {}

        for variable in self._extract_variables():
            schema[variable] = Param(required=True)

        # Add optional enhanced features
        schema['conversation_history'] = Param(type=list, default=[])
        schema['tool_results'] = Param(type=list, default=[])

        return schema

    async def _format(self, **kwargs) -> Dict[str, Any]:
        """Build messages from templates and context."""
        # Build base messages using pre-compiled templates
        messages = self._message_builder.build_messages(kwargs)

        # Add conversation history
        history = kwargs.get('conversation_history')
        if history:
            self._message_builder.insert_conversation_history(messages, history)

        # Add tool results
        tool_results = kwargs.get('tool_results')
        if tool_results:
            messages.extend(tool_results)

        return {"messages": messages}

    def specific_metadata(self) -> Dict[str, Any]:
        """Return prompt-specific metadata."""
        metadata = {}

        if self.messages_template:
            metadata["messages_template"] = self.messages_template
        else:
            if self.system_prompt:
                metadata["system_prompt"] = self.system_prompt
            if self.user_prompt:
                metadata["user_prompt"] = self.user_prompt

        return metadata
