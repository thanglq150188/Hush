"""Prompt Node for hush-providers.

This module provides PromptNode for building chat messages from templates.
Flat input design: reserved keys for prompts, everything else is template variables.
"""

from typing import Dict, Any, Optional, List

from hush.core.nodes import BaseNode
from hush.core.configs import NodeType
from hush.core.utils.common import Param


# Reserved keys that are not template variables
RESERVED_KEYS = frozenset([
    'system_prompt',
    'user_prompt',
    'messages_template',
    'conversation_history',
    'tool_results',
])


class PromptNode(BaseNode):
    """Node for building chat messages from prompt templates.

    Flat input design - reserved keys for prompts, everything else becomes
    template variables for formatting.

    Reserved keys:
    - system_prompt: Optional system prompt template string
    - user_prompt: User prompt template string
    - messages_template: Optional list of message dicts (takes precedence)
    - conversation_history: Optional list of previous messages
    - tool_results: Optional list of tool results

    All other input keys are used as template variables.

    Example:
        ```python
        # Simple usage - flat inputs
        prompt = PromptNode(
            name="chat_prompt",
            inputs={
                "system_prompt": "You are {role}.",
                "user_prompt": "Help me with: {task}",
                "role": "Claude",      # template var
                "task": "coding"       # template var
            }
        )

        # With PARENT forwarding
        prompt = PromptNode(
            name="prompt",
            inputs={
                "system_prompt": "You are helpful.",
                "user_prompt": "Query: {query}",
                "*": PARENT  # forwards query and other vars from parent
            }
        )

        # Complex messages_template
        prompt = PromptNode(
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
                "image_url": "https://..."
            }
        )
        ```
    """

    __slots__ = ['system_prompt', 'user_prompt', 'messages_template']

    type: NodeType = "prompt"

    # Fixed input schema - only reserved keys
    INPUT_SCHEMA = {
        'system_prompt': Param(type=str, required=False, default=None),
        'user_prompt': Param(type=str, required=False, default=None),
        'messages_template': Param(type=list, required=False, default=None),
        'conversation_history': Param(type=list, required=False, default=[]),
        'tool_results': Param(type=list, required=False, default=[]),
    }

    # Fixed output schema
    OUTPUT_SCHEMA = {
        'messages': Param(type=list, required=True),
    }

    def __init__(
        self,
        inputs: Dict[str, Any] = None,
        outputs: Dict[str, Any] = None,
        **kwargs
    ):
        """Initialize PromptNode.

        Args:
            inputs: Input mappings. Reserved keys for prompts, others are template vars.
            outputs: Output variable mappings
            **kwargs: Additional keyword arguments for BaseNode.
        """
        super().__init__(**kwargs)

        # Copy fixed schema
        parsed_inputs = {
            k: Param(type=v.type, required=v.required, default=v.default)
            for k, v in self.INPUT_SCHEMA.items()
        }
        parsed_outputs = {
            k: Param(type=v.type, required=v.required, default=v.default)
            for k, v in self.OUTPUT_SCHEMA.items()
        }

        # Normalize user inputs
        normalized_inputs = self._normalize_params(inputs)
        normalized_outputs = self._normalize_params(outputs)

        # Add non-reserved keys from user inputs to schema (template variables)
        for key, param in normalized_inputs.items():
            if key not in RESERVED_KEYS and key != "__FORWARD_WILDCARD__":
                parsed_inputs[key] = Param(type=Any, required=False, default=None)

        self.inputs = self._merge_params(parsed_inputs, normalized_inputs)
        self.outputs = self._merge_params(parsed_outputs, normalized_outputs)

        # Extract static values for precompilation (optional optimization)
        self.system_prompt = None
        self.user_prompt = None
        self.messages_template = None

        if 'system_prompt' in self.inputs:
            val = self.inputs['system_prompt'].value
            if isinstance(val, str):
                self.system_prompt = val

        if 'user_prompt' in self.inputs:
            val = self.inputs['user_prompt'].value
            if isinstance(val, str):
                self.user_prompt = val

        if 'messages_template' in self.inputs:
            val = self.inputs['messages_template'].value
            if isinstance(val, list):
                self.messages_template = val

        # Set core function
        self.core = self._format

    def _format_value(self, value: Any, vars: Dict[str, Any]) -> Any:
        """Recursively format template variables in a value."""
        if isinstance(value, str):
            return value.format_map(vars) if '{' in value else value
        elif isinstance(value, dict):
            return {k: self._format_value(v, vars) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._format_value(item, vars) for item in value]
        return value

    def _build_messages(
        self,
        system_prompt: Optional[str],
        user_prompt: Optional[str],
        messages_template: Optional[List[Dict]],
        vars: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build messages from templates and vars."""
        messages = []

        # messages_template takes precedence
        if messages_template:
            for msg in messages_template:
                formatted = self._format_value(msg, vars)
                messages.append(formatted)
            return messages

        # Build from system_prompt + user_prompt
        if system_prompt:
            content = system_prompt.format_map(vars) if '{' in system_prompt else system_prompt
            messages.append({"role": "system", "content": content})

        if user_prompt:
            content = user_prompt.format_map(vars) if '{' in user_prompt else user_prompt
            messages.append({"role": "user", "content": content})

        return messages

    async def _format(self, **kwargs) -> Dict[str, Any]:
        """Build messages from templates and context.

        Pops reserved keys, uses remaining kwargs as template variables.
        """
        # Pop reserved keys
        system_prompt = kwargs.pop('system_prompt', None)
        user_prompt = kwargs.pop('user_prompt', None)
        messages_template = kwargs.pop('messages_template', None)
        conversation_history = kwargs.pop('conversation_history', None) or []
        tool_results = kwargs.pop('tool_results', None) or []

        # Remaining kwargs are template variables
        vars = kwargs

        # Build base messages
        messages = self._build_messages(
            system_prompt, user_prompt, messages_template, vars
        )

        # Insert conversation history before last user message
        if conversation_history:
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get('role') == 'user':
                    last_user_idx = i
                    break

            if last_user_idx is None:
                messages.extend(conversation_history)
            else:
                messages[last_user_idx:last_user_idx] = conversation_history

        # Add tool results at the end
        if tool_results:
            messages.extend(tool_results)

        return {"messages": messages}

    def specific_metadata(self) -> Dict[str, Any]:
        """Return prompt-specific metadata."""
        metadata = {}

        if self.system_prompt:
            metadata["system_prompt"] = self.system_prompt
        if self.user_prompt:
            metadata["user_prompt"] = self.user_prompt
        if self.messages_template:
            metadata["messages_template"] = self.messages_template

        return metadata
