"""Prompt Node for hush-providers.

This module provides PromptNode for building chat messages from templates.
Unified design: single `prompt` input that accepts multiple formats.
"""

from typing import Dict, Any, Optional, List, Union

from hush.core.nodes import BaseNode
from hush.core.configs import NodeType
from hush.core.utils.common import Param


# Reserved keys that are not template variables
RESERVED_KEYS = frozenset([
    'prompt',
    'conversation_history',
    'tool_results',
    # Legacy keys for backward compatibility
    'system_prompt',
    'user_prompt',
    'messages_template',
])


class PromptNode(BaseNode):
    """Node for building chat messages from prompt templates.

    Unified design - single `prompt` input that accepts 3 formats:

    1. **String** → Simple user message
       ```python
       prompt = "Hello {name}"  # → [{"role": "user", "content": "Hello Alice"}]
       ```

    2. **Dict with system/user keys** → System + user messages
       ```python
       prompt = {
           "system": "You are {role}.",
           "user": "Help with: {task}"
       }
       ```

    3. **List** → Full messages array (for multimodal, complex cases)
       ```python
       prompt = [
           {"role": "system", "content": "You are helpful."},
           {"role": "user", "content": [
               {"type": "text", "text": "Describe: {query}"},
               {"type": "image_url", "image_url": {"url": "{image_url}"}}
           ]}
       ]
       ```

    All other input keys are used as template variables for formatting.

    Example:
        ```python
        # Simple usage - string prompt
        prompt = PromptNode(
            name="chat_prompt",
            inputs={
                "prompt": "Help me with: {task}",
                "task": "coding"
            }
        )

        # Dict prompt with system/user
        prompt = PromptNode(
            name="chat_prompt",
            inputs={
                "prompt": {"system": "You are {role}.", "user": "{query}"},
                "role": "helpful",
                "query": "Hello"
            }
        )

        # Dynamic prompt from parent node (Pattern 2)
        prompt = PromptNode(
            name="prompt",
            inputs={
                "prompt": PARENT["generated_prompt"],  # Dynamic from workflow
                "query": PARENT["query"],
                "*": PARENT
            }
        )

        # With conversation history and tool results
        prompt = PromptNode(
            name="prompt",
            inputs={
                "prompt": {"system": "You are helpful.", "user": "{query}"},
                "conversation_history": PARENT["history"],
                "tool_results": PARENT["tool_results"],
                "*": PARENT
            }
        )
        ```
    """

    __slots__ = []

    type: NodeType = "prompt"

    # Fixed input schema - only reserved keys
    INPUT_SCHEMA = {
        'prompt': Param(type=(str, dict, list), required=False, default=None),
        'conversation_history': Param(type=list, required=False, default=[]),
        'tool_results': Param(type=list, required=False, default=[]),
        # Legacy support
        'system_prompt': Param(type=str, required=False, default=None),
        'user_prompt': Param(type=str, required=False, default=None),
        'messages_template': Param(type=list, required=False, default=None),
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

        # Set core function
        self.core = self._format

    def _format_value(self, value: Any, vars: Dict[str, Any]) -> Any:
        """Recursively format template variables in a value."""
        if isinstance(value, str):
            try:
                return value.format_map(vars) if '{' in value else value
            except KeyError:
                # Return as-is if variable not found
                return value
        elif isinstance(value, dict):
            return {k: self._format_value(v, vars) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._format_value(item, vars) for item in value]
        return value

    def _prompt_to_messages(
        self,
        prompt: Union[str, Dict, List],
        vars: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert prompt to messages array.

        Args:
            prompt: Prompt in one of 3 formats (str, dict, list)
            vars: Template variables for formatting

        Returns:
            List of message dicts
        """
        if prompt is None:
            return []

        # Form 1: String → user message only
        if isinstance(prompt, str):
            content = self._format_value(prompt, vars)
            return [{"role": "user", "content": content}]

        # Form 2: Dict with system/user keys
        if isinstance(prompt, dict):
            messages = []

            # Check for system/user keys (new format)
            if "system" in prompt or "user" in prompt:
                if "system" in prompt:
                    content = self._format_value(prompt["system"], vars)
                    messages.append({"role": "system", "content": content})
                if "user" in prompt:
                    content = self._format_value(prompt["user"], vars)
                    messages.append({"role": "user", "content": content})
                return messages

            # Otherwise treat as a single message dict
            formatted = self._format_value(prompt, vars)
            return [formatted]

        # Form 3: List → full messages array
        if isinstance(prompt, list):
            return [self._format_value(msg, vars) for msg in prompt]

        raise TypeError(f"Invalid prompt type: {type(prompt)}. Expected str, dict, or list.")

    def _build_messages_legacy(
        self,
        system_prompt: Optional[str],
        user_prompt: Optional[str],
        messages_template: Optional[List[Dict]],
        vars: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Build messages from legacy format (backward compatibility)."""
        messages = []

        # messages_template takes precedence
        if messages_template:
            for msg in messages_template:
                formatted = self._format_value(msg, vars)
                messages.append(formatted)
            return messages

        # Build from system_prompt + user_prompt
        if system_prompt:
            content = self._format_value(system_prompt, vars)
            messages.append({"role": "system", "content": content})

        if user_prompt:
            content = self._format_value(user_prompt, vars)
            messages.append({"role": "user", "content": content})

        return messages

    async def _format(self, **kwargs) -> Dict[str, Any]:
        """Build messages from templates and context.

        Pops reserved keys, uses remaining kwargs as template variables.
        """
        # Pop reserved keys
        prompt = kwargs.pop('prompt', None)
        conversation_history = kwargs.pop('conversation_history', None) or []
        tool_results = kwargs.pop('tool_results', None) or []

        # Legacy keys
        system_prompt = kwargs.pop('system_prompt', None)
        user_prompt = kwargs.pop('user_prompt', None)
        messages_template = kwargs.pop('messages_template', None)

        # Remaining kwargs are template variables
        vars = kwargs

        # Build base messages - prefer new `prompt` format over legacy
        if prompt is not None:
            messages = self._prompt_to_messages(prompt, vars)
        elif system_prompt or user_prompt or messages_template:
            # Legacy format
            messages = self._build_messages_legacy(
                system_prompt, user_prompt, messages_template, vars
            )
        else:
            messages = []

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

        # Check for prompt input
        if 'prompt' in self.inputs:
            val = self.inputs['prompt'].value
            if val is not None:
                metadata["prompt"] = val

        # Legacy metadata
        if 'system_prompt' in self.inputs:
            val = self.inputs['system_prompt'].value
            if isinstance(val, str):
                metadata["system_prompt"] = val

        if 'user_prompt' in self.inputs:
            val = self.inputs['user_prompt'].value
            if isinstance(val, str):
                metadata["user_prompt"] = val

        if 'messages_template' in self.inputs:
            val = self.inputs['messages_template'].value
            if isinstance(val, list):
                metadata["messages_template"] = val

        return metadata
