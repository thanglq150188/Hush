"""LLM Node for hush-providers.

This module provides LLMNode that uses ResourceHub to access LLM resources.
Follows hush-core design patterns with Param-based schema.
"""

import asyncio
from datetime import datetime
from time import perf_counter
from typing import Dict, Any, Optional, TYPE_CHECKING

from hush.core.nodes import BaseNode
from hush.core.configs import NodeType
from hush.core.utils.common import Param
from hush.core.registry import ResourceHub, get_hub
from hush.core import STREAM_SERVICE, LOGGER

if TYPE_CHECKING:
    from hush.core.states import MemoryState


class LLMNode(BaseNode):
    """LLM node for executing language model operations in workflows.

    Uses ResourceHub to access LLM resources by resource_key.
    Supports streaming and instant response modes.

    Example:
        ```python
        from hush.core import GraphNode, START, END, PARENT
        from hush.providers import LLMNode

        with GraphNode(name="chat") as workflow:
            llm = LLMNode(
                name="chat",
                resource_key="gpt-4",
                inputs={"messages": PARENT["messages"]},
                outputs=PARENT
            )
            START >> llm >> END

        workflow.build()
        ```
    """

    __slots__ = ['resource_key', 'instant_response', '_llm']

    type: NodeType = "llm"

    def __init__(
        self,
        resource_key: Optional[str] = None,
        instant_response: Optional[bool] = None,
        inputs: Dict[str, Any] = None,
        outputs: Dict[str, Any] = None,
        **kwargs
    ):
        """Initialize LLMNode.

        Args:
            resource_key: Resource key for LLM in ResourceHub (e.g., "gpt-4")
            instant_response: Whether to return instant response without calling LLM
            inputs: Input variable mappings
            outputs: Output variable mappings
            **kwargs: Additional keyword arguments for BaseNode
        """
        # Initialize base without inputs/outputs first
        super().__init__(**kwargs)

        self.resource_key = resource_key
        self.instant_response = instant_response
        self.contain_generation = True

        # Define input/output schema
        input_schema = {
            "messages": Param(type=list, required=True),
            "temperature": Param(type=float, default=0.0),
            "max_tokens": Param(type=int, default=None),
            "stream_options": Param(type=dict, default={"include_usage": True}),
        }

        output_schema = {
            "role": Param(type=str, default="assistant"),
            "content": Param(type=str, required=True),
            "finish_reason": Param(type=str, default=None),
            "model_used": Param(type=str, required=True),
            "tokens_used": Param(type=dict, default={}),
            "tool_calls": Param(type=list, default=[]),
            "thinking_content": Param(type=str, default=None),
            "context_used": Param(type=int, default=0),
            "error_code": Param(type=int, default=None),
            "error_message": Param(type=str, default=None),
        }

        # Merge with user-provided
        self.inputs = self._merge_params(input_schema, inputs)
        self.outputs = self._merge_params(output_schema, outputs)

        # Set up LLM backend
        if not self.instant_response:
            try:
                hub = ResourceHub.instance()
            except RuntimeError:
                hub = get_hub()

            self._llm = hub.llm(self.resource_key)

            if self.stream:
                self.core = self._llm.stream
            else:
                self.core = self._llm.generate
        else:
            self._llm = None

    async def run(
        self,
        state: 'MemoryState',
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the LLM node with streaming support via STREAM_SERVICE.

        Handles streaming internally with STREAM_SERVICE.push() for each chunk.
        """
        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()
        perf_start = perf_counter()
        _inputs = {}
        _outputs = {}

        response = ""
        thinking_content = ""
        finish_reason = "stop"
        tokens_used = {}
        tool_calls = []

        try:
            _inputs = self.get_inputs(state, context_id=context_id)

            if self.instant_response:
                # Instant response mode
                response = _inputs.get("response", "This is a default message")
                _outputs = {
                    "role": "assistant",
                    "content": response,
                    "model_used": self.resource_key or "instant",
                    "finish_reason": "stop",
                    "tokens_used": {"prompt": 0, "completion": 0, "total": 0},
                }

            elif self.stream:
                # Streaming mode with STREAM_SERVICE
                LOGGER.info(f"Streaming mode for {self.name}...")
                channel_name = self.identity(context_id)

                async for chunk in self.core(**_inputs):
                    # Extract usage info
                    if chunk.usage:
                        tokens_used = chunk.usage.model_dump()

                    if chunk.choices:
                        choice = chunk.choices[0]

                        # Accumulate thinking content
                        if hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content:
                            thinking_content += choice.delta.reasoning_content

                        # Accumulate main content
                        if choice.delta.content:
                            response += choice.delta.content

                        # Capture finish reason
                        if choice.finish_reason:
                            finish_reason = choice.finish_reason

                        # Capture tool calls
                        if choice.delta.tool_calls:
                            tool_calls.extend([tc.model_dump() for tc in choice.delta.tool_calls])

                    # Push chunk to STREAM_SERVICE
                    asyncio.create_task(STREAM_SERVICE.push(
                        request_id=request_id,
                        channel_name=channel_name,
                        data=chunk
                    ))

                # Signal end of stream
                asyncio.create_task(STREAM_SERVICE.end(request_id, channel_name))

                _outputs = {
                    "role": "assistant",
                    "content": response,
                    "finish_reason": finish_reason,
                    "model_used": self.resource_key,
                    "tokens_used": tokens_used,
                    "tool_calls": tool_calls,
                    "thinking_content": thinking_content if thinking_content else None,
                    "context_used": len(str(_inputs.get("messages", []))) // 4,
                }

            else:
                # Non-streaming generate mode
                LOGGER.info(f"Generate mode for {self.name}...")
                completion = await self.core(**_inputs)

                response = completion.choices[0].message.content or ""
                finish_reason = completion.choices[0].finish_reason

                # Extract thinking content
                if hasattr(completion.choices[0].message, 'reasoning_content'):
                    thinking_content = completion.choices[0].message.reasoning_content or ""

                # Extract usage info
                if completion.usage:
                    tokens_used = completion.usage.model_dump()

                # Extract tool calls
                if completion.choices[0].message.tool_calls:
                    tool_calls = [tc.model_dump() for tc in completion.choices[0].message.tool_calls]

                _outputs = {
                    "role": "assistant",
                    "content": response,
                    "finish_reason": finish_reason,
                    "model_used": self.resource_key or completion.model,
                    "tokens_used": tokens_used,
                    "tool_calls": tool_calls,
                    "thinking_content": thinking_content if thinking_content else None,
                    "context_used": len(str(_inputs.get("messages", []))) // 4,
                }

            self.store_result(state, _outputs, context_id)

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            LOGGER.error(f"Error in node {self.name}: {str(e)}")
            LOGGER.error(error_msg)
            state[self.full_name, "error", context_id] = error_msg

        finally:
            perf_end = perf_counter()
            latency_ms = (perf_end - perf_start) * 1000
            self._log(request_id, context_id, _inputs, _outputs, latency_ms)

        return _outputs

    def specific_metadata(self) -> Dict[str, Any]:
        """Return LLM-specific metadata dictionary."""
        return {
            "model": self.resource_key
        }
