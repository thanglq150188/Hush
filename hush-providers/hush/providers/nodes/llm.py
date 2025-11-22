"""LLM Node for hush-providers.

This module provides LLMNode that uses ResourceHub to access LLM resources.
It matches the original beeflow implementation using resource_key.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import asyncio

from hush.core import BaseNode, WorkflowState, STREAM_SERVICE, LOGGER
from hush.core.schema import ParamSet
from hush.core.configs import NodeType
from hush.core.utils.common import fake_chunk_from
from hush.core.registry import ResourceHub, get_hub


class LLMNode(BaseNode):
    """LLM node for executing language model operations in workflows.

    Uses ResourceHub to access LLM resources by resource_key.
    Supports streaming, instant response, and comprehensive error handling.

    Example:
        ```python
        from hush.core import WorkflowEngine, START, END, INPUT, OUTPUT, RESOURCE_HUB
        from hush.providers import LLMNode  # Plugin auto-registers!
        from hush.providers.llms.config import OpenAIConfig

        # Register LLM config (optional - can also use resources.yaml)
        config = OpenAIConfig(api_type="openai", api_key="...", model="gpt-4")
        RESOURCE_HUB.register(config, persist=False)

        # Create workflow
        with WorkflowEngine(name="chat") as workflow:
            llm = LLMNode(
                name="chat",
                resource_key="openai:gpt-4",  # Uses global RESOURCE_HUB
                inputs={"messages": INPUT},
                outputs={"content": OUTPUT},
                stream=True
            )
            START >> llm >> END

        workflow.compile()
        result = await workflow.run(inputs={"messages": [{"role": "user", "content": "Hello!"}]})
        ```
    """

    __slots__ = ['resource_key', 'instant_response']

    type: NodeType = "llm"

    input_schema: ParamSet = (
        ParamSet.new()
            # Core parameters
            .var("messages: List", required=True)

            # Generation control
            .var("temperature: float = 0.0")
            .var("max_tokens: Optional[int] = None")
            .var("stream_options: Dict[str, Any] = {'include_usage': True}")
            .build()
    )

    output_schema: ParamSet = (
        ParamSet.new()
            # Basic response
            .var("role: str = 'assistant'")
            .var("content: str", required=True)

            # Generation metadata
            .var("finish_reason: Optional[str] = None")
            .var("model_used: str", required=True)
            .var("tokens_used: Dict[str, int] = {}")

            # Tool usage (when available)
            .var("tool_calls: List[Dict] = []")

            # Thinking process (when enabled)
            .var("thinking_content: Optional[str] = None")

            # Context monitoring
            .var("context_used: int = 0")

            # Error handling
            .var("error_code: Optional[int] = None")
            .var("error_message: Optional[str] = None")

            # Custom metadata
            .var("custom_data: Dict[str, Any] = {}")
            .build()
    )

    def __init__(
        self,
        resource_key: Optional[str] = None,
        instant_response: Optional[bool] = None,
        **kwargs
    ):
        """Initialize LLMNode.

        Args:
            resource_key: Resource key for LLM in ResourceHub (e.g., "gpt-4")
            instant_response: Whether to return instant response without calling LLM
            **kwargs: Additional keyword arguments for BaseNode
        """
        super().__init__(**kwargs)

        self.resource_key = resource_key
        self.instant_response = instant_response
        self.contain_generation = True

        if not self.instant_response:
            # Try to get hub (prefers singleton for backwards compatibility, then global)
            try:
                hub = ResourceHub.instance()
            except RuntimeError:
                # Fall back to global hub if no singleton set
                hub = get_hub()

            llm = hub.llm(self.resource_key)

            if self.stream:
                self.core = llm.stream
            else:
                self.core = llm.generate

    async def run(
        self,
        state: WorkflowState,
        context_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the core logic of LLM node with the given state and inputs."""

        parent_name = self.father.full_name if self.father else None
        state.record_execution(self.full_name, parent_name, context_id)

        request_id = state.request_id
        start_time = datetime.now()

        response = ""
        thinking_content = ""
        finish_reason = "stop"
        tokens_used = {"prompt": 0, "completion": 0, "total": 0}
        tool_calls = []
        thinking_content = None
        context_used = 0

        try:
            _inputs = self.get_inputs(state, context_id=context_id)

            # Log truncated inputs for debugging
            LOGGER.info("request[%s] - running NODE: %s[%s] (%s), inputs = %s",
                    request_id, self.name, context_id, str(self.type).upper(), str(_inputs)[:200])

            if self.instant_response:
                response = _inputs.get("response", "This is a default message")
                if self.stream:
                    asyncio.create_task(STREAM_SERVICE.push(
                        request_id=request_id,
                        channel_name=self.identity(context_id),
                        data=fake_chunk_from(content=response, model=self.resource_key, chat_id=self.id)
                    ))

            else:
                if self.stream:
                    LOGGER.info("Streaming mode...")
                    first_chunk = True
                    finish_reason = None
                    tokens_used = {}
                    tool_calls = []

                    async for chunk in self.core(**_inputs):
                        if first_chunk:
                            state.set_by_index(
                                index=self.metrics["completion_start_time"],
                                value=datetime.now(),
                                context_id=context_id
                            )
                            first_chunk = False

                        # Extract data from chunk
                        if chunk.usage:
                            tokens_used = chunk.usage.model_dump()
                            state.set_by_index(
                                index=self.metrics["usage"],
                                value=tokens_used,
                                context_id=context_id
                            )

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

                        asyncio.create_task(STREAM_SERVICE.push(
                            request_id=request_id,
                            channel_name=self.identity(context_id),
                            data=chunk
                        ))

                else:
                    LOGGER.info("Generate mode...")
                    completion = await self.core(**_inputs)
                    response = completion.choices[0].message.content or ""
                    finish_reason = completion.choices[0].finish_reason

                    # Extract thinking content from non-streaming response
                    if hasattr(completion.choices[0].message, 'reasoning_content'):
                        thinking_content = completion.choices[0].message.reasoning_content or ""

                    # Extract usage info
                    if completion.usage:
                        tokens_used = completion.usage.model_dump()
                    else:
                        tokens_used = {"prompt": 0, "completion": 0, "total": 0}

                    state.set_by_index(
                        index=self.metrics["usage"],
                        value=tokens_used,
                        context_id=context_id
                    )

                    # Extract tool calls
                    if completion.choices[0].message.tool_calls:
                        tool_calls = [tc.model_dump() for tc in completion.choices[0].message.tool_calls]
                    else:
                        tool_calls = []

            # Calculate context used (estimate based on input)
            context_used = len(str(_inputs.get("messages", []))) // 4  # Rough token estimate

        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            # Handle errors
            state.set_by_index(self.output_indexes["error_code"], 500, context_id)
            state.set_by_index(self.output_indexes["error_message"], error_msg, context_id)
            state[self.full_name, "error", context_id] = error_msg

            LOGGER.error(f"LLM execution failed: {e}")

            default_answers = 'Sorry, an error occurred while processing your request. Please try again later.'

            if self.stream:
                asyncio.create_task(STREAM_SERVICE.push(
                    request_id=request_id,
                    channel_name=self.identity(context_id),
                    data=fake_chunk_from(
                        content=default_answers,
                        model="error"
                    )
                ))

        finally:
            # Always record timing metrics
            await asyncio.sleep(0.000001)

            end_time = datetime.now()
            state.set_by_index(self.metrics['start_time'], start_time, context_id=context_id)
            state.set_by_index(self.metrics['end_time'], end_time, context_id=context_id)

            # Store all outputs
            state.set_by_index(self.output_indexes["content"], response, context_id)
            state.set_by_index(self.output_indexes["role"], "assistant", context_id)
            state.set_by_index(self.output_indexes["model_used"], self.resource_key, context_id)
            if finish_reason:
                state.set_by_index(self.output_indexes["finish_reason"], finish_reason, context_id)
            if tokens_used:
                state.set_by_index(self.output_indexes["tokens_used"], tokens_used, context_id)
            if tool_calls:
                state.set_by_index(self.output_indexes["tool_calls"], tool_calls, context_id)
            if thinking_content:
                state.set_by_index(self.output_indexes["thinking_content"], thinking_content if thinking_content else None, context_id)
            if context_used:
                state.set_by_index(self.output_indexes["context_used"], context_used, context_id)

            if self.stream:
                asyncio.create_task(STREAM_SERVICE.end(request_id, self.identity(context_id)))

            _outputs = self.get_result(state)

            LOGGER.info("request[%s] - running NODE: %s[%s] (%s), outputs = %s",
                    request_id, self.name, context_id, str(self.type).upper(), str(_outputs)[:200])

            return _outputs

    def specific_metadata(self) -> Dict[str, Any]:
        """Return LLM-specific metadata dictionary."""
        return {
            "model": self.resource_key
        }
