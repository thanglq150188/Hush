"""LLM Chain Node for hush-providers.

This module provides LLMChainNode - a composite node that combines
PromptNode, LLMNode, and optionally ParserNode into a single reusable chain.
Migrated from beeflow with hush-core design patterns.
"""

from typing import Dict, Any, Optional, List

from hush.core.nodes import GraphNode, ParserNode, START, END, PARENT
from hush.core.configs import NodeType

from hush.providers.nodes.prompt import PromptNode
from hush.providers.nodes.llm import LLMNode


class LLMChainNode(GraphNode):
    """A composite node combining prompt formatting, LLM generation, and optional parsing.

    This is a fundamental building block for LLM workflows, providing two modes:

    1. **Text Generation Mode** (when extract_schema is empty):
       ```
       Input -> PromptNode -> LLMNode -> Output
       ```
       - Takes input variables and formats them into prompts
       - Generates text using the specified LLM model
       - Returns raw LLM output (content, role, etc.)

    2. **Structured Output Mode** (when extract_schema is provided):
       ```
       Input -> PromptNode -> LLMNode -> ParserNode -> Output
       ```
       - Takes input variables and formats them into prompts
       - Generates text using the specified LLM model
       - Parses the LLM output into structured data based on extract_schema
       - Returns parsed structured data

    Features:
    - Auto-variable detection from prompt templates
    - Flexible prompting with system and user prompts
    - Support for complex message templates (multi-turn, images)
    - Streaming support
    - Parser integration for structured outputs

    Example:
        ```python
        from hush.core import WorkflowEngine, START, END, INPUT, OUTPUT
        from hush.providers.nodes import LLMChainNode

        # Simple text generation
        with WorkflowEngine(name="chat") as workflow:
            chain = LLMChainNode(
                name="summarizer",
                resource_key="gpt-4",
                system_prompt="You are a helpful summarization assistant.",
                user_prompt="Summarize: {text}",
                inputs={"text": INPUT},
                outputs={"content": OUTPUT}
            )
            START >> chain >> END

        # Structured output generation
        with WorkflowEngine(name="classifier") as workflow:
            chain = LLMChainNode(
                name="classifier",
                resource_key="gpt-4",
                user_prompt="Classify this text: {text}\\n\\nOutput XML: <category>...</category><confidence>...</confidence>",
                extract_schema=["category: str", "confidence: float"],
                parser="xml",
                inputs={"text": INPUT},
                outputs=OUTPUT
            )
            START >> chain >> END
        ```
    """

    __slots__ = [
        'resource_key',
        'user_prompt',
        'system_prompt',
        'messages_template',
        'extract_schema',
        'parser',
        'enable_thinking'
    ]

    type: NodeType = "graph"

    def __init__(
        self,
        resource_key: Optional[str] = None,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        messages_template: Optional[List[Dict[str, Any]]] = None,
        extract_schema: Optional[List[str]] = None,
        parser: str = "xml",
        enable_thinking: bool = False,
        **kwargs
    ):
        """Initialize LLMChainNode.

        Args:
            resource_key: Resource key for LLM in ResourceHub (e.g., "gpt-4")
            user_prompt: User prompt template with {variable} placeholders
            system_prompt: System prompt template with {variable} placeholders
            messages_template: Complex message template for multi-turn conversations
            extract_schema: List of output variables for structured parsing
                           (e.g., ["category: str", "confidence: float"])
            parser: Parser format for structured output ("xml", "json", "yaml")
            enable_thinking: Whether to enable thinking mode in the LLM
            **kwargs: Additional keyword arguments for GraphNode
        """
        super().__init__(**kwargs)

        self.resource_key = resource_key
        self.user_prompt = user_prompt
        self.system_prompt = system_prompt
        self.messages_template = messages_template
        self.extract_schema = extract_schema
        self.parser = parser
        self.enable_thinking = enable_thinking
        self.contain_generation = True

        self._build_graph()

    def _build_graph(self):
        """Build the internal processing graph based on configuration."""
        with self:
            # Step 1: Create prompt formatting node
            _prompt = PromptNode(
                name="prompt",
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                messages_template=self.messages_template,
                inputs=PARENT
            )

            if self.extract_schema:
                # Mode 1: Structured output pipeline (Prompt -> LLM -> Parser)
                _llm = LLMNode(
                    name="llm",
                    resource_key=self.resource_key,
                    inputs={"messages": _prompt["messages"]}
                )

                _parser = ParserNode(
                    name="parser",
                    format=self.parser,
                    extract_schema=self.extract_schema,
                    inputs={"text": _llm["content"]},
                    outputs=PARENT
                )

                START >> _prompt >> _llm >> _parser >> END

            else:
                # Mode 2: Simple text generation pipeline (Prompt -> LLM)
                _llm = LLMNode(
                    name="llm",
                    resource_key=self.resource_key,
                    inputs={"messages": _prompt["messages"]},
                    outputs=PARENT,
                    stream=getattr(self, 'stream', False)
                )

                START >> _prompt >> _llm >> END

        # Build the internal graph
        self.build()

    def specific_metadata(self) -> Dict[str, Any]:
        """Return LLMChainNode-specific metadata."""
        metadata = {
            "resource_key": self.resource_key,
            "parser": self.parser if self.extract_schema else None
        }

        if self.messages_template:
            metadata["messages_template"] = self.messages_template
        else:
            if self.system_prompt:
                metadata["system_prompt"] = self.system_prompt
            if self.user_prompt:
                metadata["user_prompt"] = self.user_prompt

        if self.extract_schema:
            metadata["extract_schema"] = self.extract_schema

        return {k: v for k, v in metadata.items() if v is not None}
