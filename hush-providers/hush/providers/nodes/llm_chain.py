"""LLM Chain Node for hush-providers.

This module provides LLMChainNode - a composite node that combines
PromptNode, LLMNode, and optionally ParserNode into a single reusable chain.
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

    Example:
        ```python
        from hush.core import WorkflowEngine, START, END, INPUT, OUTPUT
        from hush.providers.nodes import LLMChainNode

        # Simple text generation - prompts passed via inputs
        with WorkflowEngine(name="chat") as workflow:
            chain = LLMChainNode(
                name="summarizer",
                resource_key="gpt-4",
                inputs={
                    "system_prompt": "You are a helpful summarization assistant.",
                    "user_prompt": "Summarize: {text}",
                    "text": INPUT,
                    "*": PARENT
                },
                outputs={"content": OUTPUT}
            )
            START >> chain >> END

        # Structured output generation
        with WorkflowEngine(name="classifier") as workflow:
            chain = LLMChainNode(
                name="classifier",
                resource_key="gpt-4",
                extract_schema=["category: str", "confidence: float"],
                parser="xml",
                inputs={
                    "user_prompt": "Classify: {text}\\n<category>...</category>",
                    "text": INPUT,
                    "*": PARENT
                },
                outputs=OUTPUT
            )
            START >> chain >> END
        ```
    """

    __slots__ = [
        'resource_key',
        'extract_schema',
        'parser',
        'enable_thinking'
    ]

    type: NodeType = "graph"

    def __init__(
        self,
        resource_key: Optional[str] = None,
        extract_schema: Optional[List[str]] = None,
        parser: str = "xml",
        enable_thinking: bool = False,
        **kwargs
    ):
        """Initialize LLMChainNode.

        Args:
            resource_key: Resource key for LLM in ResourceHub (e.g., "gpt-4")
            extract_schema: List of output variables for structured parsing
                           (e.g., ["category: str", "confidence: float"])
            parser: Parser format for structured output ("xml", "json", "yaml")
            enable_thinking: Whether to enable thinking mode in the LLM
            **kwargs: Additional keyword arguments for GraphNode
                     - inputs: Should include system_prompt, user_prompt, or
                       messages_template along with {"*": PARENT} for forwarding
        """
        super().__init__(**kwargs)

        self.resource_key = resource_key
        self.extract_schema = extract_schema
        self.parser = parser
        self.enable_thinking = enable_thinking
        self.contain_generation = True

        self._build_graph()

    def _build_graph(self):
        """Build the internal processing graph based on configuration."""
        with self:
            # Step 1: Create prompt formatting node - forwards all inputs from parent
            _prompt = PromptNode(
                name="prompt",
                inputs={"*": PARENT}
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
                    outputs={"*": PARENT}
                )

                START >> _prompt >> _llm >> _parser >> END

            else:
                # Mode 2: Simple text generation pipeline (Prompt -> LLM)
                _llm = LLMNode(
                    name="llm",
                    resource_key=self.resource_key,
                    inputs={"messages": _prompt["messages"]},
                    outputs={"*": PARENT},
                    stream=getattr(self, 'stream', False)
                )

                START >> _prompt >> _llm >> END

        # Build the internal graph
        self.build()

    def specific_metadata(self) -> Dict[str, Any]:
        """Return LLMChainNode-specific metadata."""
        metadata = {
            "resource_key": self.resource_key,
        }

        if self.extract_schema:
            metadata["extract_schema"] = self.extract_schema
            metadata["parser"] = self.parser

        return {k: v for k, v in metadata.items() if v is not None}
