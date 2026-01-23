"""LLM Chain Node for hush-providers.

This module provides LLMChainNode - a composite node that combines
PromptNode, LLMNode, and optionally ParserNode into a single reusable chain.
"""

from typing import Dict, Any, Optional, List, Union

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

        # Simple text generation with new unified prompt
        with WorkflowEngine(name="chat") as workflow:
            chain = LLMChainNode(
                name="summarizer",
                resource_key="gpt-4",
                inputs={
                    "prompt": {
                        "system": "You are a helpful summarization assistant.",
                        "user": "Summarize: {text}"
                    },
                    "text": INPUT,
                    "*": PARENT
                },
                outputs={"content": OUTPUT}
            )
            START >> chain >> END

        # String prompt (user message only)
        with WorkflowEngine(name="chat") as workflow:
            chain = LLMChainNode(
                name="chat",
                resource_key="gpt-4",
                inputs={
                    "prompt": "Hello {name}, how can I help?",
                    "name": INPUT["name"],
                    "*": PARENT
                },
                outputs={"content": OUTPUT}
            )
            START >> chain >> END

        # Load balanced with fallback
        with WorkflowEngine(name="chat") as workflow:
            chain = LLMChainNode(
                name="chat",
                resource_key=["gpt-4o", "gpt-4o-mini"],
                ratios=[0.7, 0.3],  # 70% gpt-4o, 30% gpt-4o-mini
                fallback=["claude-sonnet"],  # Fallback if all primary fail
                inputs={
                    "prompt": {"system": "You are helpful.", "user": "{query}"},
                    "*": PARENT
                },
                outputs={"content": OUTPUT}
            )
            START >> chain >> END

        # Structured output with JSON mode
        with WorkflowEngine(name="classifier") as workflow:
            chain = LLMChainNode(
                name="classifier",
                resource_key="gpt-4",
                response_format={"type": "json_object"},
                inputs={
                    "prompt": {"user": "Classify and return JSON: {text}"},
                    "*": PARENT
                },
                outputs=OUTPUT
            )
            START >> chain >> END

        # Legacy format (backward compatible)
        with WorkflowEngine(name="chat") as workflow:
            chain = LLMChainNode(
                name="summarizer",
                resource_key="gpt-4",
                inputs={
                    "system_prompt": "You are a helpful assistant.",
                    "user_prompt": "Summarize: {text}",
                    "*": PARENT
                },
                outputs={"content": OUTPUT}
            )
            START >> chain >> END
        ```
    """

    __slots__ = [
        'resource_key',
        'ratios',
        'fallback',
        'extract_schema',
        'parser',
        'response_format',
        'enable_thinking'
    ]

    type: NodeType = "graph"

    def __init__(
        self,
        resource_key: Optional[Union[str, List[str]]] = None,
        ratios: Optional[List[float]] = None,
        fallback: Optional[List[str]] = None,
        extract_schema: Optional[List[str]] = None,
        parser: str = "xml",
        response_format: Optional[Dict[str, Any]] = None,
        enable_thinking: bool = False,
        **kwargs
    ):
        """Initialize LLMChainNode.

        Args:
            resource_key: Resource key(s) for LLM in ResourceHub.
                - Single string: "gpt-4"
                - List for load balancing: ["gpt-4", "claude-3"]
            ratios: Weight ratios for load balancing. Must sum to 1.0.
                Only used when resource_key is a list.
                Example: [0.7, 0.3] for 70%/30% distribution
            fallback: Fallback resource key(s) to use when primary model fails.
                List of resource keys from ResourceHub, tried in order.
            extract_schema: List of output variables for structured parsing
                (e.g., ["category: str", "confidence: float"])
            parser: Parser format for structured output ("xml", "json", "yaml")
            response_format: OpenAI response format for JSON mode.
                - {"type": "json_object"} for JSON mode
                - {"type": "json_schema", "json_schema": {...}} for strict schema
            enable_thinking: Whether to enable thinking mode in the LLM
            **kwargs: Additional keyword arguments for GraphNode
                - inputs: Should include prompt (new) or system_prompt/user_prompt (legacy)
                  along with {"*": PARENT} for forwarding
        """
        super().__init__(**kwargs)

        self.resource_key = resource_key
        self.ratios = ratios
        self.fallback = fallback
        self.extract_schema = extract_schema
        self.parser = parser
        self.response_format = response_format
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

            # Build LLM inputs - always include messages from prompt
            llm_inputs = {"messages": _prompt["messages"]}

            # Pass response_format from parent if configured
            if self.response_format:
                llm_inputs["response_format"] = self.response_format

            if self.extract_schema:
                # Mode 1: Structured output pipeline (Prompt -> LLM -> Parser)
                _llm = LLMNode(
                    name="llm",
                    resource_key=self.resource_key,
                    ratios=self.ratios,
                    fallback=self.fallback,
                    inputs=llm_inputs
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
                    ratios=self.ratios,
                    fallback=self.fallback,
                    inputs=llm_inputs,
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

        # Load balancing info
        if isinstance(self.resource_key, list):
            metadata["load_balancing"] = True
            if self.ratios:
                metadata["ratios"] = self.ratios

        # Fallback info
        if self.fallback:
            metadata["fallback"] = self.fallback

        # Structured output info
        if self.extract_schema:
            metadata["extract_schema"] = self.extract_schema
            metadata["parser"] = self.parser

        # Response format
        if self.response_format:
            metadata["response_format"] = self.response_format

        return {k: v for k, v in metadata.items() if v is not None}
