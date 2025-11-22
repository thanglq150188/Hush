"""Node configuration types and classes for hush-core."""

from typing import (
    Optional,
    Dict,
    List,
    Union,
    Any,
    Literal
)
from pydantic import BaseModel, Field, model_validator
import uuid


NodeType = Literal[
    "data",
    "llm", "embedding", "rerank",
    "branch", "for", "while", "stream",
    "code", "lambda", "parser", "prompt", "doc-processor",
    "milvus", "mongo", "s3",
    "graph",
    "default",
    "dummy",
    "tool-executor",
    "mcp"
]


class Reference(BaseModel):
    """Reference to another node's output or a literal value."""
    id: Optional[str] = None
    field: Optional[str] = None
    value: Optional[Any] = None

    @model_validator(mode="before")
    def validate_reference(cls, values):
        value = values.get('value')
        id = values.get('id')
        field = values.get('field')

        if value is not None and (id is not None or field is not None):
            raise ValueError("Cannot specify both 'value' and ('id', 'field')")

        if value is None and (id is None or field is None):
            raise ValueError("Must specify either 'value' or both 'id' and 'field'")

        return values


class NodeConfig(BaseModel):
    """Configuration for nodes in the workflow."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: NodeType = "default"

    contain_generation: bool = False
    name: Optional[str] = None

    description: str = ""

    status_message: str = ""

    # Ports and connections
    inputs: Dict[str, Any] = {}
    outputs: Dict[str, Any] = {}

    # Node connections
    sources: List[str] = []
    targets: List[str] = []

    # Port definitions with type information
    input_schema: Any = None
    output_schema: Any = None

    stream: bool = False

    start: bool = False
    end: bool = False

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="before")
    def convert_node_instances_to_names(cls, values):
        """Convert Node instances in inputs to their string names."""
        if "inputs" in values and isinstance(values["inputs"], dict):
            converted_inputs = {}
            for key, value in values["inputs"].items():
                converted_inputs[key] = cls._convert_value(value)
            values["inputs"] = converted_inputs
        return values

    @classmethod
    def _convert_value(cls, value):
        """Convert a single value, handling Node instances recursively."""
        if cls._is_node_instance(value):
            return value.name

        elif isinstance(value, dict):
            converted_dict = {}
            for k, v in value.items():
                if cls._is_node_instance(k):
                    converted_dict[k.name] = v
                else:
                    converted_dict[k] = v
            return converted_dict

        else:
            return value

    @classmethod
    def _is_node_instance(cls, obj):
        """Check if an object is a Node instance using duck typing."""
        return (
            hasattr(obj, 'name') and
            hasattr(obj, 'type') and
            hasattr(obj, 'id') and
            isinstance(getattr(obj, 'name'), str)
        )
