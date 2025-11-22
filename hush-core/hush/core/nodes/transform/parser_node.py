"""Parser node for extracting structured data from text."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Literal
import re
import json
import xml.etree.ElementTree as ET

from hush.core.nodes.base import BaseNode
from hush.core.configs.node_config import NodeType
from hush.core.schema import ParamSet


ParserType = Literal["json", "xml", "yaml", "regex", "key_value"]


@dataclass
class ExtractField:
    """Represents a field to extract with its path and type information."""
    output_key: str
    chain_path: List[str]
    type_hint: str

    @classmethod
    def from_string(cls, schema_str: str) -> 'ExtractField':
        """Parse schema string like 'company.user.address: dict' into ExtractField."""
        if ":" not in schema_str:
            schema_str += ": Any"

        chain_text, type_hint = schema_str.split(":", 1)
        chain_path = chain_text.strip().split(".")
        output_key = chain_path[-1]

        return cls(
            output_key=output_key,
            chain_path=chain_path,
            type_hint=type_hint.strip()
        )


def parse_json(text: str) -> Dict[str, Any]:
    """Parse JSON text."""
    # Clean markdown code blocks
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    return json.loads(text)


def parse_xml(text: str) -> Dict[str, Any]:
    """Parse XML text to dictionary."""
    def xml_to_dict(element):
        result = {}
        for child in element:
            if len(child) == 0:
                result[child.tag] = child.text
            else:
                child_dict = xml_to_dict(child)
                if child.tag in result:
                    if not isinstance(result[child.tag], list):
                        result[child.tag] = [result[child.tag]]
                    result[child.tag].append(child_dict)
                else:
                    result[child.tag] = child_dict
        return result

    # Clean markdown code blocks
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    root = ET.fromstring(text)
    return {root.tag: xml_to_dict(root)} if len(root) > 0 else {root.tag: root.text}


def parse_yaml(text: str) -> Dict[str, Any]:
    """Parse YAML text (basic implementation)."""
    try:
        import yaml
        # Clean markdown code blocks
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        return yaml.safe_load(text)
    except ImportError:
        raise ImportError("pyyaml is required for YAML parsing. Install with: pip install pyyaml")


class ParserNode(BaseNode):
    """Node for parsing text into structured data using various parsers."""

    type: NodeType = "parser"

    __slots__ = [
        'backend',
        'format',
        'separator',
        'template',
        'pattern',
        'extract_schema',
        'extract_fields'
    ]

    input_schema: ParamSet = (
        ParamSet.new()
            .var("text: str", required=True)
            .build()
    )

    def __init__(
        self,
        format: Optional[ParserType] = None,
        extract_schema: Optional[List[str]] = None,
        separator: Optional[str] = None,
        template: Optional[str] = None,
        pattern: Optional[str] = None,
        **kwargs
    ):
        """Initialize ParserNode.

        Args:
            format: Parser format type (json, xml, yaml, regex, key_value)
            extract_schema: List of output variable definitions
            separator: Separator for KEY_VALUE parser
            template: Template for STRUCTURED parser
            pattern: Pattern for REGEX parser
            **kwargs: Additional keyword arguments for NodeConfig
        """

        super().__init__(**kwargs)

        self.format = format or "xml"
        self.separator = separator
        self.template = template
        self.pattern = pattern
        self.extract_schema = extract_schema

        if not self.extract_schema:
            raise TypeError(f"{self.name}'s extract_schema has not been provided")

        # Parse schema into structured format
        self.extract_fields = [
            ExtractField.from_string(schema_str)
            for schema_str in self.extract_schema
        ]

        # Build output schema from extract_fields
        output_schema = ParamSet.new()
        for field in self.extract_fields:
            output_schema = output_schema.var(f"{field.output_key}: {field.type_hint}")
        self.output_schema = output_schema.build()

        # Set up parser backend
        self.backend = self._create_parser()

        self.core = self._process

    def _create_parser(self):
        """Create parser function based on format."""
        if self.format == "json":
            return parse_json
        elif self.format == "xml":
            return parse_xml
        elif self.format == "yaml":
            return parse_yaml
        elif self.format == "regex":
            if not self.pattern:
                raise ValueError("Pattern required for regex parser")

            def regex_parser(text: str) -> Dict[str, Any]:
                match = re.search(self.pattern, text)
                if match:
                    return match.groupdict()
                return {}

            return regex_parser
        elif self.format == "key_value":
            sep = self.separator or "="

            def kv_parser(text: str) -> Dict[str, Any]:
                result = {}
                for line in text.split("\n"):
                    if sep in line:
                        key, value = line.split(sep, 1)
                        result[key.strip()] = value.strip()
                return result

            return kv_parser
        else:
            return parse_xml

    def _extract_value_by_path(self, data: Dict[str, Any], chain_path: List[str]) -> Any:
        """Extract value from nested dictionary using chain path."""
        current = data
        for key in chain_path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    async def _process(self, text: str) -> Dict[str, Any]:
        """Parse text and return structured data according to extract_fields."""

        if not text:
            return {}

        # Parse the text using the backend parser
        parsed_data = self.backend(text)

        # Extract values according to extract_fields
        result = {}
        for field in self.extract_fields:
            extracted_value = self._extract_value_by_path(parsed_data, field.chain_path)
            result[field.output_key] = extracted_value

        return result

    def specific_metadata(self) -> Dict[str, Any]:
        """Return subclass-specific metadata dictionary."""
        return {
            "format": self.format,
            "separator": self.separator,
            "template": self.template,
            "pattern": self.pattern
        }
