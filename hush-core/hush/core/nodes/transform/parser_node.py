"""Node parser để trích xuất dữ liệu có cấu trúc từ text."""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Literal
import re
import json
import xml.etree.ElementTree as ET

from hush.core.nodes.base import BaseNode
from hush.core.configs.node_config import NodeType
from hush.core.utils.common import Param


ParserType = Literal["json", "xml", "yaml", "regex", "key_value"]


@dataclass
class ExtractField:
    """Biểu diễn một field cần trích xuất với path và thông tin type."""
    output_key: str
    chain_path: List[str]
    type_hint: str

    @classmethod
    def from_string(cls, schema_str: str) -> 'ExtractField':
        """Parse chuỗi schema như 'company.user.address: dict' thành ExtractField."""
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
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
    return json.loads(text)


def parse_xml(text: str) -> Dict[str, Any]:
    """Parse XML text thành dictionary.

    Handles both single-root and multiple top-level elements.
    For multiple elements like <a>1</a><b>2</b>, wraps in <root> and flattens.
    """
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

    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    # Try parsing as-is first
    try:
        root = ET.fromstring(text)
        return {root.tag: xml_to_dict(root)} if len(root) > 0 else {root.tag: root.text}
    except ET.ParseError:
        # Multiple root elements - wrap in <root> and flatten result
        wrapped = f"<root>{text}</root>"
        root = ET.fromstring(wrapped)
        return xml_to_dict(root)


def parse_yaml(text: str) -> Dict[str, Any]:
    """Parse YAML text."""
    try:
        import yaml
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        return yaml.safe_load(text)
    except ImportError:
        raise ImportError("pyyaml là bắt buộc để parse YAML")


# Module-level parser lookup for O(1) format selection
_PARSER_MAP = {
    "json": parse_json,
    "xml": parse_xml,
    "yaml": parse_yaml,
}


class ParserNode(BaseNode):
    """Node parse text thành dữ liệu có cấu trúc.

    Hỗ trợ nhiều format: JSON, XML, YAML, regex, key-value.
    Trích xuất các field theo chain path (ví dụ: 'user.address.city').
    """

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

    def __init__(
        self,
        format: Optional[ParserType] = None,
        extract_schema: Optional[List[str]] = None,
        separator: Optional[str] = None,
        template: Optional[str] = None,
        pattern: Optional[str] = None,
        inputs: Dict[str, Any] = None,
        outputs: Dict[str, Any] = None,
        **kwargs
    ):
        if not extract_schema:
            raise TypeError("extract_schema là bắt buộc")

        # Parse schema thành format có cấu trúc
        extract_fields = [
            ExtractField.from_string(schema_str)
            for schema_str in extract_schema
        ]

        # Parse inputs/outputs từ extract_schema
        parsed_inputs = {"text": Param(type=str, required=True)}
        parsed_outputs = {field.output_key: Param() for field in extract_fields}

        # Gọi super().__init__ không truyền inputs/outputs
        super().__init__(**kwargs)

        # Merge parsed với user-provided
        self.inputs = self._merge_params(parsed_inputs, inputs)
        self.outputs = self._merge_params(parsed_outputs, outputs)

        self.format = format or "xml"
        self.separator = separator
        self.template = template
        self.pattern = pattern
        self.extract_schema = extract_schema
        self.extract_fields = extract_fields

        self.backend = self._create_parser()
        self.core = self._process

    def _create_parser(self):
        """Tạo parser function dựa trên format."""
        # O(1) lookup for common formats
        parser = _PARSER_MAP.get(self.format)
        if parser is not None:
            return parser

        # Special cases that need instance attributes
        if self.format == "regex":
            if not self.pattern:
                raise ValueError("Pattern là bắt buộc cho regex parser")

            def regex_parser(text: str) -> Dict[str, Any]:
                match = re.search(self.pattern, text)
                return match.groupdict() if match else {}

            return regex_parser

        if self.format == "key_value":
            sep = self.separator or "="

            def kv_parser(text: str) -> Dict[str, Any]:
                result = {}
                for line in text.split("\n"):
                    if sep in line:
                        key, value = line.split(sep, 1)
                        result[key.strip()] = value.strip()
                return result

            return kv_parser

        # Default fallback
        return parse_xml

    def _extract_value_by_path(self, data: Dict[str, Any], chain_path: List[str]) -> Any:
        """Trích xuất giá trị từ nested dictionary theo chain path."""
        current = data
        for key in chain_path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    async def _process(self, text: str) -> Dict[str, Any]:
        """Parse text và trích xuất các field."""
        if not text:
            return {}

        parsed_data = self.backend(text)

        result = {}
        for field in self.extract_fields:
            result[field.output_key] = self._extract_value_by_path(parsed_data, field.chain_path)

        return result

    def specific_metadata(self) -> Dict[str, Any]:
        """Trả về metadata riêng của subclass."""
        return {
            "format": self.format,
            "separator": self.separator,
            "template": self.template,
            "pattern": self.pattern
        }
