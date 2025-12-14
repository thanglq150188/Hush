"""Parser node for extracting structured data from text."""

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

    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text

    root = ET.fromstring(text)
    return {root.tag: xml_to_dict(root)} if len(root) > 0 else {root.tag: root.text}


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
        raise ImportError("pyyaml is required for YAML parsing")


class ParserNode(BaseNode):
    """Node for parsing text into structured data."""

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
        **kwargs
    ):
        if not extract_schema:
            raise TypeError("extract_schema is required")

        # Parse schema into structured format
        extract_fields = [
            ExtractField.from_string(schema_str)
            for schema_str in extract_schema
        ]

        # Build schemas
        input_schema = {"text": Param(type=str, required=True)}
        output_schema = {field.output_key: Param(type=Any) for field in extract_fields}

        super().__init__(
            input_schema=input_schema,
            output_schema=output_schema,
            **kwargs
        )

        self.format = format or "xml"
        self.separator = separator
        self.template = template
        self.pattern = pattern
        self.extract_schema = extract_schema
        self.extract_fields = extract_fields

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
                return match.groupdict() if match else {}

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
        """Parse text and extract fields."""
        if not text:
            return {}

        parsed_data = self.backend(text)

        result = {}
        for field in self.extract_fields:
            result[field.output_key] = self._extract_value_by_path(parsed_data, field.chain_path)

        return result

    def specific_metadata(self) -> Dict[str, Any]:
        """Return subclass-specific metadata."""
        return {
            "format": self.format,
            "separator": self.separator,
            "template": self.template,
            "pattern": self.pattern
        }


if __name__ == "__main__":
    import asyncio
    from hush.core.states import StateSchema

    async def main():
        schema = StateSchema("test")
        state = schema.create_state()

        # Test 1: JSON parser
        print("=" * 50)
        print("Test 1: JSON Parser")
        print("=" * 50)

        json_parser = ParserNode(
            name="json_parser",
            format="json",
            extract_schema=[
                "user.name",
                "user.age",
                "status",
            ],
            inputs={"text": '{"user": {"name": "John", "age": 30}, "status": "active"}'}
        )

        print(f"Input schema: {json_parser.input_schema}")
        print(f"Output schema: {json_parser.output_schema}")

        result = await json_parser.run(state)
        print(f"Result: {result}")

        # Test 2: XML parser
        print("\n" + "=" * 50)
        print("Test 2: XML Parser")
        print("=" * 50)

        xml_text = """
        <response>
            <user>
                <name>Alice</name>
                <email>alice@example.com</email>
            </user>
            <code>200</code>
        </response>
        """

        xml_parser = ParserNode(
            name="xml_parser",
            format="xml",
            extract_schema=[
                "response.user.name",
                "response.user.email",
                "response.code",
            ],
            inputs={"text": xml_text}
        )

        result2 = await xml_parser.run(state)
        print(f"Result: {result2}")

        # Test 3: Key-Value parser
        print("\n" + "=" * 50)
        print("Test 3: Key-Value Parser")
        print("=" * 50)

        kv_text = """
        name=Bob
        age=25
        city=New York
        """

        kv_parser = ParserNode(
            name="kv_parser",
            format="key_value",
            separator="=",
            extract_schema=["name", "age", "city"],
            inputs={"text": kv_text}
        )

        result3 = await kv_parser.run(state)
        print(f"Result: {result3}")

        # Test 4: Regex parser
        print("\n" + "=" * 50)
        print("Test 4: Regex Parser")
        print("=" * 50)

        regex_text = "User: john_doe, Email: john@example.com, Score: 95"

        regex_parser = ParserNode(
            name="regex_parser",
            format="regex",
            pattern=r"User: (?P<username>\w+), Email: (?P<email>[\w@.]+), Score: (?P<score>\d+)",
            extract_schema=["username", "email", "score"],
            inputs={"text": regex_text}
        )

        result4 = await regex_parser.run(state)
        print(f"Result: {result4}")

        # Test 5: Quick test using __call__
        print("\n" + "=" * 50)
        print("Test 5: Quick test using __call__")
        print("=" * 50)

        quick_parser = ParserNode(
            name="quick_json",
            format="json",
            extract_schema=["name", "age"]
        )
        result5 = quick_parser(text='{"name": "Bob", "age": 25}')
        print(f"quick_parser(text=...) = {result5}")

        print("\n" + "=" * 50)
        print("All tests passed!")
        print("=" * 50)

    asyncio.run(main())
