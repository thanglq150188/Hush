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
    """Parse XML text thành dictionary."""
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
        raise ImportError("pyyaml là bắt buộc để parse YAML")


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
        **kwargs
    ):
        if not extract_schema:
            raise TypeError("extract_schema là bắt buộc")

        # Parse schema thành format có cấu trúc
        extract_fields = [
            ExtractField.from_string(schema_str)
            for schema_str in extract_schema
        ]

        # Xây dựng schemas
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
        """Tạo parser function dựa trên format."""
        if self.format == "json":
            return parse_json
        elif self.format == "xml":
            return parse_xml
        elif self.format == "yaml":
            return parse_yaml
        elif self.format == "regex":
            if not self.pattern:
                raise ValueError("Pattern là bắt buộc cho regex parser")

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


if __name__ == "__main__":
    import asyncio
    from hush.core.states import StateSchema, MemoryState
    from hush.core.nodes import GraphNode, START, END, PARENT

    def test(name: str, condition: bool):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        if not condition:
            raise AssertionError(f"Test failed: {name}")

    async def main():
        # =====================================================================
        # Test 1: JSON parser in a graph with state
        # =====================================================================
        print("\n" + "=" * 50)
        print("Test 1: JSON Parser in graph with state")
        print("=" * 50)

        with GraphNode(name="json_workflow") as graph:
            json_parser = ParserNode(
                name="json_parser",
                format="json",
                extract_schema=[
                    "user.name",
                    "user.age",
                    "status",
                ],
                inputs={"text": PARENT["text"]}
            )
            START >> json_parser >> END

        graph.build()
        schema = StateSchema(graph)

        json_text = '{"user": {"name": "John", "age": 30}, "status": "active"}'
        state = MemoryState(schema, inputs={"text": json_text})

        result = await json_parser.run(state)
        test("JSON: name extracted", result["name"] == "John")
        test("JSON: age extracted", result["age"] == 30)
        test("JSON: status extracted", result["status"] == "active")

        # Verify state was updated
        test("state has name", state["json_workflow.json_parser", "name", None] == "John")

        # =====================================================================
        # Test 2: XML parser in graph with state
        # =====================================================================
        print("\n" + "=" * 50)
        print("Test 2: XML Parser in graph with state")
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

        with GraphNode(name="xml_workflow") as graph2:
            xml_parser = ParserNode(
                name="xml_parser",
                format="xml",
                extract_schema=[
                    "response.user.name",
                    "response.user.email",
                    "response.code",
                ],
                inputs={"text": PARENT["text"]}
            )
            START >> xml_parser >> END

        graph2.build()
        schema2 = StateSchema(graph2)
        state2 = MemoryState(schema2, inputs={"text": xml_text})

        result2 = await xml_parser.run(state2)
        test("XML: name extracted", result2["name"] == "Alice")
        test("XML: email extracted", result2["email"] == "alice@example.com")
        test("XML: code extracted", result2["code"] == "200")

        # =====================================================================
        # Test 3: Key-Value parser in graph with state
        # =====================================================================
        print("\n" + "=" * 50)
        print("Test 3: Key-Value Parser in graph with state")
        print("=" * 50)

        kv_text = """name=Bob
age=25
city=New York"""

        with GraphNode(name="kv_workflow") as graph3:
            kv_parser = ParserNode(
                name="kv_parser",
                format="key_value",
                separator="=",
                extract_schema=["name", "age", "city"],
                inputs={"text": PARENT["text"]}
            )
            START >> kv_parser >> END

        graph3.build()
        schema3 = StateSchema(graph3)
        state3 = MemoryState(schema3, inputs={"text": kv_text})

        result3 = await kv_parser.run(state3)
        test("KV: name extracted", result3["name"] == "Bob")
        test("KV: age extracted", result3["age"] == "25")
        test("KV: city extracted", result3["city"] == "New York")

        # =====================================================================
        # Test 4: Regex parser in graph with state
        # =====================================================================
        print("\n" + "=" * 50)
        print("Test 4: Regex Parser in graph with state")
        print("=" * 50)

        regex_text = "User: john_doe, Email: john@example.com, Score: 95"

        with GraphNode(name="regex_workflow") as graph4:
            regex_parser = ParserNode(
                name="regex_parser",
                format="regex",
                pattern=r"User: (?P<username>\w+), Email: (?P<email>[\w@.]+), Score: (?P<score>\d+)",
                extract_schema=["username", "email", "score"],
                inputs={"text": PARENT["text"]}
            )
            START >> regex_parser >> END

        graph4.build()
        schema4 = StateSchema(graph4)
        state4 = MemoryState(schema4, inputs={"text": regex_text})

        result4 = await regex_parser.run(state4)
        test("Regex: username extracted", result4["username"] == "john_doe")
        test("Regex: email extracted", result4["email"] == "john@example.com")
        test("Regex: score extracted", result4["score"] == "95")

        # =====================================================================
        # Test 5: Schema extraction
        # =====================================================================
        print("\n" + "=" * 50)
        print("Test 5: Schema extraction")
        print("=" * 50)

        test("json_parser has 'text' in input_schema", "text" in json_parser.input_schema)
        test("json_parser has 'name' in output_schema", "name" in json_parser.output_schema)
        test("json_parser has 'age' in output_schema", "age" in json_parser.output_schema)
        test("json_parser has 'status' in output_schema", "status" in json_parser.output_schema)

        # =====================================================================
        # Test 6: Quick __call__ test
        # =====================================================================
        print("\n" + "=" * 50)
        print("Test 6: Quick __call__ test")
        print("=" * 50)

        quick_parser = ParserNode(
            name="quick_json",
            format="json",
            extract_schema=["name", "age"]
        )
        result5 = quick_parser(text='{"name": "Bob", "age": 25}')
        test("quick call: name", result5["name"] == "Bob")
        test("quick call: age", result5["age"] == 25)

        # =====================================================================
        # Summary
        # =====================================================================
        print("\n" + "=" * 50)
        print("All ParserNode tests passed!")
        print("=" * 50)

    asyncio.run(main())
