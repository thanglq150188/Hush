"""Tests for ParserNode - text parsing and extraction node."""

import pytest
from hush.core.nodes.transform.parser_node import ParserNode
from hush.core.nodes.graph.graph_node import GraphNode
from hush.core.nodes.base import START, END, PARENT
from hush.core.states import StateSchema, MemoryState


# ============================================================
# Test 1: JSON Parser
# ============================================================

class TestJSONParser:
    """Test JSON format parsing."""

    @pytest.mark.asyncio
    async def test_json_parser_in_graph(self):
        """Test JSON parser within a graph context."""
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
        assert result["name"] == "John"
        assert result["age"] == 30
        assert result["status"] == "active"

    @pytest.mark.asyncio
    async def test_json_parser_updates_state(self):
        """Test that parser updates state correctly."""
        with GraphNode(name="json_workflow") as graph:
            json_parser = ParserNode(
                name="json_parser",
                format="json",
                extract_schema=["user.name"],
                inputs={"text": PARENT["text"]}
            )
            START >> json_parser >> END

        graph.build()
        schema = StateSchema(graph)

        json_text = '{"user": {"name": "John"}}'
        state = MemoryState(schema, inputs={"text": json_text})

        await json_parser.run(state)
        assert state["json_workflow.json_parser", "name", None] == "John"

    def test_json_parser_quick_call(self):
        """Test JSON parser with direct __call__."""
        parser = ParserNode(
            name="quick_json",
            format="json",
            extract_schema=["name", "age"]
        )
        result = parser(text='{"name": "Bob", "age": 25}')
        assert result["name"] == "Bob"
        assert result["age"] == 25


# ============================================================
# Test 2: XML Parser
# ============================================================

class TestXMLParser:
    """Test XML format parsing."""

    @pytest.mark.asyncio
    async def test_xml_parser_in_graph(self):
        """Test XML parser within a graph context."""
        xml_text = """
        <response>
            <user>
                <name>Alice</name>
                <email>alice@example.com</email>
            </user>
            <code>200</code>
        </response>
        """

        with GraphNode(name="xml_workflow") as graph:
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

        graph.build()
        schema = StateSchema(graph)
        state = MemoryState(schema, inputs={"text": xml_text})

        result = await xml_parser.run(state)
        assert result["name"] == "Alice"
        assert result["email"] == "alice@example.com"
        assert result["code"] == "200"


# ============================================================
# Test 3: Key-Value Parser
# ============================================================

class TestKeyValueParser:
    """Test key-value format parsing."""

    @pytest.mark.asyncio
    async def test_kv_parser_in_graph(self):
        """Test key-value parser within a graph context."""
        kv_text = """name=Bob
age=25
city=New York"""

        with GraphNode(name="kv_workflow") as graph:
            kv_parser = ParserNode(
                name="kv_parser",
                format="key_value",
                separator="=",
                extract_schema=["name", "age", "city"],
                inputs={"text": PARENT["text"]}
            )
            START >> kv_parser >> END

        graph.build()
        schema = StateSchema(graph)
        state = MemoryState(schema, inputs={"text": kv_text})

        result = await kv_parser.run(state)
        assert result["name"] == "Bob"
        assert result["age"] == "25"
        assert result["city"] == "New York"


# ============================================================
# Test 4: Regex Parser
# ============================================================

class TestRegexParser:
    """Test regex format parsing."""

    @pytest.mark.asyncio
    async def test_regex_parser_in_graph(self):
        """Test regex parser within a graph context."""
        regex_text = "User: john_doe, Email: john@example.com, Score: 95"

        with GraphNode(name="regex_workflow") as graph:
            regex_parser = ParserNode(
                name="regex_parser",
                format="regex",
                pattern=r"User: (?P<username>\w+), Email: (?P<email>[\w@.]+), Score: (?P<score>\d+)",
                extract_schema=["username", "email", "score"],
                inputs={"text": PARENT["text"]}
            )
            START >> regex_parser >> END

        graph.build()
        schema = StateSchema(graph)
        state = MemoryState(schema, inputs={"text": regex_text})

        result = await regex_parser.run(state)
        assert result["username"] == "john_doe"
        assert result["email"] == "john@example.com"
        assert result["score"] == "95"


# ============================================================
# Test 5: Schema Extraction
# ============================================================

class TestParserSchemaExtraction:
    """Test automatic schema extraction."""

    def test_parser_has_text_input(self):
        """Test that parser always has 'text' input."""
        parser = ParserNode(
            name="test_parser",
            format="json",
            extract_schema=["name"]
        )
        assert "text" in parser.inputs

    def test_parser_outputs_match_schema(self):
        """Test that outputs match extract_schema."""
        parser = ParserNode(
            name="test_parser",
            format="json",
            extract_schema=["name", "age", "status"]
        )
        assert "name" in parser.outputs
        assert "age" in parser.outputs
        assert "status" in parser.outputs

    def test_parser_nested_schema_outputs(self):
        """Test outputs with nested schema (dot notation)."""
        parser = ParserNode(
            name="test_parser",
            format="json",
            extract_schema=["user.name", "user.email"]
        )
        # Output keys should be the last part of the path
        assert "name" in parser.outputs
        assert "email" in parser.outputs


# ============================================================
# Test 6: Error Handling
# ============================================================

class TestParserErrors:
    """Test parser error handling."""

    def test_missing_extract_schema_raises(self):
        """Test that missing extract_schema raises error."""
        with pytest.raises(TypeError):
            ParserNode(
                name="bad_parser",
                format="json"
            )

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self):
        """Test that invalid JSON returns empty dict and captures error."""
        with GraphNode(name="invalid_json_graph") as graph:
            parser = ParserNode(
                name="test_parser",
                format="json",
                extract_schema=["name"],
                inputs={"text": PARENT["text"]}
            )
            START >> parser >> END

        graph.build()
        schema = StateSchema(graph)
        state = MemoryState(schema, inputs={"text": "not valid json"})
        await parser.run(state)

        # Error should be captured in state
        error = state["invalid_json_graph.test_parser", "error", None]
        assert error is not None
        assert "JSON" in error or "json" in error.lower()
