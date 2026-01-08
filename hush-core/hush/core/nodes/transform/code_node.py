"""Code execution node for running Python functions in workflows."""

from typing import Dict, Callable, Any, Optional, List
import asyncio
import inspect
import ast
from functools import wraps

from hush.core.nodes.base import BaseNode
from hush.core.configs.node_config import NodeType
from hush.core.utils.common import Param


def code_node(func):
    """
    Decorator that converts a function into a CodeNode factory.

    Usage:
        @code_node
        def my_function(arg1, arg2):
            return {"result": value}

        node = my_function(inputs={"arg1": PARENT, "arg2": "value"})
    """
    @wraps(func)
    def wrapper(inputs=None, name=None, **kwargs):
        node_name = name or func.__name__
        if node_name.endswith("_fn"):
            node_name = node_name[:-3]

        return CodeNode(
            name=node_name,
            code_fn=func,
            inputs=inputs or {},
            **kwargs
        )

    wrapper.__wrapped__ = func
    return wrapper


def ensure_async(func: Callable) -> Callable:
    """Ensure a function is async."""
    if asyncio.iscoroutinefunction(func):
        return func

    async def async_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return async_wrapper


TYPE_MAP = {
    "str": str, "string": str,
    "int": int, "integer": int,
    "float": float,
    "bool": bool, "boolean": bool,
    "list": list, "List": list,
    "dict": dict, "Dict": dict,
    "any": Any, "Any": Any,
}


def parse_default_value(value_str: str, param_type: type) -> Any:
    """Parse a default value string into the appropriate type.

    Returns None if parsing fails.
    """
    value_str = value_str.strip()

    # Handle empty string
    if not value_str:
        if param_type == list:
            return []
        elif param_type == dict:
            return {}
        elif param_type == str:
            return ""
        return None

    try:
        if param_type == bool:
            return value_str.lower() in ("true", "1", "yes")
        elif param_type == int:
            return int(value_str)
        elif param_type == float:
            return float(value_str)
        elif param_type == list:
            return []
        elif param_type == dict:
            return {}
        else:
            return value_str  # str or Any
    except (ValueError, TypeError):
        return None


def parse_comment(comment: str) -> tuple:
    """Parse comment like '(type) description' or '(type=default) description'.

    Only recognizes known types in parentheses at the START of the comment.

    Examples:
        '(str) greeting message' -> (str, None, 'greeting message')
        '(int=0) the count' -> (int, 0, 'the count')
        '(bool=true) enabled flag' -> (bool, True, 'enabled flag')
        'just a description' -> (Any, None, 'just a description')
        'use (carefully) here' -> (Any, None, 'use (carefully) here')
    """
    comment = comment.strip()
    default = None

    if comment.startswith("(") and ")" in comment:
        close_idx = comment.index(")")
        type_part = comment[1:close_idx].strip()

        # Check for default value: (type=default)
        if "=" in type_part:
            type_str, default_str = type_part.split("=", 1)
            type_str = type_str.strip()
        else:
            type_str = type_part
            default_str = None

        # Only treat as type if it's a known type
        if type_str in TYPE_MAP:
            param_type = TYPE_MAP[type_str]
            description = comment[close_idx + 1:].strip()

            # Parse default value if provided
            if default_str is not None:
                default = parse_default_value(default_str, param_type)

            return param_type, default, description

    # No valid type found, treat entire comment as description
    return Any, None, comment


def _extract_dict_keys(dict_node: ast.Dict, source_lines: List[str]) -> Dict[str, Param]:
    """Extract keys from an AST Dict node."""
    schema = {}
    for key_node in dict_node.keys:
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            key_name = key_node.value

            # Extract type, default, and description from inline comment
            param_type = Any
            default = None
            description = ""
            for src_line in source_lines:
                # Check for both single and double quotes
                if "#" in src_line and (f'"{key_name}"' in src_line or f"'{key_name}'" in src_line):
                    comment = src_line.split("#", 1)[1].strip()
                    param_type, default, description = parse_comment(comment)
                    break

            schema[key_name] = Param(
                type=param_type,
                default=default,
                description=description
            )
    return schema


def extract_return_schema(func: Callable) -> Dict[str, Param]:
    """Extract return schema from function source code using AST.

    Supports:
        # Regular functions
        return {"key": value}
        return {"key": value,  # description}
        return {"key": value,  # (type) description}

        # Lambda functions
        lambda x: {"key": value}
        lambda x: {"key": value,  # (type) description}
    """
    try:
        source = inspect.getsource(func)
        source_lines = source.splitlines()
        source = inspect.cleandoc(source)
        tree = ast.parse(source)

        schema = {}
        for node in ast.walk(tree):
            # Handle regular function return statements
            if isinstance(node, ast.Return) and node.value:
                if isinstance(node.value, ast.Dict):
                    schema.update(_extract_dict_keys(node.value, source_lines))

            # Handle lambda expressions (body is the return value)
            elif isinstance(node, ast.Lambda):
                if isinstance(node.body, ast.Dict):
                    schema.update(_extract_dict_keys(node.body, source_lines))

        return schema
    except Exception:
        return {}


class CodeNode(BaseNode):
    """Node that executes a Python function."""

    type: NodeType = "code"

    __slots__ = ["code_fn", "source"]

    def __init__(
        self,
        code_fn: Optional[Callable] = None,
        return_keys: Optional[List[str]] = None,
        **kwargs
    ):
        # Build schemas before super().__init__
        input_schema, output_schema = self._build_schemas(code_fn, return_keys)

        super().__init__(
            input_schema=input_schema,
            output_schema=output_schema,
            **kwargs
        )

        self.code_fn = code_fn
        self.core = ensure_async(code_fn) if code_fn else None

        # Get source code
        try:
            self.source = inspect.getsource(code_fn) if code_fn else ""
        except:
            self.source = str(code_fn) if code_fn else ""

        # Set description from docstring if not provided
        if not self.description and code_fn and code_fn.__doc__:
            self.description = code_fn.__doc__.strip().split('\n')[0]

    def _build_schemas(
        self,
        code_fn: Optional[Callable],
        return_keys: Optional[List[str]]
    ) -> tuple:
        """Build input/output schemas from function signature and source."""
        if code_fn is None:
            return {}, {}

        # Build input schema from function parameters
        input_schema = {}
        sig = inspect.signature(code_fn)

        for param_name, param in sig.parameters.items():
            param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
            has_default = param.default != inspect.Parameter.empty
            default_val = param.default if has_default else None

            input_schema[param_name] = Param(
                type=param_type,
                required=not has_default,
                default=default_val
            )

        # Build output schema: explicit return_keys > AST parsing
        if return_keys:
            output_schema = {key: Param(type=Any) for key in return_keys}
        else:
            # Parse return schema from source code (with type hints and descriptions)
            output_schema = extract_return_schema(code_fn)

        return input_schema, output_schema

    def specific_metadata(self) -> Dict[str, Any]:
        """Return subclass-specific metadata."""
        return {
            "code_fn": self.source[:200] + "..." if len(self.source) > 200 else self.source,
            "function_name": self.code_fn.__name__ if self.code_fn else "unknown"
        }


if __name__ == "__main__":
    def test(name: str, condition: bool):
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {name}")
        if not condition:
            raise AssertionError(f"Test failed: {name}")

    @code_node
    def add_numbers(a: int, b: int = 10):
        """Add two numbers."""
        return {"result": a + b}  # (int) the sum

    @code_node
    def process_data(name: str, count: int):
        """Process data with type hints and descriptions."""
        return {
            "message": f"Hello, {name}!",  # (str) greeting message
            "total": count * 2,  # (int) doubled count
            "status": "success",  # just a description, type defaults to Any
        }

    @code_node
    def increment(x: int):
        return {"x": x + 1}

    # =========================================================================
    # Test 1: Basic __call__ usage
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 1: Basic __call__ usage")
    print("=" * 50)

    node = add_numbers()
    result = node(a=10, b=20)
    test("add_numbers(a=10, b=20) = 30", result == {"result": 30})

    result = node(a=5, b=3)
    test("add_numbers(a=5, b=3) = 8", result == {"result": 8})

    # With default value
    result = node(a=7)
    test("add_numbers(a=7) uses default b=10", result == {"result": 17})

    # =========================================================================
    # Test 2: Increment function
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 2: Increment function")
    print("=" * 50)

    inc = increment()
    result = inc(x=5)
    test("increment(x=5) = 6", result == {"x": 6})

    result = inc(x=0)
    test("increment(x=0) = 1", result == {"x": 1})

    result = inc(x=-1)
    test("increment(x=-1) = 0", result == {"x": 0})

    # =========================================================================
    # Test 3: Process data with multiple outputs
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 3: Process data with multiple outputs")
    print("=" * 50)

    proc = process_data()
    result = proc(name="World", count=5)
    test("message is 'Hello, World!'", result["message"] == "Hello, World!")
    test("total is 10", result["total"] == 10)
    test("status is 'success'", result["status"] == "success")

    result = proc(name="Alice", count=3)
    test("message is 'Hello, Alice!'", result["message"] == "Hello, Alice!")
    test("total is 6", result["total"] == 6)

    # =========================================================================
    # Test 4: Schema extraction
    # =========================================================================
    print("\n" + "=" * 50)
    print("Test 4: Schema extraction")
    print("=" * 50)

    node = add_numbers()
    test("input_schema has 'a'", "a" in node.input_schema)
    test("input_schema has 'b'", "b" in node.input_schema)
    test("output_schema has 'result'", "result" in node.output_schema)
    test("'a' is required", node.input_schema["a"].required == True)
    test("'b' has default 10", node.input_schema["b"].default == 10)

    proc = process_data()
    test("output has 'message'", "message" in proc.output_schema)
    test("output has 'total'", "total" in proc.output_schema)
    test("output has 'status'", "status" in proc.output_schema)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 50)
    print("All CodeNode tests passed!")
    print("=" * 50)
